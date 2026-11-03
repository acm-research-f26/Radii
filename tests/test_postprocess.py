"""Post-processing of predicted masks and the FIGO error report."""

import csv

import nibabel as nib
import numpy as np

from figo.figo_eval import agreement_case_full, error_report
from figo.labels import CAVITY, CYST, MYOMA, WALL
from figo.postprocess import PostprocessConfig, postprocess

ANISO = (0.5, 0.5, 4.0)  # thick slices on the last axis, like UMD


def _uterus():
    seg = np.zeros((60, 60, 10), np.uint8)
    seg[10:50, 10:50, 2:8] = WALL
    seg[25:35, 25:35, 4:6] = CAVITY
    seg[12:20, 12:20, 3:6] = MYOMA
    return seg


def test_keeps_largest_uterus_and_drops_stray_blobs():
    seg = _uterus()
    seg[52:56, 52:56, 2:4] = WALL  # stray false-positive blob
    seg[2:6, 50:56, 1:3] = MYOMA  # fibroid floating outside the uterus
    seg[2:5, 2:5, 0:2] = CYST  # cysts are never touched
    out, stats = postprocess(seg, ANISO)
    assert not (out[52:56, 52:56, 2:4] == WALL).any()
    assert not (out[2:6, 50:56, 1:3] == MYOMA).any()
    assert (out[12:20, 12:20, 3:6] == MYOMA).all()  # attached fibroid kept
    assert (out[2:5, 2:5, 0:2] == CYST).all()
    assert stats["removed_components_vox"] > 0


def test_drops_detached_fibroid_when_largest_is_off():
    seg = _uterus()
    seg[2:6, 50:56, 1:3] = MYOMA
    out, stats = postprocess(seg, ANISO, PostprocessConfig(keep_largest_uterus=False, fill_wall_holes=False))
    assert not (out[2:6, 50:56, 1:3] == MYOMA).any()
    assert stats["removed_fibroid_vox"] == 4 * 6 * 2


def test_fills_small_wall_holes_only():
    seg = _uterus()
    seg[40:42, 40:42, 3] = 0  # small hole: 4 px * 0.25 mm2 = 1 mm2
    seg[38:48, 12:22, 6] = 0  # big hole: 100 px * 0.25 = 25 mm2
    out, _ = postprocess(seg, ANISO, PostprocessConfig(max_hole_mm2=10))
    assert (out[40:42, 40:42, 3] == WALL).all()
    assert (out[38:48, 12:22, 6] == 0).all()


def test_noop_on_clean_mask():
    seg = _uterus()
    out, stats = postprocess(seg, ANISO)
    np.testing.assert_array_equal(out, seg)
    assert sum(stats.values()) == 0


def test_error_report_counts_confusions_and_misses():
    ref = _uterus()
    ref[40:46, 40:46, 3:5] = MYOMA  # second fibroid, missed by the prediction
    pred = _uterus()
    pairs, missed, n_ref, n_pred = agreement_case_full(ref, pred, ANISO)
    assert (n_ref, n_pred, len(pairs), len(missed)) == (2, 1, 1, 1)
    r = error_report(pairs, missed)
    assert r["n_missed"] == 1 and r["recall"] == 0.5
    assert r["missed_thin_le_2_slices"] == 1


def test_postprocess_cli_and_errors_cli(tmp_path):
    from figo.figo_eval import main as figo_main
    from figo.postprocess import main as pp_main

    seg = _uterus()
    seg[52:56, 52:56, 2:4] = WALL
    aff = np.diag([*ANISO, 1])
    (tmp_path / "in").mkdir()
    nib.save(nib.Nifti1Image(seg, aff), tmp_path / "in" / "C1.nii.gz")
    pp_main(["--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out")])
    clean = np.asarray(nib.load(tmp_path / "out" / "C1.nii.gz").dataobj)
    assert clean.dtype == np.uint8 and not (clean[52:56, 52:56, 2:4] == WALL).any()

    agree = tmp_path / "m_figo_agreement.csv"
    with open(agree, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ref_figo", "pred_figo", "ref_volume_ml", "ref_n_slices"])
        w.writeheader()
        w.writerows([{"ref_figo": "3", "pred_figo": "2", "ref_volume_ml": 2, "ref_n_slices": 3},
                     {"ref_figo": "4", "pred_figo": "4", "ref_volume_ml": 5, "ref_n_slices": 4}])
    out = tmp_path / "table.csv"
    figo_main(["errors", "--csvs", str(agree), "--out", str(out)])
    with open(out) as f:
        row = next(csv.DictReader(f))
    assert row["model"] == "m" and "3->2 x1" in row["top_errors"]
