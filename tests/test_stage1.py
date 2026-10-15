import json
import shutil

import nibabel as nib
import numpy as np
import pytest
import torch

from figo import evaluate, figo_eval, infer, nnunet, prepare, train
from figo.io import discover_cases, is_gzip, load_nifti
from figo.labels import CAVITY, MYOMA, WALL
from figo.metrics import case_metrics, dice, hd95_assd
from figo.models import MODELS, build_model
from figo.preprocess import to_native_labels, to_network_space

# sagittal-like: thick axis is L-R (first canonical axis)
AFFINE = np.diag([6.6, 0.9, 0.9, 1.0])


def fake_case(rng, shape=(8, 64, 64)):
    seg = np.zeros(shape, np.uint8)
    c = np.array(shape) // 2 + rng.integers(-3, 4, 3) * [0, 1, 1]
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    d2 = ((g[0] - c[0]) * 6.6) ** 2 + ((g[1] - c[1]) * 0.9) ** 2 + ((g[2] - c[2]) * 0.9) ** 2
    seg[d2 <= 20**2] = WALL
    seg[(d2 <= 8**2)] = CAVITY
    f = c + [0, 14, 0]
    seg[((g[0] - f[0]) * 6.6) ** 2 + ((g[1] - f[1]) * 0.9) ** 2 + ((g[2] - f[2]) * 0.9) ** 2 <= 6**2] = MYOMA
    img = 100 + 50 * seg + rng.normal(0, 5, shape)
    return img.astype(np.float32), seg


@pytest.fixture(scope="module")
def umd(tmp_path_factory):
    root = tmp_path_factory.mktemp("UMD")
    rng = np.random.default_rng(0)
    for i in range(6):
        cid = f"UMD_{i:03d}"
        d = root / cid
        d.mkdir()
        img, seg = fake_case(rng)
        nib.save(nib.Nifti1Image(img, AFFINE), d / f"{cid}_t2.nii.gz")
        seg_nii = nib.Nifti1Image(seg.astype(np.float32), AFFINE)
        if i < 2:  # reproduce the UMD defect: uncompressed NIfTI with a .nii.gz name
            nib.save(seg_nii, d / f"{cid}_seg.nii")
            shutil.move(d / f"{cid}_seg.nii", d / f"{cid}_seg.nii.gz")
        else:
            nib.save(seg_nii, d / f"{cid}_seg.nii.gz")
    return root


def test_defensive_loader(umd):
    bad = umd / "UMD_000" / "UMD_000_seg.nii.gz"
    assert not is_gzip(bad)
    with pytest.raises(Exception):
        np.asarray(nib.load(bad).dataobj)  # what breaks a naive loader
    assert load_nifti(bad).shape == (8, 64, 64)
    assert len(discover_cases(umd)) == 6


def test_preprocess_roundtrip():
    rng = np.random.default_rng(1)
    img, seg = fake_case(rng)
    img_nii, seg_nii = nib.Nifti1Image(img, AFFINE), nib.Nifti1Image(seg, AFFINE)
    x, y, geom = to_network_space(img_nii, seg_nii, target_inplane=0.9)
    assert x.shape == (64, 64, 8) and geom.net_spacing[2] == pytest.approx(6.6)  # thick axis last
    probs = np.eye(5, dtype=np.float32)[y].transpose(3, 0, 1, 2)
    back = to_native_labels(probs, geom)
    np.testing.assert_array_equal(np.asarray(back.dataobj), seg)
    # downsampling in-plane keeps the slice count and restores native shape
    x2, _, g2 = to_network_space(img_nii, None, target_inplane=1.8)
    assert x2.shape == (32, 32, 8)
    back2 = to_native_labels(np.zeros((5,) + x2.shape, np.float32), g2)
    assert back2.shape == seg.shape and np.allclose(back2.affine, AFFINE)


def test_metrics():
    a = np.zeros((20, 20, 20), bool)
    a[5:15, 5:15, 5:15] = True
    b = np.roll(a, 2, axis=0)
    assert dice(a, a) == 1.0
    hd, assd = hd95_assd(a, b, (1.0, 1.0, 1.0))
    assert hd == pytest.approx(2.0)
    hd, _ = hd95_assd(a, b, (3.0, 1.0, 1.0))
    assert hd == pytest.approx(6.0)
    assert np.isnan(dice(np.zeros_like(a), np.zeros_like(a)))
    assert hd95_assd(a, np.zeros_like(a), (1, 1, 1))[0] == np.inf


@pytest.mark.parametrize("name", sorted(MODELS))
def test_models_train_and_eval_shapes(name):
    m = build_model(name)
    x = torch.randn(2, 1, 32, 32, 8)
    m.train()
    out = m(x)
    outs = out if isinstance(out, list) else [out]
    assert outs[0].shape == (2, 5, 32, 32, 8)
    outs[0].mean().backward()
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (2, 5, 32, 32, 8)


def test_end_to_end(umd, tmp_path):
    data = tmp_path / "data"
    prepare.main(["--umd-dir", str(umd), "--out-dir", str(data), "--cache", "--val-frac", "0.2"])
    splits = json.loads((data / "splits.json").read_text())
    assert len(splits["train"]) == 4 and len(splits["val"]) == 1 and len(splits["test"]) == 1
    assert all(is_gzip(p) for p in (data / "clean").glob("*.nii.gz"))

    run = tmp_path / "run"
    targs = ["--model", "unet3d", "--cache-dir", str(data / "cache_0.9mm"), "--splits", str(data / "splits.json"),
             "--out-dir", str(run), "--patch-size", "32", "32", "8", "--epochs", "2", "--iters-per-epoch", "2",
             "--val-every", "1", "--workers", "0"]
    train.main(targs[:-6] + ["--epochs", "1"] + targs[-6:])
    ckpt = torch.load(run / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 0
    train.main(targs)  # resumes from last.pt and runs epoch 1 only
    assert torch.load(run / "last.pt", weights_only=False)["epoch"] == 1
    assert (run / "best.pt").exists()
    assert len((run / "log.csv").read_text().strip().splitlines()) == 3

    preds = tmp_path / "preds"
    infer.main(["--checkpoint", str(run / "best.pt"), "--data-dir", str(data / "clean"),
                "--splits", str(data / "splits.json"), "--subset", "test", "--out-dir", str(preds)])
    test_id = splits["test"][0]
    assert load_nifti(preds / f"{test_id}.nii.gz").shape == (8, 64, 64)

    evaluate.main(["--pred-dir", str(preds), "--data-dir", str(data / "clean"), "--splits",
                   str(data / "splits.json"), "--out", str(tmp_path / "seg.csv")])
    assert (tmp_path / "seg.summary.json").exists()

    # expert masks scored against themselves: perfect FIGO agreement
    gt_as_pred = tmp_path / "gt_as_pred"
    gt_as_pred.mkdir()
    shutil.copyfile(data / "clean" / f"{test_id}_seg.nii.gz", gt_as_pred / f"{test_id}.nii.gz")
    figo_eval.main(["agreement", "--ref-dir", str(data / "clean"), "--pred-dir", str(gt_as_pred),
                    "--splits", str(data / "splits.json"), "--out", str(tmp_path / "agree.csv")])
    s = json.loads((tmp_path / "agree.summary.json").read_text())
    assert s["detection_recall"] == 1.0 and s["exact_agreement"] == 1.0

    nnunet.convert(data / "clean", data / "splits.json", tmp_path / "raw", tmp_path / "pre")
    ds = tmp_path / "raw" / nnunet.DATASET_NAME
    assert len(list((ds / "imagesTr").glob("*_0000.nii.gz"))) == 5
    assert json.loads((ds / "dataset.json").read_text())["numTraining"] == 5


def test_case_metrics_keys():
    _, seg = fake_case(np.random.default_rng(2))
    row = case_metrics(seg, seg, (6.6, 0.9, 0.9))
    assert row["dice_wall"] == 1.0 and row["hd95_cavity_wall_interface"] == 0.0


def test_stage3_and_selection_cli(umd, tmp_path, capsys):
    from figo import select_models, stage3

    data = tmp_path / "data"
    prepare.main(["--umd-dir", str(umd), "--out-dir", str(data), "--val-frac", "0.2"])
    stage3.main(["--seg-dir", str(data / "clean"), "--suffix", "_seg", "--out", str(tmp_path / "figo.csv")])
    figo_eval.main(["distribution", "--csv", str(tmp_path / "figo.csv")])
    assert "fibroids" in capsys.readouterr().out

    res = tmp_path / "results"
    res.mkdir()
    for m, d in (("a", 0.9), ("b", 0.5)):
        seg = {k: {"mean": v} for k, v in (("dice_cavity", d), ("dice_wall", d), ("dice_myoma", d),
                                           ("hd95_cavity_wall_interface", 10 * (1 - d)), ("hd95_serosa", 1.0))}
        (res / f"{m}_seg.summary.json").write_text(json.dumps(seg))
        (res / f"{m}_figo_agreement.summary.json").write_text(json.dumps(
            {"kappa_coarse": d, "exact_agreement": d, "detection_recall": d}))
    select_models.main(["--results-dir", str(res)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln[:2] in ("a ", "b ")]
    assert lines[0].startswith("a ")
