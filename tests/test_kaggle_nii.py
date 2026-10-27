"""Kaggle unpacks uploaded .nii.gz files to .nii; the pipeline must still find them."""

import gzip

import nibabel as nib
import numpy as np
import pytest

from figo.io import is_gzip, load_nifti, resolve_nifti
from figo.labels import MYOMA, WALL


def _save(path, data, gz):
    img = nib.Nifti1Image(data, np.diag([0.45, 0.45, 6.6, 1]))
    nib.save(img, path)
    assert is_gzip(path) == gz


def test_gz_path_falls_back_to_plain_nii(tmp_path):
    seg = np.zeros((8, 8, 4), np.uint8)
    seg[2:5, 2:5, 1:3] = MYOMA
    _save(tmp_path / "C1_seg.nii", seg, gz=False)
    assert resolve_nifti(tmp_path / "C1_seg.nii.gz") == tmp_path / "C1_seg.nii"
    np.testing.assert_array_equal(np.asarray(load_nifti(tmp_path / "C1_seg.nii.gz").dataobj), seg)


def test_plain_path_falls_back_to_gz(tmp_path):
    _save(tmp_path / "C2_t2.nii.gz", np.ones((4, 4, 2), np.float32), gz=True)
    assert resolve_nifti(tmp_path / "C2_t2.nii") == tmp_path / "C2_t2.nii.gz"


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_nifti(tmp_path / "nope_seg.nii.gz")


def test_stage3_discovers_plain_nii(tmp_path):
    from figo.stage3 import main

    seg = np.zeros((40, 40, 6), np.uint8)
    seg[5:35, 5:35, 1:5] = WALL
    seg[15:25, 15:25, 2:4] = MYOMA
    _save(tmp_path / "C3_seg.nii", seg, gz=False)
    out = tmp_path / "figo.csv"
    main(["--seg-dir", str(tmp_path), "--suffix", "_seg", "--out", str(out)])
    assert "C3" in out.read_text()


def test_nnunet_convert_gzips_plain_nii(tmp_path):
    from figo.nnunet import _copy_gz

    _save(tmp_path / "C4_seg.nii", np.zeros((4, 4, 2), np.uint8), gz=False)
    dst = tmp_path / "out.nii.gz"
    _copy_gz(tmp_path / "C4_seg.nii.gz", dst)
    assert is_gzip(dst)
    gzip.decompress(dst.read_bytes())  # valid gzip stream
