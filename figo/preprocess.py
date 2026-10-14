"""Geometry handling for the custom (non-nnU-Net) models.

UMD volumes are sagittal with ~0.45 x 0.45 mm in-plane and ~6.6 mm slices.
Network space is defined as:
  1. reorient to closest canonical (RAS) orientation,
  2. transpose so the thick (largest-spacing) axis is last -> arrays are (H, W, D),
  3. resample in-plane only to ``target_inplane`` mm (slices are never interpolated),
  4. clip to the 0.5/99.5 intensity percentiles and z-score.

``to_native_labels`` inverts steps 1-3 for a predicted probability map so that
predictions can be scored against the expert masks at native resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import nibabel as nib
import numpy as np
from scipy import ndimage

DEFAULT_INPLANE_MM = 0.9


@dataclass
class Geometry:
    canonical_affine: np.ndarray
    canonical_shape: tuple[int, int, int]
    perm: tuple[int, int, int]  # canonical axes -> network axes
    native_spacing: tuple[float, float, float]  # after perm, before resampling
    net_spacing: tuple[float, float, float]
    net_shape: tuple[int, int, int]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["canonical_affine"] = np.asarray(self.canonical_affine).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(
            canonical_affine=np.asarray(d["canonical_affine"], dtype=float),
            canonical_shape=tuple(int(x) for x in d["canonical_shape"]),
            perm=tuple(int(x) for x in d["perm"]),
            native_spacing=tuple(float(x) for x in d["native_spacing"]),
            net_spacing=tuple(float(x) for x in d["net_spacing"]),
            net_shape=tuple(int(x) for x in d["net_shape"]),
        )


def thick_axis_last(zooms) -> tuple[int, int, int]:
    thick = int(np.argmax(zooms))
    return tuple([a for a in range(3) if a != thick] + [thick])


def _resize(arr: np.ndarray, shape, order: int) -> np.ndarray:
    if tuple(arr.shape) == tuple(shape):
        return arr
    factors = [n / o for n, o in zip(shape, arr.shape)]
    out = ndimage.zoom(arr, factors, order=order, mode="nearest", grid_mode=True)
    assert out.shape == tuple(shape), (out.shape, shape)
    return out


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [0.5, 99.5])
    image = np.clip(image, lo, hi)
    std = image.std()
    return ((image - image.mean()) / (std if std > 0 else 1.0)).astype(np.float32)


def to_network_space(
    image_nii: nib.Nifti1Image,
    seg_nii: nib.Nifti1Image | None = None,
    target_inplane: float = DEFAULT_INPLANE_MM,
):
    """Return ``(image, seg_or_None, Geometry)`` in network space."""
    can = nib.as_closest_canonical(image_nii)
    zooms = np.asarray(can.header.get_zooms()[:3], dtype=float)
    perm = thick_axis_last(zooms)
    img = np.asarray(can.dataobj, dtype=np.float32).transpose(perm)
    spacing = zooms[list(perm)]
    net_shape = (
        max(1, int(round(img.shape[0] * spacing[0] / target_inplane))),
        max(1, int(round(img.shape[1] * spacing[1] / target_inplane))),
        img.shape[2],
    )
    net_spacing = (
        spacing[0] * img.shape[0] / net_shape[0],
        spacing[1] * img.shape[1] / net_shape[1],
        spacing[2],
    )
    geom = Geometry(
        canonical_affine=can.affine,
        canonical_shape=tuple(int(s) for s in can.shape[:3]),
        perm=perm,
        native_spacing=tuple(float(s) for s in spacing),
        net_spacing=tuple(float(s) for s in net_spacing),
        net_shape=net_shape,
    )
    img = normalize(_resize(img, net_shape, order=1))

    seg = None
    if seg_nii is not None:
        seg_can = nib.as_closest_canonical(seg_nii)
        if not np.allclose(seg_can.affine, can.affine, atol=1e-3):
            raise ValueError("image and segmentation affines differ; run figo.prepare first")
        seg = np.asarray(seg_can.dataobj).astype(np.uint8).transpose(perm)
        seg = _resize(seg, net_shape, order=0)
    return img, seg, geom


def to_native_labels(probs: np.ndarray, geom: Geometry) -> nib.Nifti1Image:
    """Map class probabilities ``(C, H, W, D)`` in network space to a label image
    in canonical orientation at native resolution."""
    native_net_shape = tuple(geom.canonical_shape[a] for a in geom.perm)
    up = np.stack([_resize(p.astype(np.float32), native_net_shape, order=1) for p in probs])
    labels = up.argmax(0).astype(np.uint8)
    labels = labels.transpose(np.argsort(geom.perm))
    out = nib.Nifti1Image(labels, geom.canonical_affine)
    out.set_data_dtype(np.uint8)
    return out
