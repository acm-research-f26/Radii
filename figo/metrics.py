"""Overlap and boundary metrics at native voxel spacing.

Conventions: Dice is NaN when a structure is absent from both masks. Distance
metrics are NaN when absent from both, and ``inf`` when present in only one
(reported separately as a detection failure rather than silently averaged).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .labels import BACKGROUND, CAVITY, CYST, LABEL_NAMES, MYOMA, WALL

_FACE = ndimage.generate_binary_structure(3, 1)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, _FACE, border_value=0)


def _directed(a_pts: np.ndarray, b_pts: np.ndarray, spacing) -> np.ndarray:
    dt = ndimage.distance_transform_edt(~b_pts, sampling=spacing)
    return dt[a_pts]


def point_set_distances(a_pts: np.ndarray, b_pts: np.ndarray, spacing):
    """Symmetric distances between two voxel point sets (boolean masks)."""
    if not a_pts.any() and not b_pts.any():
        return None
    if not a_pts.any() or not b_pts.any():
        return np.array([np.inf])
    return np.concatenate([_directed(a_pts, b_pts, spacing), _directed(b_pts, a_pts, spacing)])


def hd95_assd(a: np.ndarray, b: np.ndarray, spacing, already_surface=False):
    sa, sb = (a, b) if already_surface else (surface(a), surface(b))
    d = point_set_distances(sa, sb, spacing)
    if d is None:
        return float("nan"), float("nan")
    if np.isinf(d).any():
        return float("inf"), float("inf")
    return float(np.percentile(d, 95)), float(d.mean())


def cavity_wall_interface(seg: np.ndarray) -> np.ndarray:
    """Endometrial surface: cavity voxels touching myometrium or fibroid.
    This boundary decides submucosal vs intramural (FIGO 0-3)."""
    tissue = (seg == WALL) | (seg == MYOMA)
    return (seg == CAVITY) & ndimage.binary_dilation(tissue, _FACE)


def serosal_surface(seg: np.ndarray) -> np.ndarray:
    """Outer uterine surface: uterine voxels touching background.
    This boundary decides intramural vs subserosal (FIGO 4-7)."""
    uterus = np.isin(seg, (WALL, CAVITY, MYOMA))
    return uterus & ndimage.binary_dilation(seg == BACKGROUND, _FACE)


def case_metrics(pred: np.ndarray, gt: np.ndarray, spacing) -> dict:
    row = {}
    for lab, name in LABEL_NAMES.items():
        p, g = pred == lab, gt == lab
        row[f"dice_{name}"] = dice(p, g)
        if lab != CYST:  # cysts are tiny/rare; distances are dominated by misses
            row[f"hd95_{name}"], row[f"assd_{name}"] = hd95_assd(p, g, spacing)
    for name, fn in (("cavity_wall_interface", cavity_wall_interface), ("serosa", serosal_surface)):
        row[f"hd95_{name}"], row[f"assd_{name}"] = hd95_assd(fn(pred), fn(gt), spacing, already_surface=True)
    row["dice_uterus"] = dice(np.isin(pred, (WALL, CAVITY, MYOMA)), np.isin(gt, (WALL, CAVITY, MYOMA)))
    return row
