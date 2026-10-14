"""Stage 2: split the semantic myoma label into individual fibroids.

Connectivity is anisotropy-aware: full 8-connectivity within a slice, but only
face (direct) neighbours across slices, so two fibroids that merely touch
diagonally across a 6.6 mm gap are not merged. Fibroids that physically touch
within a slice remain one component; this is a known limitation of CC-based
separation and is flagged downstream via the component's size.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .labels import MYOMA


def anisotropic_structure(spacing) -> np.ndarray:
    thick = int(np.argmax(spacing))
    s = np.zeros((3, 3, 3), dtype=bool)
    centre = [slice(None)] * 3
    centre[thick] = 1
    s[tuple(centre)] = True  # full 3x3 neighbourhood in the slice plane
    idx = [1, 1, 1]
    for k in (0, 2):
        idx[thick] = k
        s[tuple(idx)] = True  # direct neighbour in the adjacent slices
    return s


def fibroid_instances(seg: np.ndarray, spacing, min_volume_mm3: float = 50.0):
    """Label individual fibroids.

    Returns ``(instance_map, info)`` where ``instance_map`` is int32 with
    0 = not a fibroid and 1..N = fibroid id (sorted by decreasing volume), and
    ``info`` is a list of dicts (id, n_voxels, volume_mm3, n_slices, bbox).
    Components smaller than ``min_volume_mm3`` are dropped.
    """
    spacing = np.asarray(spacing, dtype=float)
    voxel_mm3 = float(np.prod(spacing))
    cc, n = ndimage.label(seg == MYOMA, structure=anisotropic_structure(spacing))
    if n == 0:
        return np.zeros(seg.shape, np.int32), []
    sizes = np.bincount(cc.ravel(), minlength=n + 1)[1:]
    keep = [i + 1 for i in np.argsort(-sizes) if sizes[i] * voxel_mm3 >= min_volume_mm3]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    inst = remap[cc]

    thick = int(np.argmax(spacing))
    info = []
    for new_id, sl in enumerate(ndimage.find_objects(inst), start=1):
        if sl is None:
            continue
        m = inst[sl] == new_id
        nvox = int(m.sum())
        other = tuple(a for a in range(3) if a != thick)
        info.append(
            {
                "id": new_id,
                "n_voxels": nvox,
                "volume_mm3": nvox * voxel_mm3,
                "n_slices": int(m.any(axis=other).sum()),
                "bbox": [(s.start, s.stop) for s in sl],
            }
        )
    return inst, info
