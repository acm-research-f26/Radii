"""Optional post-processing of predicted label maps (CPU only, no retraining).

Three cleanup steps, each switchable, applied in this order:

1. ``keep_largest_uterus``: keep only the largest connected uterus component
   (wall + cavity + myoma); any other uterus-label blob becomes background.
2. ``drop_detached_fibroids``: delete myoma components that touch no wall or
   cavity voxel (fibroid blobs lying outside the uterus). With step 1 on, this
   mostly matters when the largest component is itself an isolated fibroid.
3. ``fill_wall_holes``: per slice, fill background holes enclosed by the uterus
   that are smaller than ``max_hole_mm2`` with wall. Done in-plane because the
   slices are 6.6 mm thick.

Caveat: steps 1 and 2 also delete genuinely detached fibroids (FIGO 8,
parasitic). Report FIGO 8 recall separately when post-processing is on.
Nabothian cysts (label 4) are never changed.

    python -m figo.postprocess --in-dir preds/mednext --out-dir preds_pp/mednext \
        --splits data/splits.json --subset val
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .instances import anisotropic_structure
from .io import load_nifti
from .labels import CAVITY, MYOMA, WALL

UTERUS = (WALL, CAVITY, MYOMA)


@dataclass
class PostprocessConfig:
    keep_largest_uterus: bool = True
    drop_detached_fibroids: bool = True
    fill_wall_holes: bool = True
    max_hole_mm2: float = 50.0


def _largest_component(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    cc, n = ndimage.label(mask, structure=structure)
    if n <= 1:
        return mask
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == int(np.argmax(sizes))


def postprocess(seg: np.ndarray, spacing, cfg: PostprocessConfig | None = None) -> tuple[np.ndarray, dict]:
    """Return ``(cleaned_seg, stats)``; ``stats`` counts voxels changed by each step."""
    cfg = cfg or PostprocessConfig()
    spacing = tuple(float(s) for s in spacing)
    out = seg.copy()
    struct = anisotropic_structure(spacing)
    stats = {"removed_components_vox": 0, "removed_fibroid_vox": 0, "filled_hole_vox": 0}

    if cfg.keep_largest_uterus:
        uterus = np.isin(out, UTERUS)
        if uterus.any():
            keep = _largest_component(uterus, struct)
            drop = uterus & ~keep
            stats["removed_components_vox"] = int(drop.sum())
            out[drop] = 0

    if cfg.drop_detached_fibroids:
        myoma = out == MYOMA
        if myoma.any():
            cc, _ = ndimage.label(myoma, structure=struct)
            tissue = np.isin(out, (WALL, CAVITY))
            near = ndimage.binary_dilation(tissue, structure=struct)
            touching = np.unique(cc[near & myoma])
            detached = myoma & ~np.isin(cc, touching[touching > 0])
            stats["removed_fibroid_vox"] = int(detached.sum())
            out[detached] = 0

    if cfg.fill_wall_holes:
        thick = int(np.argmax(spacing))
        inplane = [s for i, s in enumerate(spacing) if i != thick]
        max_vox = cfg.max_hole_mm2 / (inplane[0] * inplane[1])
        uterus = np.isin(out, UTERUS)
        for k in range(out.shape[thick]):
            idx = [slice(None)] * 3
            idx[thick] = k
            idx = tuple(idx)
            sl_u = uterus[idx]
            if not sl_u.any():
                continue
            holes = ndimage.binary_fill_holes(sl_u) & ~sl_u & (out[idx] == 0)
            if not holes.any():
                continue
            lab, _ = ndimage.label(holes)
            sizes = np.bincount(lab.ravel())
            small = np.isin(lab, np.nonzero((sizes <= max_vox) & (np.arange(len(sizes)) > 0))[0])
            if small.any():
                sl = out[idx]
                sl[small] = WALL
                out[idx] = sl
                stats["filled_hole_vox"] += int(small.sum())
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True, help="folder with <case_id>.nii.gz predictions")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits")
    p.add_argument("--subset", default="val")
    p.add_argument("--no-largest", action="store_true")
    p.add_argument("--no-detached", action="store_true")
    p.add_argument("--no-holes", action="store_true")
    p.add_argument("--max-hole-mm2", type=float, default=50.0)
    args = p.parse_args(argv)

    cfg = PostprocessConfig(not args.no_largest, not args.no_detached, not args.no_holes, args.max_hole_mm2)
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.splits:
        cases = json.loads(Path(args.splits).read_text())[args.subset]
    else:
        cases = sorted({f.name.split(".nii")[0] for f in in_dir.glob("*.nii*")})
    totals = {}
    done = 0
    for cid in cases:
        try:
            nii = load_nifti(in_dir / f"{cid}.nii.gz")
        except FileNotFoundError:
            print(f"  skip {cid}: no prediction")
            continue
        seg = np.rint(np.asarray(nii.dataobj)).astype(np.uint8)
        clean, stats = postprocess(seg, nii.header.get_zooms()[:3], cfg)
        img = nib.Nifti1Image(clean, nii.affine, nii.header)
        img.set_data_dtype(np.uint8)
        nib.save(img, out_dir / f"{cid}.nii.gz")
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        done += 1
    print(f"{done} cases -> {out_dir}; voxels changed: {totals}")


if __name__ == "__main__":
    main()
