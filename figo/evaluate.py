"""Score any model's predictions against the expert masks at native resolution.

    python -m figo.evaluate --pred-dir preds/mednext --data-dir data/clean \
        --splits data/splits.json --subset test --out results/mednext_seg.csv

Works for both the custom models and nnU-Net (both write ``<case_id>.nii.gz``).
Prediction and ground truth are compared in canonical orientation, so a
prediction saved in either the original or canonical orientation is fine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from .io import load_nifti
from .metrics import case_metrics


def load_pair(pred_path, gt_path):
    pred = nib.as_closest_canonical(load_nifti(pred_path))
    gt = nib.as_closest_canonical(load_nifti(gt_path))
    if pred.shape[:3] != gt.shape[:3] or not np.allclose(pred.affine, gt.affine, atol=1e-2):
        raise ValueError(f"{pred_path}: geometry does not match ground truth")
    return (np.rint(np.asarray(pred.dataobj)).astype(np.uint8),
            np.asarray(gt.dataobj).astype(np.uint8),
            tuple(float(z) for z in gt.header.get_zooms()[:3]))


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "case_id"]
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        finite = v[np.isfinite(v)]
        summary[k] = {
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std()) if len(finite) else float("nan"),
            "n": int(len(finite)),
            "n_missed": int(np.isinf(v).sum()),  # structure present in only one of pred/gt
        }
    struct_dice = [summary[f"dice_{s}"]["mean"] for s in ("wall", "cavity", "myoma")]
    summary["macro_dice_wall_cavity_myoma"] = {"mean": float(np.nanmean(struct_dice))}
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--data-dir", required=True, help="figo.prepare clean/ folder")
    p.add_argument("--splits")
    p.add_argument("--subset", default="test")
    p.add_argument("--out", required=True, help="per-case CSV; a <out>.summary.json is written next to it")
    args = p.parse_args(argv)

    pred_dir, data = Path(args.pred_dir), Path(args.data_dir)
    if args.splits:
        cases = json.loads(Path(args.splits).read_text())[args.subset]
    else:
        cases = sorted(f.name[: -len(".nii.gz")] for f in pred_dir.glob("*.nii.gz"))
    rows = []
    for i, cid in enumerate(cases, 1):
        pred, gt, spacing = load_pair(pred_dir / f"{cid}.nii.gz", data / f"{cid}_seg.nii.gz")
        rows.append({"case_id": cid, **case_metrics(pred, gt, spacing)})
        if i % 10 == 0:
            print(f"  {i}/{len(cases)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{len(rows)} cases from {pred_dir}")
    for k, s in summary.items():
        if "std" in s:
            extra = f"  ({s['n_missed']} missed)" if s["n_missed"] else ""
            print(f"  {k:32s} {s['mean']:8.3f} +/- {s['std']:.3f}{extra}")
        elif not math.isnan(s["mean"]):
            print(f"  {k:32s} {s['mean']:8.3f}")


if __name__ == "__main__":
    main()
