"""Step 1 of the build order: load, validate and clean the UMD dataset.

    python -m figo.prepare --umd-dir /content/UMD --out-dir /content/drive/MyDrive/figo/data

Produces in ``--out-dir``:
  clean/<id>_t2.nii.gz, clean/<id>_seg.nii.gz  properly gzipped copies (original orientation;
                                                seg forced to uint8 and to the image affine)
  manifest.csv                                 per-case shape, spacing, labels, fibroid count,
                                                and whether the source seg was mislabeled gzip
  splits.json                                  fixed train/val/test split shared by ALL models
And with ``--cache`` also ``cache_<mm>mm/<id>.npz`` network-space arrays for figo.train.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from .instances import fibroid_instances
from .io import discover_cases, is_gzip, load_nifti
from .labels import LABEL_NAMES, NUM_CLASSES, VALID_LABELS
from .preprocess import DEFAULT_INPLANE_MM, to_network_space


def clean_case(case, out_dir: Path) -> dict:
    img = load_nifti(case.image)
    seg = load_nifti(case.seg)
    if img.shape[:3] != seg.shape[:3]:
        raise ValueError(f"{case.case_id}: image {img.shape} vs seg {seg.shape}")
    seg_arr = np.asarray(seg.dataobj)
    rounded = np.rint(seg_arr)
    if not np.allclose(seg_arr, rounded):
        raise ValueError(f"{case.case_id}: non-integer segmentation values")
    seg_arr = rounded.astype(np.uint8)
    found = set(np.unique(seg_arr).tolist())
    if not found <= VALID_LABELS:
        raise ValueError(f"{case.case_id}: unexpected labels {sorted(found - VALID_LABELS)}")
    affine_mismatch = float(np.abs(img.affine - seg.affine).max())

    img_out = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
    seg_out = nib.Nifti1Image(seg_arr, img.affine)  # trust the image geometry
    seg_out.set_data_dtype(np.uint8)
    nib.save(img_out, out_dir / f"{case.case_id}_t2.nii.gz")
    nib.save(seg_out, out_dir / f"{case.case_id}_seg.nii.gz")

    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    _, fibroids = fibroid_instances(seg_arr, zooms)
    counts = np.bincount(seg_arr.ravel(), minlength=NUM_CLASSES)
    row = {
        "case_id": case.case_id,
        "shape": "x".join(map(str, img.shape[:3])),
        "spacing_mm": "x".join(f"{z:.3f}" for z in zooms),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "seg_was_mislabeled_gzip": int(case.seg.name.endswith(".gz") and not is_gzip(case.seg)),
        "image_was_mislabeled_gzip": int(case.image.name.endswith(".gz") and not is_gzip(case.image)),
        "seg_affine_max_abs_diff": round(affine_mismatch, 6),
        "n_fibroids": len(fibroids),
    }
    for lab, name in LABEL_NAMES.items():
        row[f"vox_{name}"] = int(counts[lab])
    return row


def make_splits(rows: list[dict], seed: int, val_frac: float, test_frac: float) -> dict:
    """Split stratified by fibroid count so every split sees single and multi-fibroid uteri."""
    def stratum(n):
        return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"

    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = {}
    for r in rows:
        by_stratum.setdefault(stratum(r["n_fibroids"]), []).append(r["case_id"])
    splits = {"train": [], "val": [], "test": [], "seed": seed}
    for ids in by_stratum.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = round(len(ids) * test_frac)
        n_val = round(len(ids) * val_frac)
        splits["test"] += ids[:n_test]
        splits["val"] += ids[n_test : n_test + n_val]
        splits["train"] += ids[n_test + n_val :]
    for k in ("train", "val", "test"):
        splits[k].sort()
    return splits


def build_cache(clean_dir: Path, case_ids, cache_dir: Path, target_inplane: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(case_ids, 1):
        out = cache_dir / f"{cid}.npz"
        if out.exists():
            continue
        img, seg, geom = to_network_space(
            load_nifti(clean_dir / f"{cid}_t2.nii.gz"),
            load_nifti(clean_dir / f"{cid}_seg.nii.gz"),
            target_inplane,
        )
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, image=img.astype(np.float16), seg=seg, geometry=json.dumps(geom.to_dict())
        )
        tmp.replace(out)
        if i % 25 == 0:
            print(f"  cached {i}/{len(case_ids)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--umd-dir", required=True, help="unzipped UMD folder (contains UMD_<id>/ dirs)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--cache", action="store_true", help="also build network-space .npz cache")
    p.add_argument("--inplane-mm", type=float, default=DEFAULT_INPLANE_MM)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    clean = out / "clean"
    clean.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.umd_dir)
    print(f"Found {len(cases)} cases")
    rows, failures = [], []
    for i, case in enumerate(cases, 1):
        try:
            rows.append(clean_case(case, clean))
        except Exception as e:  # keep going; report every bad case at the end
            failures.append((case.case_id, repr(e)))
        if i % 25 == 0:
            print(f"  checked {i}/{len(cases)}")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    spacings = Counter(r["spacing_mm"] for r in rows)
    print(f"\nLoaded {len(rows)}/{len(cases)} cases; {len(failures)} failures")
    for cid, err in failures:
        print(f"  FAILED {cid}: {err}")
    print(f"Segmentations that were uncompressed despite .gz: "
          f"{sum(r['seg_was_mislabeled_gzip'] for r in rows)}")
    print(f"Images that were uncompressed despite .gz: "
          f"{sum(r['image_was_mislabeled_gzip'] for r in rows)}")
    print(f"Orientations: {dict(Counter(r['orientation'] for r in rows))}")
    print("Most common spacings (mm):")
    for s, n in spacings.most_common(5):
        print(f"  {s}: {n} cases")
    print(f"Fibroid instances (>=50 mm^3, CC): {sum(r['n_fibroids'] for r in rows)} "
          f"(dataset paper reports ~1077)")
    for lab, name in LABEL_NAMES.items():
        print(f"  cases containing {name}: {sum(r[f'vox_{name}'] > 0 for r in rows)}")

    splits_path = out / "splits.json"
    if splits_path.exists():
        print(f"\nKeeping existing {splits_path} (delete it to re-split)")
        splits = json.loads(splits_path.read_text())
    else:
        splits = make_splits(rows, args.seed, args.val_frac, args.test_frac)
        splits_path.write_text(json.dumps(splits, indent=1))
    print(f"Split sizes: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

    if args.cache:
        mm = f"{args.inplane_mm:g}"
        print(f"\nBuilding network-space cache at {mm} mm in-plane ...")
        build_cache(clean, [r["case_id"] for r in rows], out / f"cache_{mm}mm", args.inplane_mm)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
