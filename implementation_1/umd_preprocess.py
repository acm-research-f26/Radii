#!/usr/bin/env python3
"""
umd_preprocess.py - explore, check and clean the UMD uterine myoma MRI dataset.

Dataset: Pan et al., "Large-scale uterine myoma MRI dataset covering all FIGO
types with pixel-level annotations", Scientific Data (2024). Not included in
this repository; download it from the paper's Data Availability section.

Expected layout after unzipping (one folder per patient):
    UMD/UMD_221129_001/
        UMD_221129_001_01.dcm ... (raw DICOM slices, ignored here)
        UMD_221129_001_t2.nii.gz  (T2 sagittal image)
        UMD_221129_001_seg.nii.gz (label map)

Known quirks of the public release that this script handles:
  * macOS junk (__MACOSX, ._* files, .DS_Store) mixed in with the real data
  * ~34 label files named "_seq.nii.gz" instead of "_seg.nii.gz"
  * a label file whose name has a typo in the case number
  * ~34 label files with a .gz extension that are actually uncompressed NIfTI
    (and one of them, case 037, also has a typo in its file name)

Usage:
    pip install numpy nibabel

    python umd_preprocess.py explore --root path/to/UMD/UMD
    python umd_preprocess.py check   --root path/to/UMD/UMD
    python umd_preprocess.py build   --root path/to/UMD/UMD --out clean/

`build` writes clean/images/<case>.nii.gz and clean/labels/<case>.nii.gz
(labels as uint8: 0 background, 1 uterine wall, 2 uterine cavity, 3 myoma,
4 Nabothian cyst) plus clean/summary.csv, ready for nnU-Net conversion.
Label meanings were inferred from voxel statistics and the paper's cyst count
(127 cases); confirm visually in ITK-SNAP or 3D Slicer before relying on them.
"""

# imports 

import argparse
import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

LABEL_MEANING = {0: "background", 1: "uterine wall", 2: "uterine cavity",
                 3: "myoma", 4: "Nabothian cyst"}
IMAGE_SUFFIXES = ("_t2.nii.gz",)
# "_seq" is a known typo of "_seg" in the public release
LABEL_SUFFIXES = ("_seg.nii.gz", "_seq.nii.gz")

# Helpers

def is_junk(p: Path) -> bool:
    """macOS archive leftovers that are not real data."""
    return p.name.startswith("._") or p.name == ".DS_Store" or "__MACOSX" in p.parts


def list_cases(root: Path):
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("UMD_") and not is_junk(p))


def find_one(folder: Path, suffixes):
    """The single .nii.gz in `folder` ending with one of `suffixes`, else None."""
    hits = [p for p in folder.glob("*.nii.gz")
            if not is_junk(p) and any(p.name.endswith(s) for s in suffixes)]
    return hits[0] if len(hits) == 1 else None


def load_nifti(path: Path):
    """Load a NIfTI fully into memory. Handles files that have a .gz name but
    are actually uncompressed (as in ~34 UMD label files). Reads the raw bytes
    directly (no temp files, which break on Windows when memory-mapped).
    Returns None if the file cannot be read."""
    try:
        im = nib.load(str(path))
        return nib.Nifti1Image(np.array(im.dataobj), im.affine, im.header)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            im = nib.Nifti1Image.from_bytes(f.read())
        return nib.Nifti1Image(np.array(im.dataobj), im.affine, im.header)
    except Exception:
        return None

# Command: explore

def explore(root):
    """Print what is in the dataset folder (case count, file types, first case)."""
    root = Path(root)
    print("Folder exists:", root.exists())
    cases = list_cases(root)
    print("Number of case folders:", len(cases))
    print("First 3:", [c.name for c in cases[:3]])
    print("Last 3:", [c.name for c in cases[-3:]])

    real = [p for p in root.rglob("*") if p.is_file() and not is_junk(p)]
    print("\nReal files:", len(real))
    print("File types:", Counter(p.suffix.lower() or "(none)" for p in real).most_common())

    if cases:
        first = cases[0]
        print(f"\n--- Inside {first.name} ---")
        for p in sorted(q for q in first.rglob("*") if not is_junk(q))[:40]:
            print(" ", p.relative_to(first), "|", f"{p.stat().st_size // 1024} KB")

# Command: check

def check(root):
    """List cases whose image/label files are missing or oddly named."""
    root = Path(root)
    bad = []
    for c in list_cases(root):
        t2 = find_one(c, IMAGE_SUFFIXES)
        seg = find_one(c, LABEL_SUFFIXES)
        exact = (c / f"{c.name}_seg.nii.gz").exists()
        if t2 is None or seg is None or not exact:
            others = sorted(p.name for p in c.iterdir()
                            if not is_junk(p) and p.suffix.lower() != ".dcm")
            bad.append((c.name, t2 is not None, seg is not None, others))
    print("Cases needing attention:", len(bad))
    for name, has_t2, has_seg, others in bad:
        print(f"{name}: image_found={has_t2} label_found={has_seg} | files: {others}")

# Command: build

def build(root, out="clean"):
    """Create paired, validated image/label NIfTI files for nnU-Net."""
    root, out = Path(root), Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    cases = list_cases(root)
    print("Case folders:", len(cases))

    problems, recovered = [], []
    shapes, spacings = Counter(), Counter()
    cases_with_label, voxels_per_label = Counter(), Counter()
    rows = []

    for i, c in enumerate(cases, 1):
        t2 = find_one(c, IMAGE_SUFFIXES)
        seg = find_one(c, LABEL_SUFFIXES)
        if t2 is None or seg is None:
            problems.append((c.name, "could not find exactly one image and one label file"))
            continue
        if seg.name != f"{c.name}_seg.nii.gz":
            recovered.append((c.name, seg.name))

        img, lab = load_nifti(t2), load_nifti(seg)
        if img is None or lab is None:
            bad = t2 if img is None else seg
            problems.append((c.name, f"unreadable file: {bad.name}"))
            continue

        lab_arr = np.asarray(lab.dataobj)
        if img.shape != lab.shape:
            problems.append((c.name, f"shape mismatch {img.shape} vs {lab.shape}"))
            continue
        if not np.allclose(img.affine, lab.affine, atol=1e-3):
            problems.append((c.name, "affine mismatch (saved anyway)"))
        if not np.allclose(lab_arr, np.round(lab_arr)):
            problems.append((c.name, "label has non-integer values"))
            continue

        lab_int = np.round(lab_arr).astype(np.uint8)
        vals, counts = np.unique(lab_int, return_counts=True)
        if vals.max() > 4:
            problems.append((c.name, f"unexpected label values {vals.tolist()}"))
            continue
        for v, n in zip(vals, counts):
            cases_with_label[int(v)] += 1
            voxels_per_label[int(v)] += int(n)

        shapes[img.shape] += 1
        spacings[tuple(round(float(s), 2) for s in img.header.get_zooms()[:3])] += 1

        nib.save(img, str(out / "images" / f"{c.name}.nii.gz"))
        hdr = lab.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(lab_int, lab.affine, hdr),
                 str(out / "labels" / f"{c.name}.nii.gz"))
        rows.append([c.name, img.shape, img.header.get_zooms()[:3], sorted(set(vals.tolist()))])

        if i % 50 == 0:
            print(f"  processed {i}/{len(cases)}")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "shape", "spacing_mm", "label_values"])
        w.writerows(rows)

    print("\nCases saved:", len(rows))
    print("Label files recovered from non-standard names:", len(recovered))
    print("Problems:", len(problems))
    for p in problems:
        print("  ", p)
    print("\nLabel value -> cases containing it (total voxels):")
    for v in sorted(cases_with_label):
        print(f"  {v} ({LABEL_MEANING.get(v, '?')}): "
              f"{cases_with_label[v]} cases, {voxels_per_label[v]} voxels")
    print("\nMost common image shapes:", shapes.most_common(5))
    print("Most common voxel spacings (mm):", spacings.most_common(5))

# CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, helptext in [
        ("explore", explore, "summarize the dataset folder"),
        ("check", check, "list cases with missing or oddly named files"),
        ("build", build, "write the cleaned, paired NIfTI dataset"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--root", required=True,
                        help="folder containing the UMD_* case folders")
        if name == "build":
            sp.add_argument("--out", default="clean", help="output folder (default: clean)")
        sp.set_defaults(fn=fn)

    args = ap.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if k not in ("cmd", "fn")}
    args.fn(**kwargs)


if __name__ == "__main__":
    main()