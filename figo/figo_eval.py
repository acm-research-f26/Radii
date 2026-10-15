"""FIGO-level validation.

distribution: compare a Stage 3 CSV against published clinical frequencies.
    python -m figo.figo_eval distribution --csv results/figo_expert.csv

agreement: downstream model-selection metric - run Stages 2+3 on the expert
masks and on a model's predicted masks, match fibroids by overlap, and report
detection rate plus exact/coarse FIGO agreement and Cohen's kappa.
    python -m figo.figo_eval agreement --ref-dir data/clean --pred-dir preds/mednext \
        --splits data/splits.json --subset test --out results/mednext_figo_agreement.csv

UMD ships no FIGO labels, so "agreement" is agreement with the FIGO types
derived from the expert masks by the same rules - it measures how much the
segmentation error changes the FIGO call, not true clinical accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .io import load_nifti
from .stage3 import FigoRules, classify_case, coarse_group

# Published frequency ranges as given in the project brief (percent of fibroids).
# Confirm each against its source paper before using in the write-up.
REFERENCE_RANGES = {"4": (30, 55), "3": (13, 34)}
RARE_TYPES = ("0", "7", "8")
RARE_RANGE = (3, 5)


def distribution(rows) -> dict:
    counts = Counter(r["figo"] for r in rows)
    n = sum(counts.values())
    return {k: (v, 100 * v / n) for k, v in sorted(counts.items())}


def print_distribution(rows):
    dist = distribution(rows)
    n = sum(v for v, _ in dist.values())
    print(f"{n} fibroids")
    print(f"{'FIGO':>6} {'n':>5} {'%':>6}   reference")
    for k, (c, pct) in dist.items():
        ref = REFERENCE_RANGES.get(k) or (RARE_RANGE if k in RARE_TYPES else None)
        ref_s = ""
        if ref:
            ok = ref[0] <= pct <= ref[1]
            ref_s = f"{ref[0]}-{ref[1]}%  {'within' if ok else 'OUTSIDE'}"
        print(f"{k:>6} {c:5d} {pct:6.1f}   {ref_s}")
    groups = Counter(coarse_group(r["figo"]) for r in rows)
    print("coarse:", ", ".join(f"{g} {100 * c / n:.1f}%" for g, c in groups.most_common()))
    flagged = sum(bool(r.get("flags")) for r in rows)
    print(f"{flagged} fibroids ({100 * flagged / max(n, 1):.0f}%) carry a caution flag")


def cohen_kappa(a, b) -> float:
    cats = sorted(set(a) | set(b))
    if len(a) == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def match_instances(inst_ref, inst_pred, min_dice=0.1):
    """Greedy one-to-one matching by Dice; returns list of (ref_id, pred_id, dice)."""
    nr, npred = int(inst_ref.max()), int(inst_pred.max())
    if nr == 0 or npred == 0:
        return []
    both = (inst_ref > 0) & (inst_pred > 0)
    pair = np.bincount(inst_ref[both].astype(np.int64) * (npred + 1) + inst_pred[both],
                       minlength=(nr + 1) * (npred + 1)).reshape(nr + 1, npred + 1)
    size_r = np.bincount(inst_ref.ravel(), minlength=nr + 1)
    size_p = np.bincount(inst_pred.ravel(), minlength=npred + 1)
    dice = 2 * pair / (size_r[:, None] + size_p[None, :] + 1e-9)
    dice[0, :] = dice[:, 0] = 0
    matches, used_r, used_p = [], set(), set()
    for flat in np.argsort(-dice, axis=None):
        r, q = np.unravel_index(flat, dice.shape)
        if dice[r, q] < min_dice:
            break
        if r in used_r or q in used_p:
            continue
        used_r.add(r)
        used_p.add(q)
        matches.append((int(r), int(q), float(dice[r, q])))
    return matches


def agreement_case(ref_seg, pred_seg, spacing, rules=None):
    inst_r, rows_r = classify_case(ref_seg, spacing, rules)
    inst_p, rows_p = classify_case(pred_seg, spacing, rules)
    by_r = {r["fibroid_id"]: r for r in rows_r}
    by_p = {r["fibroid_id"]: r for r in rows_p}
    matches = match_instances(inst_r, inst_p)
    pairs = [{"ref_id": r, "pred_id": q, "dice": round(d, 3), "ref_figo": by_r[r]["figo"],
              "pred_figo": by_p[q]["figo"], "ref_volume_ml": by_r[r]["volume_ml"]} for r, q, d in matches]
    return pairs, len(rows_r), len(rows_p)


def summarize_agreement(pairs, n_ref, n_pred) -> dict:
    a = [p["ref_figo"] for p in pairs]
    b = [p["pred_figo"] for p in pairs]
    ca = [coarse_group(x) for x in a]
    cb = [coarse_group(x) for x in b]
    n = len(pairs)
    return {
        "n_ref_fibroids": n_ref,
        "n_pred_fibroids": n_pred,
        "n_matched": n,
        "detection_recall": n / n_ref if n_ref else float("nan"),
        "detection_precision": n / n_pred if n_pred else float("nan"),
        "exact_agreement": float(np.mean([x == y for x, y in zip(a, b)])) if n else float("nan"),
        "coarse_agreement": float(np.mean([x == y for x, y in zip(ca, cb)])) if n else float("nan"),
        "kappa_exact": cohen_kappa(a, b),
        "kappa_coarse": cohen_kappa(ca, cb),
    }


def _load_seg(path):
    nii = load_nifti(path)
    return np.rint(np.asarray(nii.dataobj)).astype(np.uint8), nii


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distribution")
    d.add_argument("--csv", required=True)
    a = sub.add_parser("agreement")
    a.add_argument("--ref-dir", required=True, help="clean/ folder with <id>_seg.nii.gz")
    a.add_argument("--pred-dir", required=True, help="folder with <id>.nii.gz predictions")
    a.add_argument("--splits", required=True)
    a.add_argument("--subset", default="test")
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if args.cmd == "distribution":
        with open(args.csv) as f:
            print_distribution(list(csv.DictReader(f)))
        return

    import nibabel as nib

    cases = json.loads(Path(args.splits).read_text())[args.subset]
    rules = FigoRules()
    all_pairs, n_ref, n_pred = [], 0, 0
    for cid in cases:
        ref, ref_nii = _load_seg(Path(args.ref_dir) / f"{cid}_seg.nii.gz")
        pred_nii = load_nifti(Path(args.pred_dir) / f"{cid}.nii.gz")
        # bring both into canonical orientation so voxels correspond
        ref = np.asarray(nib.as_closest_canonical(nib.Nifti1Image(ref, ref_nii.affine)).dataobj)
        can_p = nib.as_closest_canonical(pred_nii)
        pred = np.rint(np.asarray(can_p.dataobj)).astype(np.uint8)
        spacing = can_p.header.get_zooms()[:3]
        pairs, nr, npd = agreement_case(ref, pred, spacing, rules)
        all_pairs += [{"case_id": cid, **x} for x in pairs]
        n_ref += nr
        n_pred += npd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fields = ["case_id", "ref_id", "pred_id", "dice", "ref_figo", "pred_figo", "ref_volume_ml"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_pairs)
    summary = summarize_agreement(all_pairs, n_ref, n_pred)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
