"""FIGO-level validation.

distribution: compare a Stage 3 CSV against published clinical frequencies.
    python -m figo.figo_eval distribution --csv results/figo_expert.csv

agreement: downstream model-selection metric - run Stages 2+3 on the expert
masks and on a model's predicted masks, match fibroids by overlap, and report
detection rate plus exact/coarse FIGO agreement and Cohen's kappa.
    python -m figo.figo_eval agreement --ref-dir data/clean --pred-dir preds/mednext \
        --splits data/splits.json --subset test --out results/mednext_figo_agreement.csv

sensitivity: re-apply the Stage 3 rules to the contact fractions stored in a
Stage 3 CSV with each threshold nudged up and down, and count how many FIGO
calls change. No volumes are re-read, so it runs in seconds on CPU.
    python -m figo.figo_eval sensitivity --csv results/figo_expert_all.csv \
        --out results/figo_sensitivity.csv

errors: where each model's FIGO calls go wrong (ref->pred confusions) and what
its missed fibroids look like (size, slices), from the agreement CSVs.
    python -m figo.figo_eval errors --csvs results_val/*_figo_agreement.csv \
        --out results_val/figo_error_table.csv

UMD ships no FIGO labels, so "agreement" is agreement with the FIGO types
derived from the expert masks by the same rules - it measures how much the
segmentation error changes the FIGO call, not true clinical accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from .io import load_nifti
from .stage3 import FigoRules, _apply_rules, classify_case, coarse_group

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


def agreement_case_full(ref_seg, pred_seg, spacing, rules=None):
    """Matched pairs, missed expert fibroids, and fibroid counts for one case."""
    inst_r, rows_r = classify_case(ref_seg, spacing, rules)
    inst_p, rows_p = classify_case(pred_seg, spacing, rules)
    by_r = {r["fibroid_id"]: r for r in rows_r}
    by_p = {r["fibroid_id"]: r for r in rows_p}
    matches = match_instances(inst_r, inst_p)
    pairs = [{"ref_id": r, "pred_id": q, "dice": round(d, 3), "ref_figo": by_r[r]["figo"],
              "pred_figo": by_p[q]["figo"], "ref_volume_ml": by_r[r]["volume_ml"],
              "ref_n_slices": by_r[r]["n_slices"]} for r, q, d in matches]
    hit = {r for r, _, _ in matches}
    missed = [{"ref_id": r["fibroid_id"], "ref_figo": r["figo"], "ref_volume_ml": r["volume_ml"],
               "ref_n_slices": r["n_slices"]} for r in rows_r if r["fibroid_id"] not in hit]
    return pairs, missed, len(rows_r), len(rows_p)


def agreement_case(ref_seg, pred_seg, spacing, rules=None):
    pairs, _, nr, npd = agreement_case_full(ref_seg, pred_seg, spacing, rules)
    return pairs, nr, npd


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


# thresholds nudged by the sensitivity analysis; min_volume_mm3 is excluded
# because it changes Stage 2 (which fibroids exist), not the FIGO call
MM_THRESHOLDS = {"endometrium_tol_mm"}


def _row_figo(r, rules):
    dist = float("inf") if r["dist_to_cavity_mm"] in ("inf", "") else float(r["dist_to_cavity_mm"])
    figo, _, _ = _apply_rules(float(r["frac_cavity"]), float(r["frac_myometrium"]), float(r["frac_outside"]),
                              float(r["frac_cyst"]), dist, rules)
    return figo


def sensitivity(rows, base=None, delta=0.05, delta_mm=0.5) -> list[dict]:
    """One result row per (threshold, direction): how many FIGO calls change vs ``base``."""
    base = base or FigoRules()
    ref = [_row_figo(r, base) for r in rows]
    out = []
    for f in fields(base):
        if f.name == "min_volume_mm3":
            continue
        step = delta_mm if f.name in MM_THRESHOLDS else delta
        for sign in (-1, 1):
            val = round(getattr(base, f.name) + sign * step, 4)
            if val < 0:
                continue
            new = [_row_figo(r, replace(base, **{f.name: val})) for r in rows]
            changed = [(a, b) for a, b in zip(ref, new) if a != b]
            coarse = sum(coarse_group(a) != coarse_group(b) for a, b in changed)
            moves = Counter(f"{a}->{b}" for a, b in changed).most_common(3)
            out.append({
                "threshold": f.name, "base": getattr(base, f.name), "value": val,
                "n_fibroids": len(rows), "n_changed": len(changed),
                "pct_changed": round(100 * len(changed) / max(len(rows), 1), 2),
                "n_coarse_changed": coarse,
                "top_moves": "; ".join(f"{m} ({c})" for m, c in moves),
            })
    return out


def _missed_path(agreement_csv) -> Path:
    p = Path(agreement_csv)
    return p.with_name(p.stem + "_missed.csv")


def _median(xs):
    return float(np.median(xs)) if xs else float("nan")


def error_report(pairs, missed, top=5) -> dict:
    """Where a model's FIGO calls go wrong, and what its missed fibroids look like."""
    wrong = Counter(f"{p['ref_figo']}->{p['pred_figo']}" for p in pairs if p["ref_figo"] != p["pred_figo"])
    coarse = Counter(f"{coarse_group(p['ref_figo'])}->{coarse_group(p['pred_figo'])}" for p in pairs
                     if coarse_group(p["ref_figo"]) != coarse_group(p["pred_figo"]))
    per_type = {}
    for p in pairs:
        t = per_type.setdefault(p["ref_figo"], [0, 0])
        t[0] += 1
        t[1] += p["ref_figo"] == p["pred_figo"]
    vol_hit = [float(p["ref_volume_ml"]) for p in pairs]
    vol_miss = [float(m["ref_volume_ml"]) for m in missed]
    n_ref = len(pairs) + len(missed)
    thin_miss = sum(int(m["ref_n_slices"]) <= 2 for m in missed)
    small_miss = sum(float(m["ref_volume_ml"]) < 1.0 for m in missed)
    return {
        "n_matched": len(pairs), "n_missed": len(missed),
        "recall": len(pairs) / n_ref if n_ref else float("nan"),
        "top_errors": wrong.most_common(top),
        "top_coarse_errors": coarse.most_common(top),
        "per_type_accuracy": {k: (v[1], v[0]) for k, v in sorted(per_type.items())},
        "median_ml_found": _median(vol_hit), "median_ml_missed": _median(vol_miss),
        "missed_small_lt_1ml": small_miss, "missed_thin_le_2_slices": thin_miss,
        "missed_by_type": dict(Counter(m["ref_figo"] for m in missed).most_common()),
    }


def _load_seg(path):
    nii = load_nifti(path)
    return np.rint(np.asarray(nii.dataobj)).astype(np.uint8), nii


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distribution")
    d.add_argument("--csv", required=True)
    s = sub.add_parser("sensitivity")
    s.add_argument("--csv", required=True, help="Stage 3 CSV (e.g. expert masks, all cases)")
    s.add_argument("--delta", type=float, default=0.05, help="step for fraction thresholds")
    s.add_argument("--delta-mm", type=float, default=0.5, help="step for distance thresholds (mm)")
    s.add_argument("--out")
    a = sub.add_parser("agreement")
    a.add_argument("--ref-dir", required=True, help="clean/ folder with <id>_seg.nii.gz")
    a.add_argument("--pred-dir", required=True, help="folder with <id>.nii.gz predictions")
    a.add_argument("--splits", required=True)
    a.add_argument("--subset", default="test")
    a.add_argument("--out", required=True)
    e = sub.add_parser("errors", help="FIGO error table across models from agreement CSVs")
    e.add_argument("--csvs", nargs="+", required=True, help="<model>_figo_agreement.csv files")
    e.add_argument("--out", help="write one summary row per model to this CSV")
    args = p.parse_args(argv)

    if args.cmd == "distribution":
        with open(args.csv) as f:
            print_distribution(list(csv.DictReader(f)))
        return
    if args.cmd == "sensitivity":
        with open(args.csv) as f:
            res = sensitivity(list(csv.DictReader(f)), delta=args.delta, delta_mm=args.delta_mm)
        for r in res:
            print(f"{r['threshold']:30s} {r['base']:>6} -> {r['value']:<6} changed {r['n_changed']:4d} "
                  f"({r['pct_changed']:5.1f}%), coarse {r['n_coarse_changed']:3d}   {r['top_moves']}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res[0]))
                w.writeheader()
                w.writerows(res)
        return

    if args.cmd == "errors":
        rows = []
        for path in args.csvs:
            path = Path(path)
            model = path.name.replace("_figo_agreement.csv", "")
            with open(path) as f:
                pairs = list(csv.DictReader(f))
            mpath = _missed_path(path)
            missed = []
            if mpath.exists():
                with open(mpath) as f:
                    missed = list(csv.DictReader(f))
            r = error_report(pairs, missed)
            print(f"\n== {model}: {r['n_matched']} matched, {r['n_missed']} missed (recall {r['recall']:.2f})")
            print("  top FIGO errors (ref->pred):", ", ".join(f"{k} x{v}" for k, v in r["top_errors"]) or "none")
            print("  coarse errors:", ", ".join(f"{k} x{v}" for k, v in r["top_coarse_errors"]) or "none")
            print("  per-type correct/total:",
                  ", ".join(f"{t}: {c}/{n}" for t, (c, n) in r["per_type_accuracy"].items()))
            if mpath.exists():
                print(f"  missed: median {r['median_ml_missed']:.2f} ml vs found {r['median_ml_found']:.2f} ml; "
                      f"{r['missed_small_lt_1ml']}/{r['n_missed']} < 1 ml, "
                      f"{r['missed_thin_le_2_slices']}/{r['n_missed']} on <= 2 slices; by type {r['missed_by_type']}")
            rows.append({"model": model, "n_matched": r["n_matched"], "n_missed": r["n_missed"],
                         "recall": round(r["recall"], 3),
                         "top_errors": "; ".join(f"{k} x{v}" for k, v in r["top_errors"]),
                         "top_coarse_errors": "; ".join(f"{k} x{v}" for k, v in r["top_coarse_errors"]),
                         "median_ml_found": round(r["median_ml_found"], 3),
                         "median_ml_missed": round(r["median_ml_missed"], 3),
                         "missed_lt_1ml": r["missed_small_lt_1ml"],
                         "missed_le_2_slices": r["missed_thin_le_2_slices"]})
        if args.out and rows:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        return

    import nibabel as nib

    cases = json.loads(Path(args.splits).read_text())[args.subset]
    rules = FigoRules()
    all_pairs, all_missed, n_ref, n_pred = [], [], 0, 0
    for cid in cases:
        ref, ref_nii = _load_seg(Path(args.ref_dir) / f"{cid}_seg.nii.gz")
        pred_nii = load_nifti(Path(args.pred_dir) / f"{cid}.nii.gz")
        # bring both into canonical orientation so voxels correspond
        ref = np.asarray(nib.as_closest_canonical(nib.Nifti1Image(ref, ref_nii.affine)).dataobj)
        can_p = nib.as_closest_canonical(pred_nii)
        pred = np.rint(np.asarray(can_p.dataobj)).astype(np.uint8)
        spacing = can_p.header.get_zooms()[:3]
        pairs, missed, nr, npd = agreement_case_full(ref, pred, spacing, rules)
        all_pairs += [{"case_id": cid, **x} for x in pairs]
        all_missed += [{"case_id": cid, **x} for x in missed]
        n_ref += nr
        n_pred += npd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fields = ["case_id", "ref_id", "pred_id", "dice", "ref_figo", "pred_figo", "ref_volume_ml", "ref_n_slices"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_pairs)
    with open(_missed_path(out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "ref_id", "ref_figo", "ref_volume_ml", "ref_n_slices"])
        w.writeheader()
        w.writerows(all_missed)
    summary = summarize_agreement(all_pairs, n_ref, n_pred)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
