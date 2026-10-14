"""Rank Stage 1 models by the FIGO-oriented selection criteria.

    python -m figo.select_models --results-dir /content/drive/MyDrive/figo/results

Expects, per model ``<m>``: ``<m>_seg.summary.json`` (figo.evaluate) and
``<m>_figo_agreement.summary.json`` (figo.figo_eval agreement). Criteria:
  1. boundary accuracy at the cavity/wall interface (HD95, lower is better)
  2. per-structure Dice for cavity and wall (not just fibroid)
  3. downstream FIGO agreement with expert-mask FIGO (coarse kappa)
Models are ordered by mean rank across the three; carry the top 2-3 forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CRITERIA = [
    # (column, source, key, higher_is_better)
    ("HD95 cav/wall mm", "seg", "hd95_cavity_wall_interface", False),
    ("Dice cavity", "seg", "dice_cavity", True),
    ("Dice wall", "seg", "dice_wall", True),
    ("Dice myoma", "seg", "dice_myoma", True),
    ("HD95 serosa mm", "seg", "hd95_serosa", False),
    ("FIGO kappa coarse", "figo", "kappa_coarse", True),
    ("FIGO exact agr", "figo", "exact_agreement", True),
    ("fibroid recall", "figo", "detection_recall", True),
]
RANKED = ["HD95 cav/wall mm", "Dice cavity+wall", "FIGO kappa coarse"]


def _get(summary, key):
    v = summary.get(key, float("nan"))
    return v["mean"] if isinstance(v, dict) else v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    args = p.parse_args(argv)
    res = Path(args.results_dir)

    table = {}
    for seg_file in sorted(res.glob("*_seg.summary.json")):
        m = seg_file.name[: -len("_seg.summary.json")]
        seg = json.loads(seg_file.read_text())
        figo_file = res / f"{m}_figo_agreement.summary.json"
        figo = json.loads(figo_file.read_text()) if figo_file.exists() else {}
        row = {col: _get(seg if src == "seg" else figo, key) for col, src, key, _ in CRITERIA}
        row["Dice cavity+wall"] = np.nanmean([row["Dice cavity"], row["Dice wall"]])
        table[m] = row
    if not table:
        raise SystemExit(f"no *_seg.summary.json in {res}")

    models = list(table)
    ranks = np.zeros((len(models), len(RANKED)))
    for j, col in enumerate(RANKED):
        higher = col != "HD95 cav/wall mm"
        vals = np.array([table[m][col] for m in models], float)
        vals = np.where(np.isnan(vals), -np.inf if higher else np.inf, vals)
        order = np.argsort(-vals if higher else vals)
        ranks[order, j] = np.arange(1, len(models) + 1)
    mean_rank = ranks.mean(1)

    cols = [c for c, *_ in CRITERIA]
    print(f"{'model':24s} {'mean rank':>9s} " + " ".join(f"{c:>17s}" for c in cols))
    for i in np.argsort(mean_rank):
        m = models[i]
        print(f"{m:24s} {mean_rank[i]:9.2f} " + " ".join(f"{table[m][c]:17.3f}" for c in cols))
    print(f"\nRanked on: {', '.join(RANKED)}")


if __name__ == "__main__":
    main()
