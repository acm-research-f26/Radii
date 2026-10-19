"""Stage 3: geometric FIGO typing of each fibroid.

For every fibroid instance (Stage 2) we measure, at native voxel spacing, how
its surface area is shared between neighbours:

    cavity   (label 2)                      -> submucosal component
    tissue   (label 1 wall + other fibroids) -> embedded (intramural) component
    outside  (label 0 background)           -> subserosal component
    cyst     (label 4 nabothian cyst)       -> cervical hint

Surface is counted as voxel faces weighted by face area, so the 6.6 mm slices
are handled correctly. The *surface* fraction facing the cavity (or outside)
is our proxy for the protruding *volume* fraction: for a sphere cut by a plane
both are exactly 0.5 at the same cut, so the 50% FIGO thresholds coincide.

Rules (thresholds in ``FigoRules``):
  8   no contact with the segmented uterus, or mostly bordered by nabothian cysts (cervical)
  X-Y hybrid: touches both cavity and serosa
  0   submucosal, almost no myometrial contact (pedunculated)
  1   submucosal, intramural part < 50%
  2   submucosal, intramural part >= 50%
  3   intramural, abuts endometrium without protruding into cavity
  4   intramural, touches neither cavity nor serosa
  5   subserosal, intramural part >= 50%
  6   subserosal, intramural part < 50%
  7   subserosal, almost no myometrial contact (pedunculated)

Every row carries a plain-language rationale plus caution flags; the fine
distinctions (1 vs 2, 5 vs 6, 2/3/4 contact) are resolution-limited and are
flagged rather than hidden. Decision support only - the radiologist decides.

    python -m figo.stage3 --seg-dir data/clean --suffix _seg --splits data/splits.json \
        --subset test --out results/figo_expert.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .instances import fibroid_instances
from .io import load_nifti
from .labels import CAVITY, CYST, MYOMA, WALL

OUT, TISSUE, CAV, CYS = 0, 1, 2, 3

COARSE = {"0": "submucosal", "1": "submucosal", "2": "submucosal", "3": "intramural", "4": "intramural",
          "5": "subserosal", "6": "subserosal", "7": "subserosal", "8": "other"}


@dataclass
class FigoRules:
    touch_min_frac: float = 0.02  # contact below this is treated as segmentation noise
    submucosal_min_cav_frac: float = 0.10  # cavity-facing surface needed to call it protruding (2 vs 3)
    pedunculated_max_tissue_frac: float = 0.10  # myometrial contact below this -> type 0 / 7
    endometrium_tol_mm: float = 1.0  # within this distance of the cavity -> "abuts endometrium" (3 vs 4)
    cervical_min_cyst_frac: float = 0.10
    intramural_cut: float = 0.50  # FIGO 50% intramural cut-off (1 vs 2, 5 vs 6, hybrid sides)
    min_volume_mm3: float = 50.0


def coarse_group(figo: str) -> str:
    return "hybrid" if "-" in figo else COARSE[figo]


def contact_areas(inst_mask: np.ndarray, cat: np.ndarray, spacing) -> np.ndarray:
    """Surface area (mm^2) of ``inst_mask`` facing each category in ``cat``."""
    areas = np.zeros(4)
    for ax in range(3):
        face = float(np.prod([spacing[j] for j in range(3) if j != ax]))
        for a, b in ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1))):
            ia, ib = [slice(None)] * 3, [slice(None)] * 3
            ia[ax], ib[ax] = a, b
            src, nb = inst_mask[tuple(ia)], inst_mask[tuple(ib)]
            sel = src & ~nb
            areas += np.bincount(cat[tuple(ib)][sel], minlength=4)[:4] * face
    return areas


def _apply_rules(f_cav, f_tis, f_out, f_cyst, dist_cav, rules: FigoRules):
    """Return (figo, rationale, flags) from contact fractions."""
    flags = []
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    if f_cav + f_tis == 0:  # any contact counts: a pedunculated stalk is a tiny share of the surface
        return "8", "not attached to the segmented uterus (parasitic/cervical/other?)", ["review"]
    if f_cyst >= rules.cervical_min_cyst_frac:
        return "8", f"{pct(f_cyst)} of surface borders nabothian cysts -> likely cervical", ["review"]

    sub = f_cav >= rules.submucosal_min_cav_frac
    ser = f_out >= rules.touch_min_frac
    if abs(f_cav - rules.submucosal_min_cav_frac) < 0.05:
        flags.append("near_2_vs_3_threshold")

    if sub and ser:
        x = "1" if f_cav > rules.intramural_cut else "2"
        y = "6" if f_out > rules.intramural_cut else "5"
        return f"{x}-{y}", (f"hybrid: {pct(f_cav)} of surface in cavity and {pct(f_out)} outside the serosa"), flags

    if sub:
        intramural = 1 - f_cav
        if f_tis < rules.pedunculated_max_tissue_frac:
            figo, why = "0", f"only {pct(f_tis)} myometrial contact -> intracavitary/pedunculated"
        else:
            figo = "2" if intramural >= rules.intramural_cut else "1"
            why = f"contacts cavity ({pct(f_cav)} of surface); intramural ~{pct(intramural)} " + (
                ">= 50%" if figo == "2" else "< 50%")
        if abs(intramural - rules.intramural_cut) < 0.1:
            flags.append("borderline_50pct")
        return figo, why, flags

    if ser:
        intramural = 1 - f_out
        if f_tis < rules.pedunculated_max_tissue_frac:
            figo, why = "7", f"only {pct(f_tis)} myometrial contact -> subserosal pedunculated"
        else:
            figo = "5" if intramural >= rules.intramural_cut else "6"
            why = f"touches serosa ({pct(f_out)} of surface outside); intramural ~{pct(intramural)} " + (
                ">= 50%" if figo == "5" else "< 50%")
        if abs(intramural - rules.intramural_cut) < 0.1:
            flags.append("borderline_50pct")
        return figo, why, flags

    if f_cav > 0 or dist_cav <= rules.endometrium_tol_mm:
        return "3", f"fully intramural, abuts endometrium (gap {dist_cav:.1f} mm)", flags
    if dist_cav <= 2 * rules.endometrium_tol_mm + 1:
        flags.append("near_3_vs_4_threshold")
    gap = "no cavity segmented" if np.isinf(dist_cav) else f"{dist_cav:.1f} mm from cavity"
    return "4", f"fully intramural ({gap}, no serosal contact)", flags


def classify_case(seg: np.ndarray, spacing, rules: FigoRules | None = None):
    """Return ``(instance_map, rows)``; one row per fibroid."""
    rules = rules or FigoRules()
    spacing = tuple(float(s) for s in spacing)
    inst, info = fibroid_instances(seg, spacing, rules.min_volume_mm3)
    if not info:
        return inst, []
    cavity = seg == CAVITY
    dist_cav = (ndimage.distance_transform_edt(~cavity, sampling=spacing) if cavity.any()
                else np.full(seg.shape, np.inf))
    # neighbour category for every voxel (this fibroid itself is masked out per instance)
    lut = np.full(256, OUT, np.uint8)  # background and anything unexpected -> outside
    lut[[WALL, MYOMA]] = TISSUE
    lut[CAVITY], lut[CYST] = CAV, CYS
    cat_full = lut[seg]
    thick = int(np.argmax(spacing))

    rows = []
    for d in info:
        sl = tuple(slice(max(a - 1, 0), min(b + 1, n)) for (a, b), n in zip(d["bbox"], seg.shape))
        m = inst[sl] == d["id"]
        # pad so faces on the volume border count as "outside"
        m = np.pad(m, 1)
        cat = np.pad(cat_full[sl], 1, constant_values=OUT)
        areas = contact_areas(m, cat, spacing)
        total = areas.sum()
        f_out, f_tis, f_cav, f_cyst = (areas / total) if total else (1.0, 0.0, 0.0, 0.0)
        dmin = float(dist_cav[sl][inst[sl] == d["id"]].min())
        figo, why, flags = _apply_rules(f_cav, f_tis, f_out, f_cyst, dmin, rules)
        if d["n_slices"] <= 2:
            flags.append("thin_<=2_slices")
        rows.append({
            "fibroid_id": d["id"],
            "figo": figo,
            "group": coarse_group(figo),
            "volume_ml": round(d["volume_mm3"] / 1000, 3),
            "n_slices": d["n_slices"],
            "frac_cavity": round(float(f_cav), 3),
            "frac_myometrium": round(float(f_tis), 3),
            "frac_outside": round(float(f_out), 3),
            "frac_cyst": round(float(f_cyst), 3),
            "dist_to_cavity_mm": round(dmin, 2) if np.isfinite(dmin) else "inf",
            "flags": ";".join(flags),
            "rationale": why,
            "thick_axis": thick,
        })
    return inst, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--suffix", default="", help="'_seg' for expert masks in clean/, '' for predictions")
    p.add_argument("--splits")
    p.add_argument("--subset", default="test")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    seg_dir = Path(args.seg_dir)
    if args.splits:
        cases = json.loads(Path(args.splits).read_text())[args.subset]
    else:
        tail = f"{args.suffix}.nii.gz"
        cases = sorted(f.name[: -len(tail)] for f in seg_dir.glob(f"*{tail}"))
    rules = FigoRules()
    all_rows = []
    for cid in cases:
        nii = load_nifti(seg_dir / f"{cid}{args.suffix}.nii.gz")
        seg = np.rint(np.asarray(nii.dataobj)).astype(np.uint8)
        _, rows = classify_case(seg, nii.header.get_zooms()[:3], rules)
        all_rows += [{"case_id": cid, **r} for r in rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0]) if all_rows else ["case_id", "fibroid_id", "figo"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    out.with_suffix(".rules.json").write_text(json.dumps(asdict(rules), indent=1))
    print(f"{len(all_rows)} fibroids in {len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
