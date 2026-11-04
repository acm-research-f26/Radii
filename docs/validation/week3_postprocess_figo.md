# Week 3: post-processing + FIGO error table (validation, 30 cases)

Run with `notebooks/04_postprocess_figo.ipynb` (CPU) on the `fold_0/validation/` predictions used for the ensemble.
Full outputs on Drive: `results_val/postprocess/` (`postprocess_comparison.csv`, `figo_error_table.csv`,
`figo_error_table_pp.csv`, per-model `_seg` and `_figo_agreement` CSVs + summaries).
Missed fibroids are counted against Stage 3 on the expert masks. Macro Dice = mean of wall, cavity, myoma.

## Table 1: post-processing effect

| Model | Post-proc. | Macro Dice | Cavity Dice | Cavity/wall HD95 (mm) | Fibroid recall | FIGO κ (coarse) |
|---|---|---|---|---|---|---|
| nnU-Net ResEnc M | none | 0.736 | 0.727 | 6.23 | 0.568 | 0.763 |
| | on | 0.720 | 0.703 | 6.23 | 0.550 | 0.760 |
| nnU-Net 3d_fullres | none | 0.728 | 0.747 | 6.92 | 0.568 | 0.696 |
| | on | 0.722 | 0.723 | 15.27 | 0.550 | 0.692 |
| ResEnc FT control | none | 0.738 | 0.728 | 6.11 | 0.595 | 0.813 |
| | on | 0.723 | 0.705 | 6.08 | 0.577 | 0.810 |
| Ensemble (ResEnc + 3d_fullres) | none | 0.744 | 0.747 | 6.07 | 0.577 | 0.801 |
| | on | 0.723 | 0.707 | 14.73 | 0.559 | 0.799 |

Full table (wall, myoma, uterus Dice, exact κ, missed counts):

| Model | Post-proc. | Macro | Wall | Cavity | Myoma | Uterus | HD95 | Recall | κ coarse | κ exact | Missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResEnc M | none | 0.736 | 0.838 | 0.727 | 0.643 | 0.879 | 6.231 | 0.568 | 0.763 | 0.625 | 48 |
| ResEnc M | on | 0.720 | 0.817 | 0.703 | 0.641 | 0.870 | 6.225 | 0.550 | 0.760 | 0.641 | 50 |
| 3d_fullres | none | 0.728 | 0.833 | 0.747 | 0.604 | 0.847 | 6.923 | 0.568 | 0.696 | 0.591 | 48 |
| 3d_fullres | on | 0.722 | 0.830 | 0.723 | 0.613 | 0.858 | 15.265 | 0.550 | 0.692 | 0.605 | 50 |
| FT control | none | 0.738 | 0.838 | 0.728 | 0.648 | 0.881 | 6.113 | 0.595 | 0.813 | 0.647 | 45 |
| FT control | on | 0.723 | 0.818 | 0.705 | 0.647 | 0.871 | 6.081 | 0.577 | 0.810 | 0.662 | 47 |
| Ensemble | none | 0.744 | 0.842 | 0.747 | 0.644 | 0.877 | 6.073 | 0.577 | 0.801 | 0.654 | 47 |
| Ensemble | on | 0.723 | 0.819 | 0.707 | 0.643 | 0.871 | 14.733 | 0.559 | 0.799 | 0.670 | 49 |

## Table 2: FIGO errors and missed fibroids (no post-processing)

| Model | Top errors (expert → model) | Missed < 1 ml | Missed ≤ 2 slices |
|---|---|---|---|
| nnU-Net ResEnc M | 5→6 ×4, 5→4 ×3, 6→5 ×2 | 38 / 48 | 44 / 48 |
| nnU-Net 3d_fullres | 5→6 ×4, 5→4 ×4, 6→5 ×2 | 38 / 48 | 44 / 48 |
| ResEnc FT control | 5→6 ×4, 5→4 ×3, 6→5 ×2 | 35 / 45 | 41 / 45 |
| Ensemble | 5→6 ×5, 5→4 ×3, 6→5 ×2 | 38 / 47 | 43 / 47 |

Median volume of missed fibroids is 0.16–0.17 ml vs 1.45–1.59 ml for found ones.

## Findings
1. **Post-processing hurts** every model (macro −0.006 to −0.021, cavity ≈ −0.02, +2 missed fibroids each; HD95
   roughly doubles for 3d_fullres and the ensemble). Exact κ rises slightly (+0.014 to +0.016). Likely cause:
   keeping only the largest connected uterus on 6.6 mm slices drops real tissue. Not recommended as is.
2. **Missed fibroids are tiny and thin** (~80% < 1 ml, ~92% on ≤ 2 slices), so post-processing cannot raise recall.
3. **FIGO errors cluster around type 5** (5→6, 5→4, 6→5), matching the two fragile thresholds from the
   sensitivity analysis: the 50% intramural cut (5 vs 6) and serosal contact `touch_min_frac` (5 vs 4).
4. **Best raw model:** the ensemble on Dice/HD95; the FT control matches or beats it on FIGO κ coarse (0.813 vs 0.801)
   and recall (0.595 vs 0.577), so part of the gain may come from extra training rather than ensembling.
