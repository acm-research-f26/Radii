# Project Brief: Automated FIGO Classification of Uterine Fibroids from MRI

## 1. What we're building
A three-stage pipeline that takes a uterine MRI and assigns each fibroid its FIGO type (0 to 8):

1. Segment the fibroid, uterine wall, cavity, and cyst (deep-learning segmentation)
2. Separate the merged fibroid region into individual fibroids (connected components)
3. Classify each fibroid's FIGO type using geometric rules (overlap with cavity, contact with wall)

**Framing (keep this throughout):** This is a decision-support tool for radiologists, not a replacement. FIGO typing is currently manual and varies between readers. Our tool gives a fast, consistent, interpretable second read, and the clinician makes the final call.

## 2. Clinical background
- Uterine fibroids are noncancerous muscular growths in the uterine wall, present in up to 70 to 80% of women by age 50.
- FIGO type drives treatment: hysteroscopic resection (types 0 to 2), laparoscopic myomectomy (3 to 4), surgical excision (5 to 7).
- FIGO type is defined purely by geometry: relationship to the cavity (endometrium), relationship to the serosa (outer wall), and percent intramural.

FIGO rules (geometric):

| Type | Definition |
|---|---|
| 0 | entirely intracavitary (pedunculated) |
| 1 | contacts cavity, <50% intramural |
| 2 | contacts cavity, >=50% intramural |
| 3 | 100% intramural but touches the endometrium |
| 4 | intramural, touches neither cavity nor serosa |
| 5 | touches serosa, >=50% intramural |
| 6 | touches serosa, <50% intramural |
| 7 | subserosal pedunculated |
| 8 | other (e.g. cervical) |

## 3. Dataset: UMD (Uterine Myoma MRI Dataset)
Public dataset of 300 patients with sagittal T2-weighted pelvic MRI and expert pixel-level masks. Download UMD.zip from Figshare. Descriptor: *Scientific Data* 2024.

- File layout: `UMD_<id>/UMD_<id>_seg.nii.gz` (mask) and `..._t2.nii.gz` (image), NIfTI format.
- Mask labels: 0 background, 1 uterine wall, 2 uterine cavity, 3 myoma/fibroid, 4 nabothian cyst.
- About 1,077 individual fibroids in total across the 300 patients.

Critical constraints (must handle):
- **No FIGO labels ship with UMD**, only the segmentation masks. FIGO targets are therefore geometry-derived, and we cannot report true per-case FIGO accuracy. Validation compares our predicted FIGO distribution to published clinical frequencies (type 4 ~30 to 55%, type 3 ~13 to 34%, rare types ~3 to 5%), ideally plus a radiologist spot-check of a small subset.
- **Thick, anisotropic slices** (~0.45 x 0.45 x 6.6 mm; fibroids often span only 2 to 3 slices). In-plane detail is good, but through-plane depth is unreliable. Coarse FIGO distinctions (submucosal / intramural / subserosal) are reliable. The fine ones (1 vs 2, 5 vs 6, 3 vs 4 contact) are resolution-limited and must be reported with caution.
- All fibroids share one mask label (semantic, not instance), which is why Stage 2 (connected components) is needed.
- About 33 `_seg` files are uncompressed NIfTI mislabeled as gzip. They need a defensive loader (try gzip, fall back to raw) so that all 300 cases load.

## 4. Stage 1: baseline segmentation models

| Model | Architecture | Published UMD Dice (target to reproduce) |
|---|---|---|
| nnU-Net v2 (3D full-res) | Self-configuring 3D U-Net | 0.76 macro / 0.70 myoma |
| nnU-Net v2 (Residual Encoder) | 3D U-Net + residual blocks | 0.82 uterus / 0.80 fibroid |
| U-Net (standard 3D) | Classic encoder-decoder (baseline) | ~0.40 external |
| MedNeXt | ConvNeXt-based | 0.72 internal / 0.49 external |
| DA-VNet | V-Net + deep supervision + attention | 0.88 |
| AKS-UNet | U-Net + adaptive kernel + spatial attention | 0.83 (mIoU 0.72) |

Transformer and foundation-model variants (Swin-UNETR, MedSAM-2) are out of scope for the baselines.

Model selection metrics (not just Dice): (1) boundary accuracy at the cavity/wall interface (Hausdorff / surface distance), (2) per-structure Dice for cavity and wall, not just fibroid, and (3) downstream FIGO agreement, meaning which model's masks best reproduce the FIGO types from the expert masks.

## 5. Implementation environment
Google Colab (Pro or Pro+), one shared account, GPU training. Use the T4 GPU, cap the number of epochs, and checkpoint to Google Drive on every run. Because there is one account, only one GPU session can run at a time. A CPU is fine for Stages 2 and 3.

## 6. Build order
1. Set up Colab and data loading (handle the 33 mislabeled files), then confirm labels and voxel spacing.
2. Implement and train the segmentation models (one per team member), and benchmark them against the published Dice.
3. Select the best 2 to 3 models using the metrics above.
4. Stage 2: connected-components instance separation on the fibroid label.
5. Stage 3: compute per-fibroid geometry and apply the FIGO rules.
6. Validate the FIGO output against published clinical distributions, then write up.
