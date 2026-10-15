# Automated FIGO classification of uterine fibroids from MRI

## Start here

**First notebook (Ira):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acm-research-f26/Radii/blob/main/notebooks/00_setup_data.ipynb) on a **CPU runtime** (Runtime → Change runtime type → CPU). It needs no GPU.

**Prerequisites**
- Download `UMD.zip` from Figshare (link in the UMD *Scientific Data* 2024 descriptor) and upload it to `MyDrive/figo/UMD.zip`.
- Share the `MyDrive/figo/` folder with the second Colab account (Editor access), so both accounts read and write the same data and checkpoints.

**Run order:** `00` → `01` / `02` / `K1` (in parallel, one GPU session at a time per account) → `03`

| Person | Owns |
|---|---|
| Ira | `00_setup_data` + nnU-Net 3d_fullres (`02_nnunet`, `VARIANT = '3d_fullres'`) |
| Purnima | nnU-Net ResEnc (`02_nnunet`, `VARIANT = 'resenc'`) + MedNeXt (`01_train_custom_model`, `MODEL = 'mednext'`) |
| Anagha | U-Net + AKS-UNet on Kaggle (`K1_kaggle_train`, `MODEL = 'unet3d'` / `'aksunet'`) |
| Snigdha | DA-VNet (`01_train_custom_model`, `MODEL = 'davnet'`) + FIGO validation plan (`03_select_and_figo`) |

**A successful `00` run prints**
- `Loaded 300/300 cases; 0 failures`
- about **33** segmentation files reported as uncompressed despite `.gz` (mislabeled gzip, handled automatically)
- most common spacing about **0.45 × 0.45 × 6.6 mm**
- `Split sizes: train 210, val 30, test 60`

If any of these is off, stop and post the output in the team chat before anyone starts training. `data/splits.json` is then fixed for the whole project.

Team rules are in [`CLAUDE.md`](CLAUDE.md), and the GPU schedule is in [`docs/COLAB_PLAYBOOK.md`](docs/COLAB_PLAYBOOK.md).

## Overview

This is a three-stage pipeline on the UMD dataset (300 sagittal T2w pelvic MRIs):

1. **Segment** the uterine wall, cavity, fibroids and nabothian cysts (`figo.train`, `figo.infer`, nnU-Net).
2. **Separate** the single fibroid label into individual fibroids (`figo.instances`).
3. **Type** each fibroid FIGO 0–8 with interpretable geometric rules (`figo.stage3`).

> **Decision support, not diagnosis.** UMD ships no FIGO labels. All FIGO types here are derived from
> geometry, and every call comes with a plain-language rationale and caution flags. The radiologist decides.

The full project brief is in [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md), and the team's Colab rules and GPU schedule are in [`docs/COLAB_PLAYBOOK.md`](docs/COLAB_PLAYBOOK.md).

## Running on Colab

| Notebook | Open | Runtime | What it does |
|---|---|---|---|
| `notebooks/00_setup_data.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acm-research-f26/Radii/blob/main/notebooks/00_setup_data.ipynb) | CPU, once | Loads all 300 cases, writes a manifest, the fixed split and the training cache to Drive |
| `notebooks/01_train_custom_model.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acm-research-f26/Radii/blob/main/notebooks/01_train_custom_model.ipynb) | T4 | Trains, then tests U-Net / MedNeXt / DA-VNet / AKS-UNet (set `MODEL`) |
| `notebooks/02_nnunet.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acm-research-f26/Radii/blob/main/notebooks/02_nnunet.ipynb) | T4 | Trains, then tests nnU-Net v2 3d_fullres or ResEnc M (set `VARIANT`) |
| `notebooks/03_select_and_figo.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acm-research-f26/Radii/blob/main/notebooks/03_select_and_figo.ipynb) | CPU | Ranks the models, runs Stage 2/3 and the FIGO distribution check |
| `notebooks/K1_kaggle_train.ipynb` | Kaggle (import the notebook from GitHub) | Kaggle T4/P100 | Free-GPU training of the custom models with resume across 12 h sessions (set `MODEL`) |

Every training run checkpoints to Drive, so after a disconnect you just re-run the notebook and it resumes.
The repo is public, so the notebooks clone it without a token. If it is ever made private, create a
fine-grained, **read-only** GitHub token (Contents: read, this repo only) and store it as `GITHUB_TOKEN` in
**Colab Secrets** (key icon in the sidebar) or **Kaggle Secrets** (Add-ons → Secrets). Never paste a token into a
notebook cell; notebooks get shared and committed.

## Stage 1 baselines

| Model | Where | Implementation | Published UMD Dice |
|---|---|---|---|
| nnU-Net v2 3d_fullres | `02_nnunet` | official `nnunetv2`, default plans | 0.76 macro / 0.70 myoma |
| nnU-Net v2 ResEnc | `02_nnunet` | official `nnunetv2`, `nnUNetPlannerResEncM` (fits a 16 GB T4) | 0.82 uterus / 0.80 fibroid |
| U-Net (3D) | `figo/models/unet3d.py` | classic 3D U-Net (instance norm, no deep supervision) | ~0.40 (external) |
| MedNeXt | `figo/models/mednext.py` | re-implementation of MedNeXt-S (k=3) | 0.72 int. / 0.49 ext. |
| DA-VNet | `figo/models/davnet.py` | V-Net + attention-gated skips + deep supervision, built from the paper's description | 0.88 |
| AKS-UNet | `figo/models/aksunet.py` | U-Net + selective-kernel convs + spatial attention, built from the paper's description | 0.83 |

**What "reproduce" can mean here:**
- MedNeXt, DA-VNet and AKS-UNet are our own implementations of the published designs, not the authors' code, so layer-level details may differ.
- The published numbers come from different splits, epoch budgets and sometimes external test sets.
- Our runs use one fixed split, a capped epoch budget, and native-resolution scoring for every model.

So we expect to land in the same ballpark as the published Dice, not to match it exactly. Compare models against each other first. The U-Net figure of 0.40 is an *external* validation number, and our internal test set should score higher.

### Design decisions shared by all custom models
- **Anisotropy.** Volumes are reoriented to canonical and the 6.6 mm slice axis is moved last, giving `(H, W, D)`. Only the in-plane dimensions are resampled (to 0.9 mm by default); slices are never interpolated. The first two stages convolve and downsample in-plane only (`(3,3,1)` kernels), as nnU-Net plans for anisotropic data. The default patch is 192×192×16.
- **Training.** The recipe follows nnU-Net: Dice + CE loss, deep supervision (except for the plain U-Net), SGD with Nesterov momentum and a poly learning-rate schedule, AMP, foreground-oversampled patches, and in-plane rotation/scale, flip and intensity augmentation.
- **Fair comparison.** One `splits.json` is shared by all models, including nnU-Net through `splits_final.json`, with 70/10/20 train/val/test stratified by fibroid count. Every model is evaluated by the same `figo.evaluate` at native resolution.
- **Colab robustness.** `last.pt` is written atomically to Drive every epoch, a `--max-hours` budget stops training cleanly, and the nnU-Net trainer saves every 5 epochs instead of every 50.

### Model selection (`figo.select_models`)
Models are ranked by their mean rank on the brief's three criteria:
1. HD95 of the cavity/wall interface, i.e. the endometrial boundary that decides FIGO 0–3.
2. Mean Dice of cavity and wall.
3. Coarse FIGO kappa between FIGO computed from the model's masks and FIGO computed from the expert masks.

The table also shows myoma Dice, serosal-surface HD95 (which decides FIGO 4–7) and fibroid detection recall.

## Data quirks handled
- About 33 `*_seg.nii.gz` files are really uncompressed. `figo.io.load_nifti` checks the gzip magic bytes and falls back to raw parsing, and `figo.prepare` rewrites every case as properly gzipped NIfTI so that nnU-Net and MONAI can read them. The manifest records which files were affected.
- Labels are checked to be in {0..4}, segmentations are forced to the image's geometry, and spacing and orientation are logged for every case.

## Stages 2 and 3
- **Instances.** Connected components use 8-connectivity within a slice and face-only connectivity across slices, so fibroids that touch only diagonally across a 6.6 mm gap are not merged. Components under 50 mm³ are dropped. Fibroids that touch within a slice stay merged, which is a known limitation.
- **FIGO.** For each fibroid we measure how much of its surface (area-weighted voxel faces, so the anisotropy is handled correctly) faces cavity, myometrium, outside the serosa, or nabothian cysts.
  - The fraction protruding into the cavity or past the serosa is the proxy for "% intramural". For a sphere the 50% surface and 50% volume cuts coincide.
  - Thresholds live in `FigoRules`. Hybrids are reported as e.g. `2-5`.
  - Each fibroid gets a `rationale` and `flags`: `borderline_50pct`, `near_2_vs_3_threshold`, `near_3_vs_4_threshold`, `thin_<=2_slices` and `review`. These mark the resolution-limited fine distinctions explicitly.
- **Validation.** `figo.figo_eval distribution` compares the FIGO distribution from the expert masks with the published ranges. `figo.figo_eval agreement` measures how much each model's segmentation error changes the FIGO call. A radiologist spot-check of a subset is still needed.

## Local development
```bash
pip install -r requirements-dev.txt
pytest
```
The tests run on CPU in about 10 s and cover:
- the loader on a mislabeled `.nii.gz`
- the preprocessing round-trip
- the metrics
- all four models
- train → kill → resume → infer → evaluate → FIGO agreement → nnU-Net conversion on a fake UMD folder
- FIGO rules on synthetic uterus phantoms of known type (0, 3, 4, 5, 6, 7, 8)

## Project background: RADII

### The Problem

Uterine fibroids are noncancerous muscular growths in the wall of the uterus, and they are extremely common: they affect up to 70 to 80% of women by age 50, and cause heavy bleeding, pelvic pain, and fertility problems. A fibroid's size is one of the most consequential numbers in its care. It helps decide whether a patient qualifies for less invasive treatments, it is used to enroll patients in clinical trials, and, tracked over time, it is how clinicians judge whether a treatment is actually shrinking the fibroid.

In everyday practice that size is estimated quickly, by measuring a few diameters and applying a simple formula, rather than by tracing the fibroid in full. The problem is that this shortcut can be significantly inaccurate for fibroids that are irregular in shape. A 2025 study found that the quick estimate differed from a full, careful measurement by more than 20% in about 1 in 5 fibroids, and by more than 30% in about 1 in 10, with the largest difference reaching roughly 56%, even though the average error looked almost perfect. An error that large matters, because treatment success is often judged by whether a fibroid shrinks 20 to 30%, so a measurement that is off by that much can turn a real result into a false one, or hide a real one. Today a clinician has no reliable way to know, at the moment of measurement, whether the estimate they just recorded can be trusted.

### What RADII Is

RADII (a nod to the radii and diameters used to size a fibroid) studies the reliability of this everyday fibroid measurement. Using MRI scans that have already been outlined by expert radiologists, we treat the full traced outline as the true size and compare it against the quick clinical estimate for each fibroid. We then look at which measurable shape characteristics, such as elongation, lobulation, and overall irregularity, make the shortcut fail, and we build lightweight models that flag the fibroids whose measurement cannot be trusted. The goal is not to replace a radiologist's judgment, but to support it: to give an honest sense of how reliable an existing measurement is, so a quick estimate can be trusted when it is safe and double-checked when it is not.

### The Dataset

We use the public Uterine Myoma MRI Dataset (UMD), the largest openly available uterine fibroid MRI dataset. It contains sagittal T2-weighted pelvic MRI scans from 300 patients, spanning all fibroid types, each with expert pixel-level segmentation masks that label the uterine wall, cavity, fibroids, and cysts. Because these masks give an exact, traced measurement of every fibroid for free, and let us compute the quick clinical estimate from the same scan, UMD lets us measure the error of the shortcut directly, at scale, without needing to collect or annotate any new data ourselves.

## Related Models

* **VM-UNet:** Implementation and documentation located in [`models/vmunet/`](./models/vmunet/README.md).
