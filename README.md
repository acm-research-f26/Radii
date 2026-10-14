# Automated FIGO classification of uterine fibroids from MRI

This is a three-stage pipeline on the UMD dataset (300 sagittal T2w pelvic MRIs):

1. **Segment** the uterine wall, cavity, fibroids and nabothian cysts (`figo.train`, `figo.infer`, nnU-Net).
2. **Separate** the single fibroid label into individual fibroids (`figo.instances`).
3. **Type** each fibroid FIGO 0–8 with interpretable geometric rules (`figo.stage3`).

> **Decision support, not diagnosis.** UMD ships no FIGO labels. All FIGO types here are derived from
> geometry, and every call comes with a plain-language rationale and caution flags. The radiologist decides.

The full project brief is in [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md), and the team's Colab rules and GPU schedule are in [`docs/COLAB_PLAYBOOK.md`](docs/COLAB_PLAYBOOK.md).

## Running on Colab

| Notebook | Runtime | What it does |
|---|---|---|
| `notebooks/00_setup_data.ipynb` | CPU, once | Loads all 300 cases, writes a manifest, the fixed split and the training cache to Drive |
| `notebooks/01_train_custom_model.ipynb` | T4 | Trains, then tests U-Net / MedNeXt / DA-VNet / AKS-UNet (set `MODEL`) |
| `notebooks/02_nnunet.ipynb` | T4 | Trains, then tests nnU-Net v2 3d_fullres or ResEnc M (set `VARIANT`) |
| `notebooks/03_select_and_figo.ipynb` | CPU | Ranks the models, runs Stage 2/3 and the FIGO distribution check |

Every training run checkpoints to Drive, so after a disconnect you just re-run the notebook and it resumes.
If the repo is private, put a GitHub token in `REPO_URL` in the setup cell.

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
