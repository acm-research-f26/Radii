# Colab playbook (one shared account)

## Drive layout (`MyDrive/figo/`)

```
UMD.zip                      raw download (put here once)
data/clean/                  cleaned NIfTI (figo.prepare)
data/manifest.csv            per-case spacing, labels, mislabeled-gzip flag, fibroid count
data/splits.json             THE split - every model uses it; never regenerate mid-project
data/cache_0.9mm/            training cache for the custom models
runs/<model>/                last.pt (resume), best.pt, log.csv, config.json
nnUNet_results/              nnU-Net checkpoints (resume with --c)
preds/<model>/               test-set predictions, <case_id>.nii.gz
results/                     <model>_seg.csv / _figo_agreement.csv + .summary.json
```

## Rules
- **One GPU session at a time.** Book a slot in the schedule below before you connect, and
  disconnect (Runtime → Disconnect and delete runtime) when done. An idle GPU still burns units.
- Use a **T4** for everything. Notebook 00 and notebook 03 only need a **CPU** runtime.
- Set `MAX_HOURS` a bit below the time you have booked. Training stops cleanly at an epoch
  boundary, and re-running the notebook resumes it.
- Keep `EPOCHS` identical for all custom models, and identical for both nnU-Net variants, so the benchmark is fair.
- After 3 to 5 epochs, read `epoch_sec` in `runs/<model>/log.csv` (nnU-Net prints epoch time) and
  multiply it out to plan the remaining sessions. If the total is too long, reduce `EPOCHS` for **all**
  models rather than for just one.
- Never edit `splits.json`. Test cases must not be used for any tuning decision.

## Suggested order
1. Notebook 00 on CPU (anyone, once).
2. Short smoke run of each model (`EPOCHS=2`, into a scratch `runs/` folder). This catches OOM and path problems cheaply.
3. Full runs: nnU-Net 3d_fullres → nnU-Net ResEnc M → MedNeXt → AKS-UNet → DA-VNet → U-Net.
4. Inference and evaluation cells at the end of each notebook. These are short and run on a T4.
5. Notebook 03 on CPU to rank the models, then run the FIGO stage.

## Schedule

| Slot (date, time) | Person | Notebook / model | Epochs done | Notes |
|---|---|---|---|---|
| Week 1 | Ira | `00_setup_data` (CPU runtime) | – | Check 300/300 loaded, ~33 mislabeled-gzip segs, spacing ~0.45×0.45×6.6 mm, split 210/30/60 |
| Week 1 | Purnima | `02_nnunet` smoke test, `VARIANT='3d_fullres'` | | `SMOKE=True`, after 00 is done |
| Week 1 | Purnima | `02_nnunet` smoke test, `VARIANT='resenc'` | | `SMOKE=True` |
| Week 1 | Purnima | `01_train_custom_model` smoke test, `MODEL='mednext'` | | `SMOKE=True` |
| Week 1 | Anagha | Kaggle setup: private dataset from 00, `GITHUB_TOKEN` secret only if the repo goes private | – | |
| Week 1 | Anagha | `K1_kaggle_train` smoke test, `MODEL='unet3d'` | | `SMOKE=True` |
| Week 1 | Anagha | `K1_kaggle_train` smoke test, `MODEL='aksunet'` | | `SMOKE=True` |
| Week 1 | Snigdha | `01_train_custom_model` smoke test, `MODEL='davnet'` | | `SMOKE=True` |
| | | | | |

## Week 2 results (validation set, 30 cases)

Scored with `figo.evaluate` + `figo.figo_eval agreement` on `--subset val`; the test set is untouched until week 4.
Macro Dice = mean of wall, cavity and myoma Dice. Fill in your row when your model is scored.

| Model | Who | Where | Epochs | s/epoch | Macro Dice | Wall | Cavity | Myoma | Uterus | Cavity/wall HD95 (mm) | Fibroid recall | FIGO κ coarse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nnU-Net 3d_fullres | Purnima | Colab A | | | | | | | | | | |
| nnU-Net ResEnc M | Ira | Colab B | | | | | | | | | | |
| MedNeXt | Anagha | Kaggle T4 | 150 | 181 | 0.710 | 0.835 | 0.708 | 0.587 | 0.854 | 7.5 | 0.50 | 0.65 |
| AKS-UNet | Snigdha | Kaggle T4 | 150 | | | | | | | | | |
| DA-VNet | Anagha | Kaggle T4 | 150 | 86 | 0.664 | 0.793 | 0.661 | 0.537 | 0.848 | 8.8 | 0.51 | 0.59 |
| U-Net | Snigdha | Kaggle T4 | 150 | | | | | | | | | |

Notes:
- Published single-structure numbers are not macro Dice: DA-VNet's 0.878 (Liu et al. 2025) is fibroid-only Dice on a
  private axial dataset, so compare it with the Myoma column, not Macro.
- Both custom models so far find only ~50% of expert fibroids (mostly small ones); worth a week-3 look.

## Kaggle notes

- Kaggle unpacks uploaded `.nii.gz` files to `.nii` (and `.pt` checkpoints into folders, since they are zip
  archives). `figo.io.resolve_nifti` now accepts either spelling, so `figo.infer`, `figo.evaluate`, `figo.stage3`
  and `figo.figo_eval` run directly on a Kaggle `clean/` folder. To re-upload a checkpoint, zip its unpacked folder
  back up, or attach the training notebook's output instead of uploading `best.pt`.
- Commit runs (Save Version -> Save & Run All) keep `/kaggle/working`; an interactive session started by clicking a
  cell in the editor uses GPU quota in parallel, so stop it and close the editor tab.
