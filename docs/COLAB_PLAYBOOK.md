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
