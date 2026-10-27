# MedNeXt: Baseline Implementation

**Owner:** Anagha (reassigned from Purnima) · **Platform:** Kaggle, T4 (Save & Run All commit) · **Status:** done, scored on `val`

## Model
`figo/models/mednext.py`: re-implementation of MedNeXt-S (k=3), ConvNeXt-style blocks (depthwise conv, expansion,
GELU) with residual up/down-sampling and deep supervision. 5.5M parameters. Not the authors' code.

## Settings (`K1_kaggle_train`)
| Setting | Value |
|---|---|
| `MODEL` | `mednext` |
| `EPOCHS` | 150 (×250 iterations) |
| Patch × batch | 192×192×16 × 2 |
| Optimizer | SGD Nesterov, lr 1e-2, poly decay (defaults) |
| Loss | Dice + CE with deep supervision |
| Seed / split | 0 / team `splits.json` (210 train, 30 val, 60 test) |

## Timing
181 s/epoch training, +37 s on validation epochs (every 10). 150 epochs ≈ 7.7 h on one T4 (≈ 8 h including val scoring; fits one Kaggle commit run).
Peak GPU memory ≈ 7.9 GB (Colab smoke test).

## Training curve (val Dice during training, every 10 epochs)
| Epoch | Wall | Cavity | Myoma | Train loss |
|---|---|---|---|---|
| 9 | 0.59 | 0.54 | 0.23 | 0.66 |
| 29 | 0.69 | 0.63 | 0.37 | 0.44 |
| 49 | 0.70 | 0.64 | 0.39 | 0.39 |
| 89 | 0.80 | 0.68 | 0.56 | 0.34 |
| 119 | 0.82 | 0.71 | 0.55 | 0.33 |
| 149 | 0.85 | 0.72 | 0.59 | 0.31 |

Best smoothed val Dice 0.513 (epoch 149). Cyst Dice stayed 0 for the whole run.

## Validation results (`figo.evaluate`, 30 cases, native resolution)
| Macro Dice | Wall | Cavity | Myoma | Uterus | Cyst |
|---|---|---|---|---|---|
| **0.710** | 0.835 | 0.708 | 0.587 | 0.854 | 0.000 |

| Cavity/wall HD95 | Serosa HD95 | Fibroid recall / precision | FIGO exact / coarse agreement | FIGO κ exact / coarse |
|---|---|---|---|---|
| 7.5 mm | 25.3 mm | 0.50 / 0.66 (55 of 111) | 0.73 / 0.84 | 0.54 / 0.65 |

No myoma predicted in 1 case.

**vs published:** 0.72 internal / 0.49 external Dice; our macro 0.710 is in line with the internal figure.
**vs DA-VNet:** better on macro, wall, cavity, myoma, cavity/wall HD95 and FIGO κ with 4.5× fewer parameters, but
twice the time per epoch. Fibroid recall is ~0.50 for both, so the misses are not model-specific.

## Reproduce
Kaggle notebook from `notebooks/K1_kaggle_train.ipynb` with the settings above, input `umd-prepared`, GPU T4,
Internet on, **Save Version → Save & Run All**. Last cell with `--subset val` and `results_val`. Outputs on Drive:
`runs/mednext/`, `preds/mednext/`, `results_val/mednext_*`.
