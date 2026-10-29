# DA-VNet: Baseline Implementation

**Owner:** Anagha · **Platform:** Kaggle, T4 (Save & Run All commit) · **Status:** done, scored on `val`

## Model
`figo/models/davnet.py`: V-Net residual stages (1/2/3/3/3 convs, PReLU, strided-conv downsampling), additive
attention gates on every skip (Oktay et al. 2018) and deep supervision at each decoder stage. 5×5 in-plane kernels,
anisotropic layout shared with the other custom models. 24.8M parameters. Re-implemented from the paper's description
([Liu et al., *Front. Oncol.* 2025](https://doi.org/10.3389/fonc.2025.1522399)), not the authors' code.

## Settings (`K1_kaggle_train`)
| Setting | Value |
|---|---|
| `MODEL` | `davnet` |
| `EPOCHS` | 150 (×250 iterations) |
| Patch × batch | 192×192×16 × 2 |
| Optimizer | SGD Nesterov, lr 1e-2, poly decay (defaults) |
| Loss | Dice + CE with deep supervision |
| Seed / split | 0 / team `splits.json` (210 train, 30 val, 60 test) |

## Timing
86 s/epoch training, +25 s on validation epochs (every 10). 150 epochs ≈ 3.8 h on one T4. Peak GPU memory ≈ 1.9 GB
(Colab smoke test).

## Training curve (val Dice during training, every 10 epochs)
| Epoch | Wall | Cavity | Myoma | Train loss |
|---|---|---|---|---|
| 9 | 0.47 | 0.34 | 0.14 | 0.84 |
| 29 | 0.68 | 0.56 | 0.35 | 0.57 |
| 49 | 0.73 | 0.61 | 0.44 | 0.46 |
| 89 | 0.77 | 0.64 | 0.50 | 0.40 |
| 119 | 0.77 | 0.65 | 0.50 | 0.35 |
| 149 | 0.78 | 0.65 | 0.54 | 0.29 |

Plateaued around epochs 50–90, then improved as the learning rate fell. Best smoothed val Dice 0.508 (epoch 149).

## Validation results (`figo.evaluate`, 30 cases, native resolution)
| Macro Dice | Wall | Cavity | Myoma | Uterus | Cyst |
|---|---|---|---|---|---|
| **0.664** | 0.793 | 0.661 | 0.537 | 0.848 | 0.277 |

| Cavity/wall HD95 | Serosa HD95 | Fibroid recall / precision | FIGO exact / coarse agreement | FIGO κ exact / coarse |
|---|---|---|---|---|
| 8.8 mm | 18.2 mm | 0.51 / 0.63 (56 of 111) | 0.68 / 0.82 | 0.47 / 0.59 |

No cavity predicted in 2 cases, no myoma in 4.

**vs published:** the paper reports fibroid-only Dice 0.878 on a private axial dataset (245 patients, 600 epochs,
single label). Our multi-label UMD run reaches 0.848 on the whole uterus but 0.537 on fibroids, mostly because about
half the (small) fibroids are missed.

## Kaggle run log
Full log from the Kaggle run (every epoch, then the val score block): [`davnet_kaggle_train.log`](kaggle_logs/davnet_kaggle_train.log), [`davnet_kaggle_val.log`](kaggle_logs/davnet_kaggle_val.log).

## Reproduce
Kaggle notebook from `notebooks/K1_kaggle_train.ipynb` with the settings above, input `umd-prepared`, GPU T4,
Internet on, **Save Version → Save & Run All**. Last cell with `--subset val` and `results_val`. Outputs on Drive:
`runs/davnet/`, `preds/davnet/`, `results_val/davnet_*`.
