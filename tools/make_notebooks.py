"""Regenerate notebooks/*.ipynb:  python tools/make_notebooks.py notebooks"""

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

COMMON_SETUP = '''# --- shared setup: Drive + code ---------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/figo'          # shared project folder (one Colab account)
REPO_URL = 'https://github.com/acm-research-f26/Radii.git'  # public; if made private see README (token in Colab Secrets)
REPO = '/content/figo-repo'

import os, subprocess
os.makedirs(DRIVE, exist_ok=True)
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO], check=True)
else:
    subprocess.run(['git', '-C', REPO, 'pull', '-q'], check=True)
%cd {REPO}
!pip -q install -r requirements.txt'''


TIMING = '''
# timing -> how many epochs fit the budget (row 1 = pure training, row 2 = training + validation)
import csv
log = list(csv.DictReader(open(f'{RUN}/log.csv')))
for r in log:
    print('epoch', r['epoch'], r['epoch_sec'], 's')
if len(log) >= 3:
    t_train = float(log[1]['epoch_sec'])
    t_val = max(float(log[2]['epoch_sec']) - t_train, 0)
    per_epoch = t_train + t_val / 10          # validation runs every 10 epochs
    print(f'~{per_epoch:.0f} s per epoch on average; validation ~{t_val:.0f} s')
    for h in (6, 8, 10):
        print(f'  {h:2d} GPU-hours  ->  EPOCHS = {int(h * 3600 / per_epoch)}')
!nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


setup = nb([
    ("markdown", '''
# 00 - Setup and data check (run once)

Builds the cleaned dataset, manifest, fixed train/val/test split and the training cache on Google Drive.
**CPU runtime is enough for this notebook** - don't spend GPU units here.

Before running: download `UMD.zip` from the Figshare link in the UMD *Scientific Data* (2024) descriptor
and put it at `MyDrive/figo/UMD.zip`.
'''),
    ("code", COMMON_SETUP),
    ("code", '''
# unzip to local disk (much faster than reading 600 files from Drive)
!mkdir -p /content/UMD && unzip -q -n {DRIVE}/UMD.zip -d /content/UMD
!find /content/UMD -name "*_t2.nii*" | wc -l
'''),
    ("markdown", '''
`figo.prepare` loads every case with the defensive loader (gzip-sniffing, so the ~33 `_seg.nii.gz` files that are
really uncompressed still load), validates labels {0..4} and shapes, writes properly gzipped copies, records voxel
spacing/orientation per case, counts fibroid instances, creates `splits.json` (stratified by fibroid count, seeded),
and builds the 0.9 mm in-plane training cache. Check the printed summary: all 300 should load, and the mislabeled count
should be ~33.
'''),
    ("code", '''
!python -m figo.prepare --umd-dir /content/UMD --out-dir {DRIVE}/data --cache
'''),
    ("code", '''
import csv, collections
rows = list(csv.DictReader(open(f'{DRIVE}/data/manifest.csv')))
print(len(rows), 'cases')
print('spacings:', collections.Counter(r['spacing_mm'] for r in rows).most_common(5))
print('shapes  :', collections.Counter(r['shape'] for r in rows).most_common(5))
def thick_slices(r):
    sp = [float(s) for s in r['spacing_mm'].split('x')]
    return int(r['shape'].split('x')[sp.index(max(sp))])
print('slices along thick axis:', collections.Counter(thick_slices(r) for r in rows).most_common(8))
print('mislabeled gzip segs:', sum(int(r['seg_was_mislabeled_gzip']) for r in rows))
'''),
    ("code", '''
# visual sanity check: image + mask overlay on the slice with most fibroid
import numpy as np, matplotlib.pyplot as plt, nibabel as nib
from figo.io import load_nifti
cid = rows[0]['case_id']
img = nib.as_closest_canonical(load_nifti(f'{DRIVE}/data/clean/{cid}_t2.nii.gz'))
seg = nib.as_closest_canonical(load_nifti(f'{DRIVE}/data/clean/{cid}_seg.nii.gz'))
im, sg = np.asarray(img.dataobj), np.asarray(seg.dataobj)
ax = int(np.argmax(img.header.get_zooms()[:3]))
k = int(np.argmax((sg == 3).sum(axis=tuple(a for a in range(3) if a != ax))))
take = lambda a: np.take(a, k, axis=ax)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.rot90(take(im)), cmap='gray'); axs[0].set_title(cid)
axs[1].imshow(np.rot90(take(im)), cmap='gray')
axs[1].imshow(np.rot90(np.ma.masked_equal(take(sg), 0)), cmap='tab10', vmin=0, vmax=9, alpha=0.5)
axs[1].set_title('1 wall  2 cavity  3 myoma  4 cyst'); [a.axis('off') for a in axs]; plt.show()
'''),
])

train = nb([
    ("markdown", '''
# 01 - Train a custom baseline (U-Net / MedNeXt / DA-VNet / AKS-UNet)

Runtime -> Change runtime type -> **T4 GPU**. Each team member sets `MODEL` to their model.
Training checkpoints to Drive every epoch; if Colab disconnects or units run out, **just re-run all cells** and it
resumes. Set `MAX_HOURS` a little below your remaining session so it stops cleanly.
'''),
    ("code", '''
MODEL = 'mednext'      # 'unet3d' | 'mednext' | 'davnet' | 'aksunet'
SMOKE = True           # True: 3-epoch timing run into runs/smoke_<model>. False: the real run
EPOCHS = 200           # x 250 iterations; set from the smoke-test timing, identical across models
MAX_HOURS = 3.5
PATCH = '192 192 16'   # H W D(slices). On OOM: '160 160 16' or BATCH = 1
BATCH = 2
'''),
    ("code", COMMON_SETUP),
    ("code", '''
!nvidia-smi --query-gpu=name,memory.total --format=csv
# copy the cache to local disk once per session (fast random access during training)
!rsync -a {DRIVE}/data/cache_0.9mm/ /content/cache/
RUN = f'{DRIVE}/runs/' + (f'smoke_{MODEL}' if SMOKE else MODEL)
if SMOKE:
    EPOCHS = 3
'''),
    ("code", '''
!python -m figo.train --model {MODEL} --cache-dir /content/cache --splits {DRIVE}/data/splits.json \\
    --out-dir {RUN} --epochs {EPOCHS} --max-hours {MAX_HOURS} --patch-size {PATCH} --batch-size {BATCH}
'''),
    ("code", TIMING),
    ("code", '''
# learning curves
import csv, matplotlib.pyplot as plt
log = list(csv.DictReader(open(f'{RUN}/log.csv')))
plt.plot([float(r['train_loss']) for r in log], label='train loss')
v = [(int(r['epoch']), float(r['val_mean'])) for r in log if r['val_mean']]
if v: plt.plot(*zip(*v), 'o-', label='val mean Dice')
plt.legend(); plt.xlabel('epoch'); plt.show()
'''),
    ("markdown", "## Test-set inference and evaluation (run after training has finished)"),
    ("code", '''
PRED = f'{DRIVE}/preds/{MODEL}'
RES = f'{DRIVE}/results'
!python -m figo.infer --checkpoint {RUN}/best.pt --data-dir {DRIVE}/data/clean --splits {DRIVE}/data/splits.json \\
    --subset test --out-dir {PRED}
!python -m figo.evaluate --pred-dir {PRED} --data-dir {DRIVE}/data/clean --splits {DRIVE}/data/splits.json \\
    --subset test --out {RES}/{MODEL}_seg.csv
!python -m figo.figo_eval agreement --ref-dir {DRIVE}/data/clean --pred-dir {PRED} \\
    --splits {DRIVE}/data/splits.json --subset test --out {RES}/{MODEL}_figo_agreement.csv
'''),
])

nnunet = nb([
    ("markdown", '''
# 02 - nnU-Net v2 baselines (3d_fullres and Residual Encoder M)

Runtime -> **T4 GPU**. Uses the *same* train/val split as the custom models (fold 0 of `splits_final.json`);
test cases are held out in `imagesTs`. Raw/preprocessed data live on local disk (rebuilt each session, a few minutes);
**results/checkpoints live on Drive**. `nnUNetTrainer_UMD` caps epochs and saves `checkpoint_latest` every 5 epochs
(stock nnU-Net: every 50), and re-running the train cell resumes with `--c`.

ResEnc **M** is used because it fits a 16 GB T4; ResEnc L/XL need 24 GB+.
'''),
    ("code", '''
VARIANT = '3d_fullres'   # '3d_fullres' (nnU-Net default plans) | 'resenc' (nnUNetResEncUNetMPlans)
SMOKE = True             # True: 3-epoch timing run into a separate smoke/ folder. False: the real run
EPOCHS = 250             # set from the smoke-test timing; identical for both variants and across sessions
'''),
    ("code", COMMON_SETUP + '''
!pip -q install nnunetv2'''),
    ("code", '''
import os
os.environ['nnUNet_raw'] = '/content/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/content/nnUNet_preprocessed'
os.environ['nnUNet_results'] = f'{DRIVE}/smoke/nnUNet_results' if SMOKE else f'{DRIVE}/nnUNet_results'
os.environ['UMD_NNUNET_EPOCHS'] = '3' if SMOKE else str(EPOCHS)
PLANS = 'nnUNetPlans' if VARIANT == '3d_fullres' else 'nnUNetResEncUNetMPlans'
PLANNER = [] if VARIANT == '3d_fullres' else ['-pl', 'nnUNetPlannerResEncM']

!python -m figo.nnunet convert --data-dir {DRIVE}/data/clean --splits {DRIVE}/data/splits.json \\
    --nnunet-raw $nnUNet_raw --nnunet-preprocessed $nnUNet_preprocessed
!python -m figo.nnunet install-trainer
'''),
    ("code", '''
# Preprocessing is CPU work: run up to this cell once on a CPU runtime; it is cached on Drive
# and later GPU sessions just unpack it (saves GPU units every session).
import subprocess
pre = f"/content/nnUNet_preprocessed/Dataset501_UMD/{PLANS}_3d_fullres"
tar = f"{DRIVE}/nnunet_cache/{PLANS}.tar"
if not os.path.isdir(pre):
    if os.path.exists(tar):
        !tar -xf {tar} -C /content/nnUNet_preprocessed
    else:
        subprocess.run(['nnUNetv2_plan_and_preprocess', '-d', '501', '-c', '3d_fullres',
                        '--verify_dataset_integrity', *PLANNER], check=True)
        os.makedirs(f'{DRIVE}/nnunet_cache', exist_ok=True)
        !tar -cf {tar} -C /content/nnUNet_preprocessed Dataset501_UMD
!cat /content/nnUNet_preprocessed/Dataset501_UMD/{PLANS}.json | python -c "import json,sys; c=json.load(sys.stdin)['configurations']['3d_fullres']; print('patch', c['patch_size'], 'spacing', c['spacing'], 'batch', c['batch_size'])"
'''),
    ("code", '''
fold_dir = f"{os.environ['nnUNet_results']}/Dataset501_UMD/nnUNetTrainer_UMD__{PLANS}__3d_fullres/fold_0"
resume = ['--c'] if os.path.exists(f'{fold_dir}/checkpoint_latest.pth') else []
print('resuming' if resume else 'starting fresh', fold_dir)
!nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_UMD -p {PLANS} {' '.join(resume)}
'''),
    ("code", '''
# timing: nnU-Net logs one "Epoch time" line per epoch (the first is slower: warm-up)
!grep -h "Epoch time" {fold_dir}/training_log_*.txt
!nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
!nproc
'''),
    ("markdown", "## Test-set inference and evaluation (after training)"),
    ("code", '''
NAME = f'nnunet_{VARIANT}'
PRED = f'{DRIVE}/preds/{NAME}'
RES = f'{DRIVE}/results'
!nnUNetv2_predict -i $nnUNet_raw/Dataset501_UMD/imagesTs -o {PRED} -d 501 -c 3d_fullres -f 0 \\
    -tr nnUNetTrainer_UMD -p {PLANS} -chk checkpoint_best.pth
!python -m figo.evaluate --pred-dir {PRED} --data-dir {DRIVE}/data/clean --splits {DRIVE}/data/splits.json \\
    --subset test --out {RES}/{NAME}_seg.csv
!python -m figo.figo_eval agreement --ref-dir {DRIVE}/data/clean --pred-dir {PRED} \\
    --splits {DRIVE}/data/splits.json --subset test --out {RES}/{NAME}_figo_agreement.csv
'''),
])

setup["cells"] += nb([
    ("markdown", '''
## Optional: publish the prepared data as a private Kaggle dataset (for Kaggle training)
Get `kaggle.json` from kaggle.com -> Settings -> API -> *Create New Token*, and upload it when asked.
Then on Kaggle: the dataset's Settings -> Sharing -> add teammates.
'''),
    ("code", '''
!pip -q install kaggle
import json, shutil
from google.colab import files
os.makedirs('/root/.kaggle', exist_ok=True)
files.upload()                                   # choose kaggle.json
shutil.move('kaggle.json', '/root/.kaggle/kaggle.json'); os.chmod('/root/.kaggle/kaggle.json', 0o600)
user = json.load(open('/root/.kaggle/kaggle.json'))['username']
!rm -rf /content/kaggle_ds && mkdir -p /content/kaggle_ds
!cp -r {DRIVE}/data/cache_0.9mm {DRIVE}/data/clean {DRIVE}/data/splits.json {DRIVE}/data/manifest.csv /content/kaggle_ds/
json.dump({"title": "umd-prepared", "id": f"{user}/umd-prepared", "licenses": [{"name": "other"}]},
          open('/content/kaggle_ds/dataset-metadata.json', 'w'))
!kaggle datasets create -p /content/kaggle_ds --dir-mode zip    # private by default
'''),
])["cells"]

figo = nb([
    ("markdown", '''
# 03 - Model selection, Stage 2 (instances) and Stage 3 (FIGO) - CPU runtime

UMD has **no FIGO labels**: the FIGO types below are derived from masks by geometric rules. We validate
(a) the rule-derived distribution on the expert masks against published clinical frequencies, and
(b) each model by how well its masks reproduce the expert-mask FIGO calls. A radiologist spot-check of a
small subset is still needed before any accuracy claim. Decision support only.
'''),
    ("code", COMMON_SETUP),
    ("code", '''
# 1) rank Stage 1 models on cavity/wall boundary, cavity+wall Dice and downstream FIGO agreement
!python -m figo.select_models --results-dir {DRIVE}/results
'''),
    ("code", '''
# 2) FIGO types from the EXPERT masks on all 300 cases, then compare with published frequencies
!python -m figo.stage3 --seg-dir {DRIVE}/data/clean --suffix _seg --out {DRIVE}/results/figo_expert_all.csv
!python -m figo.figo_eval distribution --csv {DRIVE}/results/figo_expert_all.csv
'''),
    ("code", '''
# 3) FIGO from a selected model's predicted masks (test set) + distribution
BEST = 'nnunet_resenc'   # set after step 1
!python -m figo.stage3 --seg-dir {DRIVE}/preds/{BEST} --out {DRIVE}/results/figo_{BEST}_test.csv
!python -m figo.figo_eval distribution --csv {DRIVE}/results/figo_{BEST}_test.csv
'''),
    ("code", '''
# 4) per-fibroid report with rationale and caution flags (what a radiologist would review)
import csv
rows = list(csv.DictReader(open(f'{DRIVE}/results/figo_{BEST}_test.csv')))
for r in rows[:15]:
    print(f"{r['case_id']} #{r['fibroid_id']}: FIGO {r['figo']:4s} {r['volume_ml']:>7s} ml  {r['rationale']}"
          + (f"  [{r['flags']}]" if r['flags'] else ''))
'''),
])

kaggle = nb([
    ("markdown", '''
# K1 - Train a custom baseline on Kaggle (free GPU, ~30 h/week per person)

**Notebook settings (right panel):** Accelerator -> *GPU T4 x2* (only one is used; same chip as Colab, so
timings compare). Internet -> *On* (needs a phone-verified Kaggle account). **Add Input** -> the team's
`umd-prepared` dataset. Private repo? Add-ons -> Secrets -> `GITHUB_TOKEN`.

**Always run with Save Version -> *Save & Run All (Commit)*.** It runs in the background (up to 12 h) and
keeps `/kaggle/working` as the version's output. An interactive session loses everything when it stops.

**To resume** after a 12 h run: Add Input -> *Notebook Output Files* -> this notebook's latest version,
then Save & Run All again. The cell below picks up the newest `last.pt` automatically.
'''),
    ("code", '''
MODEL = 'unet3d'       # 'unet3d' | 'aksunet' (or any custom model)
SMOKE = True           # True: 3-epoch timing run. False: the real run
EPOCHS = 200           # must equal the Colab custom models' EPOCHS
MAX_HOURS = 11.0       # Kaggle stops sessions at 12 h; leave time to save outputs
PATCH = '192 192 16'
BATCH = 2
'''),
    ("code", '''
import glob, os, shutil, subprocess, zipfile
import torch

REPO_URL = 'https://github.com/acm-research-f26/Radii.git'
try:
    from kaggle_secrets import UserSecretsClient
    REPO_URL = REPO_URL.replace('https://', f"https://{UserSecretsClient().get_secret('GITHUB_TOKEN')}@")
except Exception:
    pass  # public repo
REPO = '/tmp/figo-repo'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO], check=True)
%cd {REPO}
!pip -q install -r requirements.txt

def locate(name):
    """Find a file/folder from the input dataset; unzip it if Kaggle kept it as a .zip."""
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if hits:
        return hits[0]
    zips = glob.glob(f'/kaggle/input/**/{name}.zip', recursive=True)
    if not zips:
        raise FileNotFoundError(f'{name} not found - did you add the umd-prepared dataset as input?')
    dst = f'/tmp/data/{name}'
    if not os.path.exists(dst):
        zipfile.ZipFile(zips[0]).extractall(dst)
    inner = glob.glob(f'{dst}/**/{name}', recursive=True)   # zip may contain the folder itself
    return inner[0] if inner and os.path.isdir(inner[0]) else dst

SPLITS, CACHE, CLEAN = locate('splits.json'), locate('cache_0.9mm'), locate('clean')
print(SPLITS, CACHE, CLEAN, len(os.listdir(CACHE)), 'cached volumes')

OUT = '/kaggle/working'
RUN = f'{OUT}/runs/' + (f'smoke_{MODEL}' if SMOKE else MODEL)
if SMOKE:
    EPOCHS = 3
# resume: copy the most advanced previous run attached as input
prev = glob.glob(f'/kaggle/input/**/runs/{os.path.basename(RUN)}/last.pt', recursive=True)
if prev and not os.path.exists(f'{RUN}/last.pt'):
    best = max(prev, key=lambda p: torch.load(p, map_location='cpu', weights_only=False)['epoch'])
    shutil.copytree(os.path.dirname(best), RUN, dirs_exist_ok=True)
    print('resuming from', best)
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    ("code", '''
!python -m figo.train --model {MODEL} --cache-dir {CACHE} --splits {SPLITS} --out-dir {RUN} \\
    --epochs {EPOCHS} --max-hours {MAX_HOURS} --patch-size {PATCH} --batch-size {BATCH} --workers 3
'''),
    ("code", TIMING),
    ("markdown", "## Test-set inference and evaluation (in the run where training finishes)"),
    ("code", '''
import json
done = torch.load(f'{RUN}/last.pt', map_location='cpu', weights_only=False)['epoch'] + 1 >= EPOCHS
if done and not SMOKE:
    PRED, RES = f'{OUT}/preds/{MODEL}', f'{OUT}/results'
    !python -m figo.infer --checkpoint {RUN}/best.pt --data-dir {CLEAN} --splits {SPLITS} --subset test --out-dir {PRED}
    !python -m figo.evaluate --pred-dir {PRED} --data-dir {CLEAN} --splits {SPLITS} --subset test --out {RES}/{MODEL}_seg.csv
    !python -m figo.figo_eval agreement --ref-dir {CLEAN} --pred-dir {PRED} --splits {SPLITS} --subset test \\
        --out {RES}/{MODEL}_figo_agreement.csv
    print("Download this version's output and copy preds/ and results/ into MyDrive/figo/ for model selection.")
else:
    print('Training not finished yet - attach this output as input and Save & Run All again.')
'''),
])

for name, book in [("00_setup_data.ipynb", setup), ("01_train_custom_model.ipynb", train),
                   ("02_nnunet.ipynb", nnunet), ("03_select_and_figo.ipynb", figo),
                   ("K1_kaggle_train.ipynb", kaggle)]:
    (OUT / name).write_text(json.dumps(book, indent=1) + "\n")
    print("wrote", OUT / name)
