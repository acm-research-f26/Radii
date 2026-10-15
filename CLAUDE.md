# Team rules

Automated FIGO classification of uterine fibroids from MRI (UMD dataset). Training runs on Google Colab
and Kaggle; data, checkpoints and predictions live on Google Drive under `MyDrive/figo/`.

## Rules
- **Never commit data, checkpoints or predictions** (`*.nii*`, `*.npz`, `*.pt`, `*.pth`, `data/`, `runs/`,
  `preds/`, `results/`). `.gitignore` already blocks them; don't force-add around it.
- **`data/splits.json` lives on Drive and is never regenerated.** Every model trains and is evaluated on that one
  split. Don't delete it, edit it, or re-run `figo.prepare` in a way that re-splits.
- **Edit notebooks only through `tools/make_notebooks.py`.** Change the generator, then run
  `python tools/make_notebooks.py notebooks` and commit both. Never hand-edit or commit notebooks saved from Colab.
- **Run `pytest` before pushing.** All tests must pass on CPU. CI runs the same tests on every push and PR.
- **One branch per person** (`ira`, `purnima`, `anagha`, `snigdha`), merged into `main` through pull requests.

## Local setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if torch is missing
pip install -r requirements-dev.txt
pytest
```
Needs Python 3.10 or newer.
