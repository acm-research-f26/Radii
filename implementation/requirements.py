#!/usr/bin/env python3
"""
install_requirements.py - install this project's Python dependencies.

Usage (run it with the same Python you use to run the project):
    python install_requirements.py             # preprocessing only (numpy, nibabel)
    python install_requirements.py --full      # everything, including nnU-Net + PyTorch
    python install_requirements.py --list      # just print what would be installed

Notes:
  * Do NOT run --full on Google Colab or Kaggle. Those environments already ship a
    GPU build of PyTorch, and reinstalling torch can replace it with a CPU-only build.
    There, install only: nnunetv2 nibabel fire SimpleITK
  * For GPU training locally, install the CUDA build of PyTorch from pytorch.org first,
    then run this script.
"""
import argparse
import subprocess
import sys

# Needed for umd_preprocess.py

PREPROCESSING = [
    "numpy",
    "nibabel",
]

# Needed for uterine_mri_tool.py (segmentation, reports, evaluation, CLI)

PIPELINE = [
    "scipy",
    "matplotlib",
    "fire",
    "SimpleITK",
    "nnunetv2",
    "torch",
]

def install(packages):
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser(description="Install project dependencies.")
    ap.add_argument("--full", action="store_true",
                    help="also install the segmentation pipeline packages (nnU-Net, PyTorch, ...)")
    ap.add_argument("--list", action="store_true",
                    help="print the packages and exit without installing")
    args = ap.parse_args()

    packages = PREPROCESSING + (PIPELINE if args.full else [])
    print("Python:", sys.executable)
    print("Packages:", ", ".join(packages))
    if args.list:
        return 0
    return install(packages)


if __name__ == "__main__":
    sys.exit(main())