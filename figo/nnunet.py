"""nnU-Net v2 helpers.

    # convert the cleaned UMD data, using the SAME split as the custom models
    python -m figo.nnunet convert --data-dir data/clean --splits data/splits.json \
        --nnunet-raw $nnUNet_raw --nnunet-preprocessed $nnUNet_preprocessed

    # make -tr nnUNetTrainer_UMD available (capped epochs, frequent checkpoints)
    python -m figo.nnunet install-trainer

Train/val cases go to imagesTr (fold 0 of ``splits_final.json`` reproduces our
train/val split); test cases go to imagesTs/labelsTs and are never seen by
nnU-Net during training or model selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_UMD"
LABELS = {"background": 0, "uterine_wall": 1, "uterine_cavity": 2, "myoma": 3, "nabothian_cyst": 4}


def convert(data_dir, splits_path, raw_root, preprocessed_root):
    data = Path(data_dir)
    splits = json.loads(Path(splits_path).read_text())
    ds = Path(raw_root) / DATASET_NAME
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / sub).mkdir(parents=True, exist_ok=True)

    def put(cid, img_sub, lab_sub):
        # figo.prepare output is correctly gzipped, so nnU-Net's reader can open it
        shutil.copyfile(data / f"{cid}_t2.nii.gz", ds / img_sub / f"{cid}_0000.nii.gz")
        shutil.copyfile(data / f"{cid}_seg.nii.gz", ds / lab_sub / f"{cid}.nii.gz")

    trainval = splits["train"] + splits["val"]
    for cid in trainval:
        put(cid, "imagesTr", "labelsTr")
    for cid in splits["test"]:
        put(cid, "imagesTs", "labelsTs")

    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "T2"},  # non-CT -> per-case z-score normalisation
        "labels": LABELS,
        "numTraining": len(trainval),
        "file_ending": ".nii.gz",
        "name": "UMD",
        "description": "Uterine Myoma MRI Dataset, sagittal T2w",
    }, indent=1))

    pre = Path(preprocessed_root) / DATASET_NAME
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "splits_final.json").write_text(json.dumps([{"train": splits["train"], "val": splits["val"]}], indent=1))
    print(f"Wrote {ds} ({len(trainval)} train+val, {len(splits['test'])} test) and {pre / 'splits_final.json'}")


def install_trainer():
    import nnunetv2

    dst = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "nnUNetTrainer_UMD.py"
    shutil.copyfile(Path(__file__).with_name("nnunet_trainer_umd.py"), dst)
    print(f"Installed nnUNetTrainer_UMD -> {dst}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--splits", required=True)
    c.add_argument("--nnunet-raw", required=True)
    c.add_argument("--nnunet-preprocessed", required=True)
    sub.add_parser("install-trainer")
    args = p.parse_args(argv)
    if args.cmd == "convert":
        convert(args.data_dir, args.splits, args.nnunet_raw, args.nnunet_preprocessed)
    else:
        install_trainer()


if __name__ == "__main__":
    main()
