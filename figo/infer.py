"""Sliding-window inference for the custom models; writes native-resolution
label maps (canonical orientation) named ``<case_id>.nii.gz``, the same naming
nnU-Net uses, so ``figo.evaluate`` scores every model the same way.

    python -m figo.infer --checkpoint runs/mednext/best.pt --data-dir data/clean \
        --splits data/splits.json --subset test --out-dir preds/mednext
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .io import load_nifti
from .models import build_model
from .preprocess import to_native_labels, to_network_space


@torch.no_grad()
def predict_volume(model, image: np.ndarray, patch, device, overlap=0.5, tta=False, amp=True) -> np.ndarray:
    """``image`` (H, W, D) network-space array -> softmax probabilities (C, H, W, D)."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(device)
    use_amp = amp and device.type == "cuda"

    def run(inp):
        with torch.autocast(device.type, enabled=use_amp):
            logits = sliding_window_inference(inp, tuple(patch), 2, model, overlap=overlap, mode="gaussian")
        return logits.float().softmax(1)

    probs = run(x)
    if tta:
        flips = [(2,), (3,), (2, 3)]
        for f in flips:
            probs += run(x.flip(f)).flip(f)
        probs /= 1 + len(flips)
    return probs[0].cpu().numpy()


def load_checkpoint_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], deep_supervision=cfg["deep_supervision"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True, help="figo.prepare clean/ folder")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits")
    p.add_argument("--subset", default="test")
    p.add_argument("--cases", nargs="*", help="explicit case ids (overrides --splits)")
    p.add_argument("--tta", action="store_true", help="average over in-plane flips (4x slower)")
    p.add_argument("--overlap", type=float, default=0.5)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint_model(args.checkpoint, device)
    cases = args.cases or json.loads(Path(args.splits).read_text())[args.subset]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    for i, cid in enumerate(cases, 1):
        dst = out / f"{cid}.nii.gz"
        if dst.exists():
            continue
        img, _, geom = to_network_space(load_nifti(data / f"{cid}_t2.nii.gz"), None, cfg["inplane_mm"])
        probs = predict_volume(model, img, cfg["patch_size"], device, args.overlap, args.tta)
        nib.save(to_native_labels(probs, geom), dst)
        print(f"[{i}/{len(cases)}] {cid}")


if __name__ == "__main__":
    main()
