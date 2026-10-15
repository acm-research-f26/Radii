"""Train a custom Stage 1 baseline with Colab-proof checkpoint/resume.

    python -m figo.train --model mednext --cache-dir data/cache_0.9mm \
        --splits data/splits.json --out-dir /content/drive/MyDrive/figo/runs/mednext

Every epoch ``last.pt`` is written atomically to ``--out-dir`` (put it on
Google Drive). Re-running the same command resumes from it, so a Colab
disconnect costs at most one epoch. ``--max-hours`` stops cleanly before the
session/compute budget runs out. ``best.pt`` tracks the best validation mean
foreground Dice; ``log.csv`` has one row per epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PatchDataset, load_cached
from .infer import predict_volume
from .labels import LABEL_NAMES, NUM_CLASSES
from .losses import DeepSupervisionLoss, DiceCE
from .models import DEEP_SUPERVISION, build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(DEEP_SUPERVISION))
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--patch-size", type=int, nargs=3, default=[192, 192, 16], help="H W D (D = slice axis)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200, help="capped for benchmarking, not SOTA")
    p.add_argument("--iters-per-epoch", type=int, default=250)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=None, help="default 1e-2 (sgd) / 1e-3 (adamw)")
    p.add_argument("--weight-decay", type=float, default=3e-5)
    p.add_argument("--fg-prob", type=float, default=0.5)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this wall time")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-deep-supervision", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def dice_per_class(pred: np.ndarray, gt: np.ndarray) -> dict:
    out = {}
    for c, name in LABEL_NAMES.items():
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        out[name] = float(2 * (p & g).sum() / denom) if denom else float("nan")
    return out


def validate(model, val_files, cfg, device):
    rows = []
    for f in val_files:
        img, seg, _ = load_cached(f)
        probs = predict_volume(model, img, cfg["patch_size"], device, overlap=0.25, amp=cfg["amp"])
        rows.append(dice_per_class(probs.argmax(0), seg))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in LABEL_NAMES.values()}


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU found - training will be extremely slow")

    lr = args.lr or (1e-2 if args.optimizer == "sgd" else 1e-3)
    cfg = {
        "model": args.model,
        "deep_supervision": DEEP_SUPERVISION[args.model] and not args.no_deep_supervision,
        "patch_size": args.patch_size,
        "amp": not args.no_amp,
        "lr": lr,
        **{k: getattr(args, k) for k in ("optimizer", "epochs", "iters_per_epoch", "batch_size", "fg_prob", "seed")},
    }

    cache = Path(args.cache_dir)
    splits = json.loads(Path(args.splits).read_text())
    train_files = [cache / f"{c}.npz" for c in splits["train"]]
    val_files = [cache / f"{c}.npz" for c in splits["val"]]
    _, _, geom = load_cached(train_files[0])
    cfg["inplane_mm"] = round(geom.net_spacing[0], 3)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.model, n_classes=NUM_CLASSES, deep_supervision=cfg["deep_supervision"]).to(device)
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda")
    loss_fn = DeepSupervisionLoss(DiceCE(NUM_CLASSES))

    start_epoch, best, ema = 0, -1.0, None
    last = out / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["config"]["model"] != args.model:
            raise SystemExit(f"{last} belongs to model {ckpt['config']['model']!r}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best, ema = ckpt["epoch"] + 1, ckpt["best"], ckpt.get("ema")
        torch.set_rng_state(ckpt["torch_rng"])
        print(f"Resumed from {last} at epoch {start_epoch} (best val Dice {best:.4f})")
    else:
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
    if start_epoch >= args.epochs:
        print("Training already finished.")
        return

    print(f"Loading {len(train_files)} training volumes ...")
    ds = PatchDataset(train_files, args.patch_size, args.iters_per_epoch * args.batch_size,
                      fg_prob=args.fg_prob, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model}: {n_params:.1f}M parameters, deep supervision={cfg['deep_supervision']}")

    log_path = out / "log.csv"
    fields = ["epoch", "lr", "train_loss", "epoch_sec"] + [f"val_{n}" for n in LABEL_NAMES.values()] + ["val_mean", "val_ema"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    t_start, epoch_times = time.time(), []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        cur_lr = lr * (1 - epoch / args.epochs) ** 0.9  # poly schedule, stateless -> resume-safe
        for g in opt.param_groups:
            g["lr"] = cur_lr
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
                outputs = model(x)
            loss = loss_fn([o.float() for o in outputs] if isinstance(outputs, list) else outputs.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        row = {"epoch": epoch, "lr": f"{cur_lr:.2e}", "train_loss": f"{np.mean(losses):.4f}"}
        is_best = False
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            val = validate(model, val_files, cfg, device)
            mean = float(np.nanmean(list(val.values())))
            ema = mean if ema is None else 0.8 * ema + 0.2 * mean  # smooth noisy val Dice for model selection
            row.update({f"val_{k}": f"{v:.4f}" for k, v in val.items()})
            row.update(val_mean=f"{mean:.4f}", val_ema=f"{ema:.4f}")
            if ema > best:
                best, is_best = ema, True
        row["epoch_sec"] = f"{time.time() - t0:.0f}"
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()))

        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best": best, "ema": ema, "config": cfg, "torch_rng": torch.get_rng_state()}
        atomic_save(state, last)
        if is_best:
            atomic_save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_ema": best}, out / "best.pt")

        epoch_times.append(time.time() - t0)
        if args.max_hours is not None:
            elapsed = time.time() - t_start
            if elapsed + 1.5 * max(epoch_times[-3:]) > args.max_hours * 3600:
                print(f"Stopping after epoch {epoch}: --max-hours budget reached. Re-run to resume.")
                return
    print(f"Done. Best smoothed val Dice {best:.4f}")


if __name__ == "__main__":
    main()
