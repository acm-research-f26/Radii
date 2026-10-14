"""Patch sampling and augmentation from the network-space ``.npz`` cache."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .labels import CAVITY, CYST, MYOMA, WALL
from .preprocess import Geometry

FG_CLASSES = (WALL, CAVITY, MYOMA, CYST)
_MAX_CENTRES = 5000


def load_cached(path: str | Path):
    with np.load(path) as z:
        geom = Geometry.from_dict(json.loads(str(z["geometry"])))
        return z["image"].astype(np.float32), z["seg"].astype(np.uint8), geom


class PatchDataset(Dataset):
    """Random 3D patches. With probability ``fg_prob`` a patch is centred on a
    voxel of a foreground class chosen uniformly among the classes present, so
    small structures (cavity, cysts) are seen as often as the wall."""

    def __init__(self, files, patch_size, n_samples, fg_prob=0.5, augment=True, seed=0):
        self.patch = tuple(patch_size)
        self.n_samples = n_samples
        self.fg_prob = fg_prob
        self.augment = augment
        self.seed = seed
        self.cases = []
        rng = np.random.default_rng(seed)
        for f in files:
            img, seg, _ = load_cached(f)
            centres = {}
            for c in FG_CLASSES:
                idx = np.argwhere(seg == c)
                if len(idx):
                    centres[c] = idx[rng.choice(len(idx), min(len(idx), _MAX_CENTRES), replace=False)]
            self.cases.append((img.astype(np.float16), seg, centres))

    def __len__(self):
        return self.n_samples

    def _crop(self, img, seg, centre):
        pads = [(0, max(0, p - s)) for p, s in zip(self.patch, img.shape)]
        if any(p[1] for p in pads):
            img = np.pad(img, pads, constant_values=img.min())
            seg = np.pad(seg, pads)
        starts = []
        for c, p, s in zip(centre, self.patch, img.shape):
            starts.append(int(np.clip(c - p // 2, 0, s - p)))
        sl = tuple(slice(st, st + p) for st, p in zip(starts, self.patch))
        return img[sl], seg[sl]

    def __getitem__(self, i):
        # torch's generator is reseeded per worker and per epoch by the DataLoader
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,))))
        img, seg, centres = self.cases[rng.integers(len(self.cases))]
        if centres and rng.random() < self.fg_prob:
            pts = centres[list(centres)[rng.integers(len(centres))]]
            centre = pts[rng.integers(len(pts))]
        else:
            centre = [rng.integers(s) for s in img.shape]
        x, y = self._crop(img.astype(np.float32), seg, centre)
        x = torch.from_numpy(np.ascontiguousarray(x))[None]
        y = torch.from_numpy(np.ascontiguousarray(y)).long()[None]
        if self.augment:
            x, y = augment(x, y, rng)
        return x, y


def _inplane_affine(x, y, rng, max_rot_deg=15.0, scale=(0.85, 1.15)):
    """Same random 2D rotation+scale applied to every slice: in-plane only,
    which respects the thick slice axis."""
    a = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    s = rng.uniform(*scale)
    theta = torch.tensor([[math.cos(a) / s, -math.sin(a) / s, 0.0],
                          [math.sin(a) / s, math.cos(a) / s, 0.0]], dtype=torch.float32)
    # (1, H, W, D) -> (D, 1, H, W): slices become the batch
    xs = x.permute(3, 0, 1, 2)
    ys = y.permute(3, 0, 1, 2).float()
    grid = F.affine_grid(theta.expand(xs.shape[0], 2, 3), list(xs.shape), align_corners=False)
    fill = xs.amin()
    xs = F.grid_sample(xs - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill
    ys = F.grid_sample(ys, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return xs.permute(1, 2, 3, 0), ys.long().permute(1, 2, 3, 0)


def augment(x, y, rng):
    if rng.random() < 0.3:
        x, y = _inplane_affine(x, y, rng)
    for ax in (1, 2, 3):
        if rng.random() < 0.5:
            x, y = x.flip(ax), y.flip(ax)
    if rng.random() < 0.15:
        x = x + torch.randn_like(x) * rng.uniform(0, 0.1)
    if rng.random() < 0.15:
        x = x * rng.uniform(0.75, 1.25)
    if rng.random() < 0.15:
        x = x + rng.uniform(-0.2, 0.2)
    if rng.random() < 0.3:  # gamma on the [0, 1]-rescaled patch
        lo, hi = x.min(), x.max()
        rngx = (hi - lo).clamp_min(1e-6)
        x = ((x - lo) / rngx) ** rng.uniform(0.7, 1.5) * rngx + lo
    return x, y
