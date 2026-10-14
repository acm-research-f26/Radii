"""Dice + cross-entropy loss with optional deep supervision (nnU-Net recipe)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceCE(nn.Module):
    """Batch soft Dice over foreground classes + cross-entropy."""

    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, target):
        target = target[:, 0] if target.ndim == logits.ndim else target
        ce = F.cross_entropy(logits, target)
        probs = logits.float().softmax(1)[:, 1:]
        onehot = F.one_hot(target, self.n_classes).movedim(-1, 1)[:, 1:].float()
        dims = (0, 2, 3, 4)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return ce + (1 - dice.mean())


class DeepSupervisionLoss(nn.Module):
    """Weights 1, 1/2, 1/4, ... (normalised) over outputs ordered full-res first;
    targets are nearest-neighbour downsampled to each output's size."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.base(outputs, target)
        w = torch.tensor([0.5**i for i in range(len(outputs))])
        w = w / w.sum()
        total = 0.0
        for wi, out in zip(w, outputs):
            t = target
            if out.shape[2:] != target.shape[2:]:
                t = F.interpolate(target.float(), size=out.shape[2:], mode="nearest").long()
            total = total + wi * self.base(out, t)
        return total
