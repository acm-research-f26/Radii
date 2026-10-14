"""Shared building blocks and the anisotropic stage layout.

All custom models take input ``(B, C, H, W, D)`` where D is the thick
(~6.6 mm) slice axis. Like nnU-Net's planning for anisotropic data, the early
stages only downsample and convolve in-plane; through-plane downsampling starts
once in-plane resolution is coarse enough to be comparable to slice spacing.
"""

from __future__ import annotations

import torch
from torch import nn

# per-stage (kernel, stride); stage 0 is full resolution
DEFAULT_KERNELS = [(3, 3, 1), (3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
DEFAULT_STRIDES = [(1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2)]


def total_stride(strides) -> tuple[int, int, int]:
    out = [1, 1, 1]
    for s in strides:
        out = [o * x for o, x in zip(out, s)]
    return tuple(out)


def pad_for(kernel, dilation=1):
    if isinstance(dilation, int):
        dilation = (dilation,) * 3
    return tuple(d * (k // 2) for k, d in zip(kernel, dilation))


def out_pad(stride):
    return tuple(s - 1 for s in stride)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=(3, 3, 3), stride=(1, 1, 1), dilation=1):
        super().__init__(
            nn.Conv3d(cin, cout, kernel, stride, pad_for(kernel, dilation), dilation=dilation, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class SegHeads(nn.Module):
    """1x1x1 output heads for deep supervision. ``feats`` is ordered
    full-resolution first. In eval mode (or without deep supervision) only the
    full-resolution logits are returned; in training a list is returned."""

    def __init__(self, channels, n_classes, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        n = len(channels) if deep_supervision else 1
        self.heads = nn.ModuleList(nn.Conv3d(c, n_classes, 1) for c in channels[:n])

    def forward(self, feats):
        if self.training and self.deep_supervision:
            return [h(f) for h, f in zip(self.heads, feats)]
        return self.heads[0](feats[0])


def check_input(x: torch.Tensor, strides) -> None:
    ts = total_stride(strides)
    if any(s % t for s, t in zip(x.shape[2:], ts)):
        raise ValueError(f"spatial shape {tuple(x.shape[2:])} must be divisible by {ts}")
