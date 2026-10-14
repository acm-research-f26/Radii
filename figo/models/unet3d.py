"""Standard 3D U-Net baseline (Cicek et al. 2016 layout): two conv blocks per
stage, max-pool down, transposed-conv up, skip concatenation. Instance norm
replaces batch norm because the batch size on a T4 is 2. No deep supervision
by default, to stay a plain baseline."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import DEFAULT_KERNELS, DEFAULT_STRIDES, ConvNormAct, SegHeads, check_input


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, n_classes=5, features=(32, 64, 128, 256, 320),
                 kernels=DEFAULT_KERNELS, strides=DEFAULT_STRIDES, deep_supervision=False):
        super().__init__()
        self.strides = strides
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        cin = in_channels
        for f, k, s in zip(features, kernels, strides):
            self.pool.append(nn.MaxPool3d(s, s) if max(s) > 1 else nn.Identity())
            self.enc.append(nn.Sequential(ConvNormAct(cin, f, k), ConvNormAct(f, f, k)))
            cin = f
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            s = strides[i]
            self.up.append(nn.ConvTranspose3d(features[i], features[i - 1], s, s))
            k = kernels[i - 1]
            self.dec.append(nn.Sequential(ConvNormAct(2 * features[i - 1], features[i - 1], k),
                                          ConvNormAct(features[i - 1], features[i - 1], k)))
        self.heads = SegHeads(list(features[:-1]), n_classes, deep_supervision)

    def forward(self, x):
        check_input(x, self.strides)
        skips = []
        for pool, enc in zip(self.pool, self.enc):
            x = enc(pool(x))
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            outs.append(x)
        return self.heads(outs[::-1])
