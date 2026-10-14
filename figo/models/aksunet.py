"""AKS-UNet: U-Net with adaptive kernel selection and spatial attention.

Re-implementation from the published description: each encoder stage uses a
selective-kernel convolution (two branches with different receptive fields -
a 3x3x3 conv and a dilated 3x3x3 conv - fused by channel-wise soft attention
over branches, as in SKNet), and each decoder stage applies a CBAM-style
spatial attention map before the output. Dilation is in-plane only on the
anisotropic stages. Not the authors' code - see README for provenance.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import DEFAULT_KERNELS, DEFAULT_STRIDES, ConvNormAct, SegHeads, check_input


class SelectiveKernelConv(nn.Module):
    def __init__(self, cin, cout, kernel, reduction=8):
        super().__init__()
        dil = (2, 2, 2 if kernel[2] > 1 else 1)
        self.branches = nn.ModuleList([ConvNormAct(cin, cout, kernel), ConvNormAct(cin, cout, kernel, dilation=dil)])
        hidden = max(cout // reduction, 16)
        self.squeeze = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(cout, hidden, 1), nn.ReLU(inplace=True))
        self.select = nn.Conv3d(hidden, cout * len(self.branches), 1)

    def forward(self, x):
        feats = torch.stack([b(x) for b in self.branches], 1)  # B, K, C, H, W, D
        z = self.squeeze(feats.sum(1))
        b, k, c = feats.shape[:3]
        attn = self.select(z).view(b, k, c, 1, 1, 1).softmax(1)
        return (feats * attn).sum(1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel=(7, 7, 3)):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel, padding=tuple(k // 2 for k in kernel))

    def forward(self, x):
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(pooled))


class AKSUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=5, features=(32, 64, 128, 256, 320),
                 kernels=DEFAULT_KERNELS, strides=DEFAULT_STRIDES, deep_supervision=True):
        super().__init__()
        self.strides = strides
        self.enc = nn.ModuleList()
        cin = in_channels
        for f, k, s in zip(features, kernels, strides):
            self.enc.append(nn.Sequential(ConvNormAct(cin, f, k, s), SelectiveKernelConv(f, f, k)))
            cin = f
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            f, k = features[i - 1], kernels[i - 1]
            self.up.append(nn.ConvTranspose3d(features[i], f, strides[i], strides[i]))
            self.dec.append(nn.Sequential(ConvNormAct(2 * f, f, k), ConvNormAct(f, f, k),
                                          SpatialAttention((7, 7, 3 if k[2] > 1 else 1))))
        self.heads = SegHeads(list(features[:-1]), n_classes, deep_supervision)

    def forward(self, x):
        check_input(x, self.strides)
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            outs.append(x)
        return self.heads(outs[::-1])
