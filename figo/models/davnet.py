"""DA-VNet: V-Net with attention-gated skips and deep supervision.

Re-implementation from the published description: V-Net residual stages
(Milletari et al. 2016; 1/2/3/3/3 convolutions per stage, PReLU, strided-conv
downsampling), additive attention gates on every skip connection (Oktay et
al. 2018), and auxiliary outputs at each decoder stage. The 5x5x5 V-Net kernel
is used as 5x5 in-plane; through-plane it follows the shared anisotropic
layout. Not the authors' code - see README for provenance.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import DEFAULT_KERNELS, DEFAULT_STRIDES, SegHeads, check_input, pad_for


def _vnet_kernel(k):
    return (5, 5, k[2])


class ResStage(nn.Module):
    def __init__(self, ch, n_convs, kernel):
        super().__init__()
        layers = []
        for _ in range(n_convs):
            layers += [nn.Conv3d(ch, ch, kernel, padding=pad_for(kernel), bias=False),
                       nn.InstanceNorm3d(ch, affine=True), nn.PReLU(ch)]
        self.body = nn.Sequential(*layers[:-1])
        self.act = nn.PReLU(ch)

    def forward(self, x):
        return self.act(self.body(x) + x)


class AttentionGate(nn.Module):
    def __init__(self, ch_x, ch_g, ch_inter):
        super().__init__()
        self.wx = nn.Conv3d(ch_x, ch_inter, 1, bias=False)
        self.wg = nn.Conv3d(ch_g, ch_inter, 1)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv3d(ch_inter, 1, 1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.psi(self.wx(x) + self.wg(g))


class DAVNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=5, base=16, n_convs=(1, 2, 3, 3, 3),
                 kernels=DEFAULT_KERNELS, strides=DEFAULT_STRIDES, deep_supervision=True):
        super().__init__()
        self.strides = strides
        n = len(strides)
        ch = [base * 2**i for i in range(n)]
        ks = [_vnet_kernel(k) for k in kernels]
        self.inp = nn.Sequential(nn.Conv3d(in_channels, base, ks[0], padding=pad_for(ks[0])),
                                 nn.InstanceNorm3d(base, affine=True), nn.PReLU(base))
        self.enc = nn.ModuleList(ResStage(ch[i], n_convs[i], ks[i]) for i in range(n))
        self.down = nn.ModuleList(
            nn.Sequential(nn.Conv3d(ch[i - 1], ch[i], strides[i], strides[i], bias=False),
                          nn.InstanceNorm3d(ch[i], affine=True), nn.PReLU(ch[i]))
            for i in range(1, n))
        self.up, self.gate, self.fuse, self.dec = (nn.ModuleList() for _ in range(4))
        for i in range(n - 1, 0, -1):
            c = ch[i - 1]
            self.up.append(nn.Sequential(nn.ConvTranspose3d(ch[i], c, strides[i], strides[i], bias=False),
                                         nn.InstanceNorm3d(c, affine=True), nn.PReLU(c)))
            self.gate.append(AttentionGate(c, c, max(c // 2, 8)))
            self.fuse.append(nn.Conv3d(2 * c, c, 1))
            self.dec.append(ResStage(c, n_convs[i - 1], ks[i - 1]))
        self.heads = SegHeads(ch[:-1], n_classes, deep_supervision)

    def forward(self, x):
        check_input(x, self.strides)
        x = self.enc[0](self.inp(x))
        skips = [x]
        for down, enc in zip(self.down, self.enc[1:]):
            x = enc(down(x))
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, gate, fuse, dec in zip(self.up, self.gate, self.fuse, self.dec):
            g = up(x)
            s = gate(skips.pop(), g)
            x = dec(fuse(torch.cat([g, s], 1)))
            outs.append(x)
        return self.heads(outs[::-1])
