"""MedNeXt (Roy et al., MICCAI 2023): a fully ConvNeXt-style 3D U-Net.

Re-implementation of the MedNeXt-S configuration (expansion ratio 2, two blocks
per stage, kernel 3) with residual ("MedNeXt-style") up/down-sampling blocks and
deep supervision. Adapted to UMD anisotropy via the shared stage layout; the
original is isotropic. Not the authors' code - see README for provenance.
"""

from __future__ import annotations

from torch import nn

from .blocks import DEFAULT_KERNELS, DEFAULT_STRIDES, SegHeads, check_input, out_pad, pad_for


class MedNeXtBlock(nn.Module):
    """depthwise conv -> GroupNorm -> 1x1 expand -> GELU -> 1x1 compress (+ residual)."""

    def __init__(self, cin, cout=None, kernel=(3, 3, 3), expansion=2, stride=(1, 1, 1), mode="same"):
        super().__init__()
        cout = cout or cin
        if mode == "up":
            self.dw = nn.ConvTranspose3d(cin, cin, kernel, stride, pad_for(kernel), out_pad(stride), groups=cin)
            self.res = nn.ConvTranspose3d(cin, cout, 1, stride, 0, out_pad(stride))
        else:
            self.dw = nn.Conv3d(cin, cin, kernel, stride, pad_for(kernel), groups=cin)
            self.res = None if (cin == cout and stride == (1, 1, 1)) else nn.Conv3d(cin, cout, 1, stride)
        self.norm = nn.GroupNorm(cin, cin)
        self.expand = nn.Conv3d(cin, expansion * cin, 1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(expansion * cin, cout, 1)

    def forward(self, x):
        y = self.compress(self.act(self.expand(self.norm(self.dw(x)))))
        return y + (x if self.res is None else self.res(x))


class MedNeXt(nn.Module):
    def __init__(self, in_channels=1, n_classes=5, base=32, blocks=2, expansion=2,
                 kernels=DEFAULT_KERNELS, strides=DEFAULT_STRIDES, deep_supervision=True):
        super().__init__()
        self.strides = strides
        n = len(strides)
        ch = [base * 2**i for i in range(n)]
        self.stem = nn.Conv3d(in_channels, base, 1)
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        for i in range(n):
            if i > 0:
                self.down.append(MedNeXtBlock(ch[i - 1], ch[i], kernels[i], expansion, strides[i], "down"))
            self.enc.append(nn.Sequential(*[MedNeXtBlock(ch[i], kernel=kernels[i], expansion=expansion)
                                            for _ in range(blocks)]))
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(n - 1, 0, -1):
            self.up.append(MedNeXtBlock(ch[i], ch[i - 1], kernels[i], expansion, strides[i], "up"))
            self.dec.append(nn.Sequential(*[MedNeXtBlock(ch[i - 1], kernel=kernels[i - 1], expansion=expansion)
                                            for _ in range(blocks)]))
        self.heads = SegHeads(ch[:-1], n_classes, deep_supervision)

    def forward(self, x):
        check_input(x, self.strides)
        x = self.enc[0](self.stem(x))
        skips = [x]
        for down, enc in zip(self.down, self.enc[1:]):
            x = enc(down(x))
            skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec in zip(self.up, self.dec):
            x = dec(up(x) + skips.pop())  # MedNeXt sums skips rather than concatenating
            outs.append(x)
        return self.heads(outs[::-1])
