"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-3-12

Conditioinal Residual Dense Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from VSR.Util.Utility import to_list

from .Ops.Blocks import CascadeRdn
from .Optim.SISR import L1Optimizer


class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        in_c, out_c = to_list(channels, 2)
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.c2 = nn.Conv2d(in_c, out_c, 3, 1, 1)

    def forward(self, inputs, skips, scale=2):
        up = F.interpolate(inputs, scale_factor=scale)
        up = self.c1(up)
        con = torch.cat([up, skips], dim=1)
        return self.c2(con)


class Crdn(nn.Module):
    def __init__(self, blocks=(4, 4), **kwargs):
        super(Crdn, self).__init__()
        self.blocks = to_list(blocks, 2)

        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.Conv2d(32, 32, 5, 1, 2))
        self.exit = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 3, 3, 1, 1))
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.down2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.up1 = Upsample([128, 64])
        self.up2 = Upsample([64, 32])
        self.cb1 = CascadeRdn(32, 32, 3, True)
        self.cb2 = CascadeRdn(64, 64, 3, True)
        self.cb3 = CascadeRdn(128, 128, 3, True)
        self.cb4 = CascadeRdn(128, 128, 3, True)
        self.cb5 = CascadeRdn(64, 64, 3, True)
        self.cb6 = CascadeRdn(32, 32, 3, True)

    def forward(self, inputs):
        entry = self.entry(inputs)
        x1 = self.cb1(entry)
        x = self.down1(x1)
        x2 = self.cb2(x)
        x = self.down2(x2)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.up1(x, x2)
        x = self.cb5(x)
        x = self.up2(x, x1)
        x = self.cb6(x)
        x += entry
        out = self.exit(x)
        return out


class CRDN(L1Optimizer):
    def __init__(self, channel=3, scale=1, **kwargs):
        self.rsr = Crdn()
        super(CRDN, self).__init__(scale=scale, channel=channel, **kwargs)

    def fn(self, x):
        return self.rsr(x)
