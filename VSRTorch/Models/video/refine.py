#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
from torch import nn
from torch.nn import functional as F

from ..Arch import Rdb, CascadeRdn, Upsample, SpaceToDepth


class SkipUp(nn.Module):
  def __init__(self, channels):
    super(SkipUp, self).__init__()
    in_c, out_c = channels
    self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
    self.c2 = nn.Conv2d(in_c, out_c, 3, 1, 1)

  def forward(self, inputs, skips, scale=2):
    up = F.interpolate(inputs, scale_factor=scale)
    up = self.c1(up)
    con = torch.cat([up, skips], dim=1)
    return self.c2(con)


class Unet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Unet, self).__init__()
    self.entry = nn.Sequential(
      nn.Conv2d(in_channels, 32, 7, 1, 3),
      nn.Conv2d(32, 32, 5, 1, 2))
    self.exit = nn.Sequential(
      nn.Conv2d(32, 32, 3, 1, 1),
      nn.Conv2d(32, out_channels, 3, 1, 1))
    self.down1 = nn.Conv2d(32, 64, 3, 2, 1)
    self.down2 = nn.Conv2d(64, 128, 3, 2, 1)
    self.up1 = SkipUp([128, 64])
    self.up2 = SkipUp([64, 32])
    self.cb1 = CascadeRdn(32, 3, True)
    self.cb2 = CascadeRdn(64, 3, True)
    self.cb3 = CascadeRdn(128, 3, True)
    self.cb4 = CascadeRdn(128, 3, True)
    self.cb5 = CascadeRdn(64, 3, True)
    self.cb6 = CascadeRdn(32, 3, True)

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


class SNet(nn.Module):
  def __init__(self, channel, N=2):
    super(SNet, self).__init__()
    self.body = nn.Sequential(
      nn.Conv2d(channel * N, 16, 3, 1, 1),
      SpaceToDepth(4),
      nn.Conv2d(256, 64, 1, 1, 0),
      CascadeRdn(64, 3),
      Upsample(64, 4), nn.Conv2d(64, channel, 3, 1, 1))

  def forward(self, *inputs):
    x = torch.cat(inputs, dim=1)
    return self.body(x)
