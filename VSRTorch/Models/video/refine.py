#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
from torch import nn

from ..Arch import Rdb, CascadeRdn, Upsample, SpaceToDepth
from ..Rsr import Upsample as RsrUp


class Unet(nn.Module):
  def __init__(self, channel, N=2):
    super(Unet, self).__init__()
    self.entry = nn.Sequential(
      nn.Conv2d(channel * N, 32, 3, 1, 1),
      SpaceToDepth(2),
      nn.Conv2d(128, 32, 1, 1, 0))
    self.exit = nn.Sequential(
      Upsample(32, 2), nn.Conv2d(32, channel, 3, 1, 1))
    self.down1 = nn.Conv2d(32, 64, 3, 2, 1)
    self.up1 = RsrUp([64, 32])
    self.cb = CascadeRdn(64, 3, True)

  def forward(self, *inputs):
    inp = torch.cat(inputs, dim=1)  # w
    c0 = self.entry(inp)  # w / 2
    c1 = self.down1(c0)  # w / 4
    x = self.cb(c1)  # w / 4
    c2 = self.up1(x, c0)
    out = self.exit(c2)
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


class SNet2(nn.Module):
  def __init__(self, channel):
    super(SNet2, self).__init__()
    self.body = nn.Sequential(
      nn.Conv2d(channel, 16, 3, 1, 1),
      SpaceToDepth(4),
      nn.Conv2d(256, 64, 1, 1, 0),
      CascadeRdn(64, 3),
      Upsample(64, 4), nn.Conv2d(64, channel, 3, 1, 1))
    self.mask = nn.Sequential(
      nn.Conv2d(channel * 3, 4, 3, 1, 1),
      SpaceToDepth(4),
      Rdb(64), Upsample(64, 4),
      nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid())

  def forward(self, *inputs):
    m = self.mask(torch.cat(inputs, dim=1))
    return self.body(inputs[1]) * m + inputs[0] * (1 - m)
