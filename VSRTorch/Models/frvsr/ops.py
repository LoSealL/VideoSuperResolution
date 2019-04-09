#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/4 下午8:51

import torch
from torch import nn

from ..Arch import Upsample


class FNet(nn.Module):
  def __init__(self, channel, gain=32):
    super(FNet, self).__init__()
    f = 32
    layers = []
    in_c = channel * 2
    for i in range(3):
      layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.MaxPool2d(2)]
      in_c = f
      f *= 2
    for i in range(3):
      layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [Upsample(f, scale=2, method='linear')]
      in_c = f
      f //= 2
    layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
    layers += [nn.Conv2d(f, 2, 3, 1, 1), nn.Tanh()]
    self.body = nn.Sequential(*layers)
    self.gain = gain

  def forward(self, *inputs):
    x = torch.cat(inputs, dim=1)
    return self.body(x) * self.gain


class RB(nn.Module):
  def __init__(self, channel):
    super(RB, self).__init__()
    conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
    conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
    self.body = nn.Sequential(conv1, nn.ReLU(True), conv2)

  def forward(self, x):
    return x + self.body(x)


class SRNet(nn.Module):
  def __init__(self, channel, scale, n_rb=10):
    super(SRNet, self).__init__()
    rbs = [RB(64) for _ in range(n_rb)]
    entry = [nn.Conv2d(channel * (scale ** 2 + 1), 64, 3, 1, 1), nn.ReLU(True)]
    up = Upsample(64, scale, method='ps')
    out = nn.Conv2d(64, channel, 3, 1, 1)
    self.body = nn.Sequential(*entry, *rbs, up, out)

  def forward(self, *inputs):
    x = torch.cat(inputs, dim=1)
    return self.body(x)
