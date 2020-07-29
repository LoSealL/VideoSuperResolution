#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 21

import torch
import torch.nn as nn

from .Optim.SISR import PerceptualOptimizer
from .Ops.Blocks import EasyConv2d, RB
from ..Util.Utility import upsample


class Espcn(nn.Module):
  def __init__(self, channel, scale):
    super(Espcn, self).__init__()
    conv1 = nn.Conv2d(channel, 64, 5, 1, 2)
    conv2 = nn.Conv2d(64, 32, 3, 1, 1)
    conv3 = nn.Conv2d(32, channel * scale * scale, 3, 1, 1)
    ps = nn.PixelShuffle(scale)
    self.body = nn.Sequential(conv1, nn.Tanh(),
                              conv2, nn.Tanh(),
                              conv3, nn.Tanh(), ps)

  def forward(self, x):
    return self.body(x)


class Srcnn(nn.Module):
  def __init__(self, channel, filters=(9, 5, 5)):
    super(Srcnn, self).__init__()
    self.net = nn.Sequential(
        EasyConv2d(channel, 64, filters[0], activation='relu'),
        EasyConv2d(64, 32, filters[1], activation='relu'),
        EasyConv2d(32, channel, filters[2], activation=None))

  def forward(self, x):
    return self.net(x)


class Vdsr(nn.Module):
  def __init__(self, channel, layers=20):
    super(Vdsr, self).__init__()
    net = [EasyConv2d(channel, 64, 3, activation='relu')]
    for i in range(1, layers - 1):
      net.append(EasyConv2d(64, 64, 3, activation='relu'))
    net.append(EasyConv2d(64, channel, 3))
    self.net = nn.Sequential(*net)

  def forward(self, x):
    return self.net(x) + x


class DnCnn(nn.Module):
  def __init__(self, channel, layers, bn):
    super(DnCnn, self).__init__()
    net = [EasyConv2d(channel, 64, 3, activation='relu', use_bn=bn)]
    for i in range(1, layers - 1):
      net.append(EasyConv2d(64, 64, 3, activation='relu', use_bn=bn))
    net.append(EasyConv2d(64, channel, 3))
    self.net = nn.Sequential(*net)

  def forward(self, x):
    return self.net(x) + x


class Drcn(nn.Module):
  def __init__(self, scale, channel, n_recur, filters):
    from torch.nn import Parameter

    super(Drcn, self).__init__()
    self.entry = nn.Sequential(
        EasyConv2d(channel, filters, 3, activation='relu'),
        EasyConv2d(filters, filters, 3, activation='relu'))
    self.exit = nn.Sequential(
        EasyConv2d(filters, filters, 3, activation='relu'),
        EasyConv2d(filters, channel, 3))
    self.conv = EasyConv2d(filters, filters, 3, activation='relu')
    self.output_weights = Parameter(torch.empty(n_recur + 1))
    torch.nn.init.uniform_(self.output_weights, 0, 1)
    self.n_recur = n_recur
    self.scale = scale

  def forward(self, x):
    bic = upsample(x, self.scale)
    y = [self.entry(bic)]
    for i in range(self.n_recur):
      y.append(self.conv(y[-1]))
    sr = [self.exit(i) for i in y[1:]]
    final = bic * self.output_weights[0]
    for i in range(len(sr)):
      final = final + self.output_weights[i + 1] * sr[i]
    return final


class Drrn(nn.Module):
  def __init__(self, channel, n_ru, n_rb, filters):
    super(Drrn, self).__init__()
    self.entry0 = EasyConv2d(channel, filters, 3, activation='relu')
    for i in range(1, n_rb):
      setattr(self, f'entry{i}',
              EasyConv2d(filters, filters, 3, activation='relu'))
    self.n_rb = n_rb
    self.rb = RB(filters, kernel_size=3, activation='relu')
    self.n_ru = n_ru
    self.exit = EasyConv2d(filters, channel, 3)

  def forward(self, x):
    for i in range(self.n_rb):
      entry = getattr(self, f'entry{i}')
      y = entry(x)
      for j in range(self.n_ru):
        y = self.rb(y)
      x = y
    return self.exit(x)


class ESPCN(PerceptualOptimizer):
  def __init__(self, scale, channel, **kwargs):
    self.espcn = Espcn(channel, scale)
    super(ESPCN, self).__init__(scale, channel, **kwargs)

  def fn(self, tensor):
    return self.espcn(tensor * 2 - 1) / 2 + 0.5


class SRCNN(PerceptualOptimizer):
  def __init__(self, scale, channel, **kwargs):
    filters = kwargs.get('filters', (9, 5, 5))
    self.srcnn = Srcnn(channel, filters)
    super(SRCNN, self).__init__(scale, channel, **kwargs)

  def fn(self, tensor):
    x = upsample(tensor, self.scale)
    return self.srcnn(x)


class VDSR(PerceptualOptimizer):
  def __init__(self, scale, channel, **kwargs):
    layers = kwargs.get('layers', 20)
    self.vdsr = Vdsr(channel, layers)
    super(VDSR, self).__init__(scale, channel, **kwargs)

  def fn(self, tensor):
    x = upsample(tensor, self.scale)
    return self.vdsr(x)


class DNCNN(PerceptualOptimizer):
  def __init__(self, channel, noise, **kwargs):
    layers = kwargs.get('layers', 15)
    bn = kwargs.get('bn', True)
    self.dncnn = DnCnn(channel, layers, bn)
    self.noise = noise / 255
    self.norm = torch.distributions.normal.Normal(0, self.noise)
    super(DNCNN, self).__init__(1, channel, **kwargs)

  def fn(self, tensor):
    if self.noise > 0:
      device = tensor.device
      noise = self.norm.sample(tensor.shape)
      tensor += noise.to(device)
    return self.dncnn(tensor)


class DRCN(PerceptualOptimizer):
  def __init__(self, scale, channel, n_recur, **kwargs):
    self.drcn = Drcn(scale, channel, n_recur, 128)
    super(DRCN, self).__init__(scale, channel, **kwargs)

  def fn(self, tensor):
    return self.drcn(tensor)


class DRRN(PerceptualOptimizer):
  def __init__(self, scale, channel, n_rb, n_ru, **kwargs):
    self.drrn = Drrn(channel, n_ru, n_rb, 128)
    super(DRRN, self).__init__(scale, channel, **kwargs)

  def fn(self, tensor):
    x = upsample(tensor, self.scale)
    return self.drrn(x)
