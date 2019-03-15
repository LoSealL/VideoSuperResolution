#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 13

import torch
import torch.nn as nn
import torch.nn.functional as F
from VSR.Util.Utility import to_list


def _get_act(name, *args, inplace=False):
  if name.lower() == 'relu':
    return nn.ReLU(inplace)
  if name.lower() in ('lrelu', 'leaky', 'leakyrelu'):
    return nn.LeakyReLU(*args, inplace=inplace)
  if name.lower() == 'prelu':
    return nn.PReLU(*args)

  raise TypeError("Unknown activation name!")


class Rdb(nn.Module):
  def __init__(self, channels, depth=3, scaling=1.0, name='Rdb', **kwargs):
    super(Rdb, self).__init__()
    self.name = name
    self.depth = depth
    self.scaling = scaling
    in_c, out_c = to_list(channels, 2)
    ks = kwargs.get('kernel_size', 3)
    stride = kwargs.get('stride', 1)
    padding = kwargs.get('padding', ks // 2)
    dilation = kwargs.get('dilation', 1)
    group = kwargs.get('group', 1)
    bias = kwargs.get('bias', True)
    for i in range(depth):
      conv = nn.Conv2d(
        in_c + out_c * i, out_c, ks, stride, padding, dilation, group, bias)
      if i < depth - 1:  # no activation after last layer
        conv = nn.Sequential(conv, nn.ReLU(True))
      setattr(self, f'conv_{i}', conv)

  def forward(self, inputs):
    fl = [inputs]
    for i in range(self.depth):
      conv = getattr(self, f'conv_{i}')
      fl.append(conv(torch.cat(fl, dim=1)))
    return fl[-1] * self.scaling + inputs

  def extra_repr(self):
    return f"{self.name}: depth={self.depth}, scaling={self.scaling}"


class Rcab(nn.Module):
  def __init__(self, channels, ratio=16, name='RCAB', **kwargs):
    super(Rcab, self).__init__()
    self.name = name
    self.ratio = ratio
    in_c, out_c = to_list(channels, 2)
    ks = kwargs.get('kernel_size', 3)
    padding = kwargs.get('padding', ks // 2)
    group = kwargs.get('group', 1)
    bias = kwargs.get('bias', True)
    self.c1 = nn.Sequential(
      nn.Conv2d(in_c, out_c, ks, 1, padding, 1, group, bias),
      nn.ReLU(True))
    self.c2 = nn.Conv2d(out_c, out_c, ks, 1, padding, 1, group, bias)
    self.c3 = nn.Sequential(
      nn.Conv2d(out_c, out_c // ratio, 1, groups=group, bias=bias),
      nn.ReLU(True))
    self.c4 = nn.Sequential(
      nn.Conv2d(out_c // ratio, in_c, 1, groups=group, bias=bias),
      nn.Sigmoid())
    self.pooling = nn.AdaptiveAvgPool2d(1)

  def forward(self, inputs):
    x = self.c1(inputs)
    y = self.c2(x)
    x = self.pooling(y)
    x = self.c3(x)
    x = self.c4(x)
    y = x * y
    return inputs + y

  def extra_repr(self):
    return f"{self.name}: ratio={self.ratio}"


class CascadeRdn(nn.Module):
  def __init__(self, channels, depth=3, use_ca=False, name='CascadeRdn',
               **kwargs):
    super(CascadeRdn, self).__init__()
    self.name = name
    self.depth = to_list(depth, 2)
    self.ca = use_ca
    in_c, out_c = to_list(channels, 2)
    for i in range(self.depth[0]):
      setattr(self, f'conv11_{i}', nn.Conv2d(in_c + out_c * (i + 1), out_c, 1))
      setattr(self, f'rdn_{i}', Rdb(channels, self.depth[1], **kwargs))
      if use_ca:
        setattr(self, f'rcab_{i}', Rcab(channels))

  def forward(self, inputs):
    fl = [inputs]
    x = inputs
    for i in range(self.depth[0]):
      rdn = getattr(self, f'rdn_{i}')
      x = rdn(x)
      if self.ca:
        rcab = getattr(self, f'rcab_{i}')
        x = rcab(x)
      fl.append(x)
      c11 = getattr(self, f'conv11_{i}')
      x = c11(torch.cat(fl, dim=1))

    return x

  def extra_repr(self):
    return f"{self.name}: depth={self.depth}, ca={self.ca}"


class _UpsampleNearest(nn.Module):
  def forward(self, x, scale):
    F.interpolate(x, scale_factor=scale)


class Upsample(nn.Module):
  def __init__(self, channel, scale, method='ps', name='Upsample', **kwargs):
    super(Upsample, self).__init__()
    self.name = name
    self.channel = channel
    self.scale = scale
    self.method = method.lower()

    _allowed_methods = ('ps', 'nearest', 'deconv')
    assert self.method in _allowed_methods

    samplers = []
    while scale > 1:
      if scale % 2 == 1:
        samplers.append(self.upsampler(self.method, scale))
      else:
        samplers.append(self.upsampler(self.method, 2))
        scale //= 2
    self.body = nn.Sequential(*samplers)

  def upsampler(self, method, scale):
    if method == 'ps':
      return nn.Sequential(
        nn.Conv2d(self.channel, self.channel * scale * scale, 3, 1, 1),
        nn.PixelShuffle(scale))
    if method == 'deconv':
      return nn.ConvTranspose2d(self.channel, self.channel, 3, scale, 1)
    if method == 'nearest':
      return _UpsampleNearest()

  def forward(self, inputs):
    return self.body(inputs)

  def extra_repr(self):
    return f"{self.name}: scale={self.scale}"
