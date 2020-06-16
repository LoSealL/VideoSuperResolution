#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 15

import torch.nn as nn
import torch.nn.functional as F

from .Blocks import Activation, EasyConv2d


class _UpsampleNearest(nn.Module):
  def __init__(self, scale):
    super(_UpsampleNearest, self).__init__()
    self.scale = scale

  def forward(self, x, scale=None):
    scale = scale or self.scale
    return F.interpolate(x, scale_factor=scale)


class _UpsampleLinear(nn.Module):
  def __init__(self, scale):
    super(_UpsampleLinear, self).__init__()
    self._mode = ('linear', 'bilinear', 'trilinear')
    self.scale = scale

  def forward(self, x, scale=None):
    scale = scale or self.scale
    mode = self._mode[x.dim() - 3]
    return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)


class Upsample(nn.Module):
  def __init__(self, channel, scale, method='ps', name='Upsample', **kwargs):
    super(Upsample, self).__init__()
    self.name = name
    self.channel = channel
    self.scale = scale
    self.method = method.lower()
    self.group = kwargs.get('group', 1)
    self.kernel_size = kwargs.get('kernel_size', 3)

    _allowed_methods = ('ps', 'nearest', 'deconv', 'linear')
    assert self.method in _allowed_methods
    act = kwargs.get('activation')

    samplers = []
    while scale > 1:
      if scale % 2 == 1 or scale == 2:
        samplers.append(self.upsampler(self.method, scale, act))
        break
      else:
        samplers.append(self.upsampler(self.method, 2, act))
        scale //= 2
    self.body = nn.Sequential(*samplers)

  def upsampler(self, method, scale, activation=None):
    body = []
    k = self.kernel_size
    if method == 'ps':
      p = k // 2  # padding
      s = 1  # strides
      body = [nn.Conv2d(self.channel, self.channel * scale * scale, k, s, p,
                        groups=self.group),
              nn.PixelShuffle(scale)]
      if activation:
        body.insert(1, Activation(activation))
    if method == 'deconv':
      q = k % 2  # output padding
      p = (k + q) // 2 - 1  # padding
      s = scale  # strides
      body = [nn.ConvTranspose2d(self.channel, self.channel, k, s, p, q,
                                 groups=self.group)]
      if activation:
        body.insert(1, Activation(activation))
    if method == 'nearest':
      body = [_UpsampleNearest(scale),
              nn.Conv2d(self.channel, self.channel, k, 1, k // 2,
                        groups=self.group)]
      if activation:
        body.append(Activation(activation))
    if method == 'linear':
      body = [_UpsampleLinear(scale),
              nn.Conv2d(self.channel, self.channel, k, 1, k // 2,
                        groups=self.group)]
      if activation:
        body.append(Activation(activation))
    return nn.Sequential(*body)

  def forward(self, x, **kwargs):
    return self.body(x)

  def extra_repr(self):
    return f"{self.name}: scale={self.scale}"


class MultiscaleUpsample(nn.Module):
  def __init__(self, channel, scales=(2, 3, 4), **kwargs):
    super(MultiscaleUpsample, self).__init__()
    for i in scales:
      self.__setattr__(f'up{i}', Upsample(channel, i, **kwargs))

  def forward(self, x, scale):
    return self.__getattr__(f'up{scale}')(x)


class SpaceToDim(nn.Module):
  def __init__(self, scale_factor, dims=(-2, -1), dim=0):
    super(SpaceToDim, self).__init__()
    self.scale_factor = scale_factor
    self.dims = dims
    self.dim = dim

  def forward(self, x):
    _shape = list(x.shape)
    shape = _shape.copy()
    dims = [x.dim() + self.dims[0] if self.dims[0] < 0 else self.dims[0],
            x.dim() + self.dims[1] if self.dims[1] < 0 else self.dims[1]]
    dims = [max(abs(dims[0]), abs(dims[1])),
            min(abs(dims[0]), abs(dims[1]))]
    if self.dim in dims:
      raise RuntimeError("Integrate dimension can't be space dimension!")
    shape[dims[0]] //= self.scale_factor
    shape[dims[1]] //= self.scale_factor
    shape.insert(dims[0] + 1, self.scale_factor)
    shape.insert(dims[1] + 1, self.scale_factor)
    dim = self.dim if self.dim < dims[1] else self.dim + 1
    dim = dim if dim <= dims[0] else dim + 1
    x = x.reshape(*shape)
    perm = [dim, dims[1] + 1, dims[0] + 2]
    perm = [i for i in range(min(perm))] + perm
    perm.extend((i for i in range(x.dim()) if i not in perm))
    x = x.permute(*perm)
    shape = _shape
    shape[self.dim] *= self.scale_factor ** 2
    shape[self.dims[0]] //= self.scale_factor
    shape[self.dims[1]] //= self.scale_factor
    return x.reshape(*shape)

  def extra_repr(self):
    return f'scale_factor={self.scale_factor}'


class SpaceToDepth(nn.Module):
  def __init__(self, block_size):
    super(SpaceToDepth, self).__init__()
    self.body = SpaceToDim(block_size, dim=1)

  def forward(self, x):
    return self.body(x)


class SpaceToBatch(nn.Module):
  def __init__(self, block_size):
    super(SpaceToBatch, self).__init__()
    self.body = SpaceToDim(block_size, dim=0)

  def forward(self, x):
    return self.body(x)
