#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/2 上午10:54

import torch
import torch.nn as nn
import torch.nn.functional as F

from VSR.Util.Utility import to_list
from VSRTorch.Util.Utility import nd_meshgrid, transpose, irtranspose


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


class STN(nn.Module):
  """Spatial transformer network.
    For optical flow based frame warping.
  """

  def __init__(self, mode='bilinear', padding_mode='zeros'):
    super(STN, self).__init__()
    self.mode = mode
    self.padding_mode = padding_mode

  def forward(self, inputs, u, v, normalized=True):
    batch = inputs.shape[0]
    device = inputs.device
    mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
    mesh = torch.stack([torch.Tensor(mesh)] * batch)
    # add flow to mesh
    _u, _v = u, v
    if not normalized:
      # flow needs to normalize to [-1, 1]
      h, w = inputs.shape[-2:]
      _u = u / w * 2
      _v = v / h * 2
    flow = torch.stack([_u, _v], dim=-1)
    assert flow.shape == mesh.shape
    mesh = mesh.to(device)
    mesh += flow
    return F.grid_sample(inputs, mesh,
                         mode=self.mode, padding_mode=self.padding_mode)


class STTN(nn.Module):
  """Spatio-temporal transformer network. (ECCV 2018)"""

  def __init__(self, transpose_ncthw=(0, 1, 2, 3, 4),
               mode='bilinear', padding_mode='zeros'):
    super(STTN, self).__init__()
    self.mode = mode
    self.padding_mode = padding_mode
    self.t = transpose_ncthw

  def forward(self, inputs, d, u, v, normalized=True):
    _error_msg = "STTN only works for 5D tensor but got {}D input!"
    if inputs.dim() != 5:
      raise ValueError(_error_msg.format(inputs.dim()))
    device = inputs.device
    batch, channel, t, h, w = (inputs.shape[i] for i in self.t)
    mesh = nd_meshgrid(t, h, w, permute=[2, 1, 0])
    mesh = torch.stack([torch.Tensor(mesh)] * batch)
    _d, _u, _v = t, w, h
    if not normalized:
      _d = d / t * 2
      _u = u / w * 2
      _v = v / h * 2
    st_flow = torch.stack([_d, _u, _v], dim=-1)
    assert st_flow.shape == mesh.shape
    mesh = mesh.to(device)
    mesh += st_flow
    inputs = transpose(inputs, self.t)
    warp = F.grid_sample(inputs, mesh, mode=self.mode,
                         padding_mode=self.padding_mode)
    # STTN warps into a single frame
    warp = warp[:, :, 0:1]
    return irtranspose(warp, self.t)
