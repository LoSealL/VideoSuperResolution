#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/21 下午4:56

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Arch import Upsample, EasyConv2d
from .Model import SuperResolution
from ..Framework.Summary import get_writer
from ..Util import Metrics, Utility
from .Loss import total_variance


class NoiseExtractor(nn.Module):
  def __init__(self, channel=3, layers=8, bn=False, **kwargs):
    super(NoiseExtractor, self).__init__()
    f = kwargs.get('filters', 32)
    ks = kwargs.get('kernel_size', 3)
    convs = [EasyConv2d(channel, f, ks, use_bn=bn, activation='lrelu')]
    for i in range(1, layers - 1):
      convs += [EasyConv2d(f, f, ks, use_bn=bn, activation='lrelu')]
    convs += [EasyConv2d(f, channel, ks)]
    self.body = nn.Sequential(*convs)

  def forward(self, x):
    return self.body(x)


class NoiseShifter(nn.Module):
  def __init__(self, channel=3, layers=8, bn=False, **kwargs):
    super(NoiseShifter, self).__init__()
    f = kwargs.get('filters', 32)
    ks = kwargs.get('kernel_size', 3)
    convs = [EasyConv2d(channel, f, ks, use_bn=bn, activation='lrelu')]
    for i in range(1, layers - 1):
      convs += [EasyConv2d(f, f, ks, use_bn=bn, activation='lrelu')]
    convs += [EasyConv2d(f, channel, ks, activation='sigmoid')]
    self.body = nn.Sequential(*convs)

  def forward(self, x):
    return self.body(x)


class NCL(nn.Module):
  def __init__(self, channels, filters=32, layers=3, **kwargs):
    super(NCL, self).__init__()
    ks = kwargs.get('kernel_size', 3)
    c = channels
    f = filters
    conv = []
    for i in range(1, layers):
      if i == 1:
        conv.append(EasyConv2d(3, f, ks, activation='lrelu'))
      else:
        conv.append(EasyConv2d(f, f, ks, activation='lrelu'))
    self.gamma = nn.Sequential(
      *conv, EasyConv2d(f, c, ks, activation='sigmoid'))
    self.beta = nn.Sequential(
      *conv.copy(), EasyConv2d(f, c, ks))

  def forward(self, x, noise=None):
    if noise is None:
      return x
    return x * self.gamma(noise) + self.beta(noise)


class CRDB(nn.Module):
  def __init__(self, channels, depth=3, scaling=1.0, name='Rdb', **kwargs):
    super(CRDB, self).__init__()
    self.name = name
    self.depth = depth
    self.scaling = scaling
    ks = kwargs.get('kernel_size', 3)
    stride = kwargs.get('stride', 1)
    padding = kwargs.get('padding', ks // 2)
    dilation = kwargs.get('dilation', 1)
    group = kwargs.get('group', 1)
    bias = kwargs.get('bias', True)
    c = channels
    for i in range(depth):
      conv = nn.Conv2d(
        c + c * i, c, ks, stride, padding, dilation, group, bias)
      if i < depth - 1:  # no activation after last layer
        conv = nn.Sequential(conv, nn.ReLU(True))
      setattr(self, f'conv_{i}', conv)
    self.ncl = NCL(c)

  def forward(self, inputs, noise):
    fl = [inputs]
    for i in range(self.depth):
      conv = getattr(self, f'conv_{i}')
      fl.append(conv(torch.cat(fl, dim=1)))
    y = fl[-1] * self.scaling + inputs
    return self.ncl(y, noise)


class CascadeRdn(nn.Module):
  def __init__(self, channels, depth=(3, 3), name='CascadeRdn', **kwargs):
    super(CascadeRdn, self).__init__()
    self.name = name
    self.depth = depth
    c = channels
    for i in range(self.depth[0]):
      setattr(self, f'conv11_{i}', nn.Conv2d(c + c * (i + 1), c, 1))
      setattr(self, f'rdn_{i}', CRDB(c, self.depth[1], **kwargs))

  def forward(self, inputs, noise):
    fl = [inputs]
    x = inputs
    for i in range(self.depth[0]):
      rdn = getattr(self, f'rdn_{i}')
      x = rdn(x, noise)
      fl.append(x)
      c11 = getattr(self, f'conv11_{i}')
      x = c11(torch.cat(fl, dim=1))

    return x


class Drn(nn.Module):
  def __init__(self, channel, scale, n_cb, **kwargs):
    super(Drn, self).__init__()
    f = kwargs.get('filters', 64)
    self.entry = nn.Sequential(
      nn.Conv2d(channel, f, 3, 1, 1), nn.Conv2d(f, f, 3, 1, 1))
    for i in range(n_cb):
      setattr(self, f'cb{i}', CascadeRdn(f))
    self.n_cb = n_cb
    self.tail = nn.Sequential(
      Upsample(f, scale), nn.Conv2d(f, channel, 3, 1, 1))

  def forward(self, x, noise=None):
    x0 = self.entry(x)
    x = x0
    for i in range(self.n_cb):
      cb = getattr(self, f'cb{i}')
      x = cb(x, noise)
    x += x0
    return self.tail(x)


class DRN(SuperResolution):
  def __init__(self, channel, scale, n_cb, noise, offset=0):
    super(DRN, self).__init__(channel=channel, scale=scale)
    self.noise = noise
    self.drn = Drn(channel, scale, n_cb)
    self.ne = NoiseExtractor(channel, bn=False)
    self.ns = NoiseShifter(channel, bn=False)
    p1 = self.trainable_variables('drn') + self.trainable_variables('ne')
    p2 = self.trainable_variables('ns')
    self.offset = offset
    if self.noise < 0:
      self.opt = torch.optim.Adam(p2, 1e-4)
    else:
      self.opt = torch.optim.Adam(p1, 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    x0 = inputs[0]
    metrics = {}
    if self.noise > 0:
      stddev = torch.rand(1) * self.noise / 255
      stddev = stddev.reshape([1, 1, 1, 1])
      noise_map = torch.randn(*x0.shape) * stddev
      noise_map = noise_map.to(x0.device)
      x0 = (x0 + noise_map).clamp(0, 1)
      noise = self.ne(x0)
      l2_noise = F.mse_loss(noise, noise_map)
      metrics['noise'] = l2_noise.detach().cpu().numpy()
    elif self.noise < 0:
      stddev = self.ns(x0)
      noise_map = torch.randn(*x0.shape, device=x0.device) * stddev
      noise = self.ne(x0) + noise_map
      l2_noise = 0
    else:
      noise = None
      l2_noise = 0

    y = self.drn(x0, noise)
    l1_image = F.l1_loss(y, labels[0])
    loss = l1_image + 10 * l2_noise
    if self.noise != 0:
      tv = total_variance(noise)
      loss += tv * 1.0e-3
      metrics['tv'] = tv.detach().cpu().numpy()
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    metrics['loss'] = loss.detach().cpu().numpy()
    metrics['image'] = l1_image.detach().cpu().numpy()
    return metrics

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    x0 = inputs[0]
    if self.noise > 0:
      stddev = torch.rand(1) * self.noise / 255
      stddev = stddev.reshape([1, 1, 1, 1])
      noise_map = torch.randn(*x0.shape) * stddev
      noise_map = noise_map.to(x0.device)
      x0 = (x0 + noise_map).clamp(0, 1)
      noise = self.ne(x0)
    elif self.offset > 0:
      noise = self.ne(x0)
      stddev = torch.ones(3, dtype=torch.float32) * self.offset / 255
      stddev = stddev.reshape([1, 3, 1, 1])
      noise_map = torch.randn(*x0.shape) * stddev
      noise_map = noise_map.to(x0.device)
      noise += noise_map
    elif self.noise < 0:
      stddev = self.ns(x0)
      noise_map = torch.randn(*x0.shape, device=x0.device) * stddev
      noise = self.ne(x0) + noise_map
    else:
      noise = None

    y = self.drn(x0, noise)
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(y, labels[0])
      writer = get_writer(self.name)
      step = kwargs['epoch']
      if writer is not None:
        writer.image('sr', y.clamp(0, 1), step=step)
        writer.image('hr', labels[0], step=step)
        writer.image('lr', x0, step=step)
    return [y.detach().cpu().numpy()], metrics
