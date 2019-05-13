#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 21

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .Arch import EasyConv2d
from .Model import SuperResolution
from .Loss import VggFeatureLoss
from ..Util import Metrics
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


class PerceptualOptimizer(SuperResolution):
  def __init__(self, scale, channel, image_weight=1, feature_weight=0,
               **kwargs):
    super(PerceptualOptimizer, self).__init__(scale, channel, **kwargs)
    if feature_weight > 0:
      # tricks: do not save weights of vgg
      self.feature = [VggFeatureLoss(['block3_conv4'], True)]
    self.w = [image_weight, feature_weight]

  def cuda(self):
    super(PerceptualOptimizer, self).cuda()
    if self.w[1] > 0:
      self.feature[0].cuda()

  def train(self, inputs, labels, learning_rate=None):
    sr = self.fn(inputs[0])
    image_loss = F.mse_loss(sr, labels[0])
    loss = image_loss * self.w[0]
    if self.w[1] > 0:
      self.feature[0].eval()
      # sr = self.fn(inputs[0])
      feat_fake = self.feature[0](sr)[0]
      feat_real = self.feature[0](labels[0])[0]
      feature_loss = F.mse_loss(feat_fake, feat_real)
      loss += feature_loss * self.w[1]
    opt = list(self.opts.values())[0]
    if learning_rate:
      for param_group in opt.param_groups:
        param_group["lr"] = learning_rate
    opt.zero_grad()
    loss.backward()
    opt.step()
    return {
      'loss': loss.detach().cpu().numpy(),
      'image': image_loss.detach().cpu().numpy(),
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.fn(inputs[0]).detach().cpu()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics

  def fn(self, tensor):
    raise NotImplementedError


class ESPCN(PerceptualOptimizer):
  def __init__(self, scale, channel, **kwargs):
    super(ESPCN, self).__init__(scale, channel, **kwargs)
    self.espcn = Espcn(channel, scale)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, tensor):
    return self.espcn(tensor * 2 - 1) / 2 + 0.5


class SRCNN(PerceptualOptimizer):
  def __init__(self, scale, channel, **kwargs):
    super(SRCNN, self).__init__(scale, channel, **kwargs)
    filters = kwargs.get('filters', (9, 5, 5))
    self.srcnn = Srcnn(channel, filters)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, tensor):
    x = upsample(tensor, self.scale)
    return self.srcnn(x)


class VDSR(PerceptualOptimizer):
  def __init__(self, scale, channel, **kwargs):
    super(VDSR, self).__init__(scale, channel, **kwargs)
    layers = kwargs.get('layers', 20)
    self.vdsr = Vdsr(channel, layers)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, tensor):
    x = upsample(tensor, self.scale)
    return self.vdsr(x)


class DNCNN(PerceptualOptimizer):
  def __init__(self, channel, scale, noise, **kwargs):
    super(DNCNN, self).__init__(1, channel, **kwargs)
    layers = kwargs.get('layers', 15)
    bn = kwargs.get('bn', True)
    self.dncnn = DnCnn(channel, layers, bn)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.noise = noise / 255
    self.norm = torch.distributions.normal.Normal(0, self.noise)

  def fn(self, tensor):
    if self.noise > 0:
      device = tensor.device
      noise = self.norm.sample(tensor.shape)
      tensor += noise.to(device)
    return self.dncnn(tensor)
