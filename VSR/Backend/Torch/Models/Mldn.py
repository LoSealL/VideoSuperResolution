#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import SuperResolution
from .Ops.Blocks import CascadeRdn
from .Ops.Scale import Upsample
from ..Framework.Summary import get_writer
from ..Util import Metrics, Utility


class NoiseExtractor(nn.Module):
  def __init__(self, channel=32, layers=7, bn=False, **kwargs):
    super(NoiseExtractor, self).__init__()
    convs = [nn.Conv2d(3, channel, 3, 1, 1), nn.ReLU(True)]
    if bn:
      convs.insert(-1, nn.BatchNorm2d(channel))
    for i in range(1, layers - 1):
      convs += [nn.Conv2d(channel, channel, 3, 1, 1), nn.ReLU(True)]
      if bn:
        convs.insert(-1, nn.BatchNorm2d(channel))
    convs += [nn.Conv2d(channel, 3, 3, 1, 1)]
    self.body = nn.Sequential(*convs)

  def forward(self, x):
    return self.body(x)


class NoiseRemover(nn.Module):
  def __init__(self, in_channel, up, **kwargs):
    super(NoiseRemover, self).__init__()
    entry = nn.Conv2d(in_channel, 64, 3, 1, 1)
    rdn1 = CascadeRdn(64, 3, True)
    rdn2 = CascadeRdn(64, 3, True)
    exits = nn.Conv2d(64, 3, 3, 1, 1)
    if up:
      up = Upsample(64, 2)
      self.body = nn.Sequential(entry, rdn1, rdn2, up, exits)
    else:
      self.body = nn.Sequential(entry, rdn1, rdn2, exits)

  def forward(self, x, noise=None):
    if noise is not None:
      x = torch.cat([x, noise], dim=1)
    x = self.body(x)
    return x


class Mldn(nn.Module):
  def __init__(self):
    super(Mldn, self).__init__()
    self.ne = NoiseExtractor(bn=True)
    self.sub_x8 = NoiseRemover(6, True)
    self.sub_x4 = NoiseRemover(6, True)
    self.sub_x2 = NoiseRemover(6, True)
    self.main = NoiseRemover(6, False)

  def forward(self, x, x2, x4, x8):
    noise = self.ne(x8)
    up4 = self.sub_x8(x8, noise)
    up2 = self.sub_x4(x4, up4)
    up1 = self.sub_x2(x2, up2)
    clean = self.main(x, up1)
    return clean, up1, up2, up4, noise


class MLDN(SuperResolution):

  def __init__(self, finetune=False, **kwargs):
    super(MLDN, self).__init__(scale=1, channel=3)
    self.drn = Mldn()
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.finetune = finetune

  def train(self, inputs, labels, learning_rate=None):
    x0 = inputs[0]
    x0 = Utility.pad_if_divide(x0, 8, 'reflect')
    if not self.finetune:
      stddev = torch.rand(3) * 75 / 255
      stddev = stddev.reshape([1, 3, 1, 1])
      noise_map = torch.randn(*x0.shape) * stddev
      noise_map = noise_map.to(x0.device)
      x0 += noise_map
      x0 = torch.clamp(x0, 0, 1)
    x8 = F.interpolate(x0, scale_factor=1 / 8, mode='bilinear')
    x4 = F.interpolate(x0, scale_factor=1 / 4, mode='bilinear')
    x2 = F.interpolate(x0, scale_factor=1 / 2, mode='bilinear')
    label0 = labels[0]
    label2 = F.interpolate(label0, scale_factor=1 / 2, mode='bilinear')
    label4 = F.interpolate(label0, scale_factor=1 / 4, mode='bilinear')
    label8 = F.interpolate(label0, scale_factor=1 / 8, mode='bilinear')

    clean, sub1, sub2, sub4, noise = self.drn(x0, x2, x4, x8)

    l1_clean = F.l1_loss(clean, label0)
    l1_sub1 = F.l1_loss(sub1, label0)
    l1_sub2 = F.l1_loss(sub2, label2)
    l1_sub4 = F.l1_loss(sub4, label4)
    l2_noise = F.mse_loss(noise, x8 - label8)

    loss = l1_sub4 + l2_noise + l1_sub2 + l1_sub1 + l1_clean

    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {
      'noise': l2_noise.detach().cpu().numpy(),
      'x8': l1_sub4.detach().cpu().numpy(),
      'x4': l1_sub2.detach().cpu().numpy(),
      'x2': l1_sub1.detach().cpu().numpy(),
      'x0': l1_clean.detach().cpu().numpy(),
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    x0 = inputs[0]
    x0 = Utility.pad_if_divide(x0, 8, 'reflect')
    x8 = F.interpolate(x0, scale_factor=1 / 8, mode='bilinear')
    x4 = F.interpolate(x0, scale_factor=1 / 4, mode='bilinear')
    x2 = F.interpolate(x0, scale_factor=1 / 2, mode='bilinear')
    if labels is not None:
      label0 = labels[0]
      label2 = F.interpolate(label0, scale_factor=1 / 2, mode='bilinear')
      label4 = F.interpolate(label0, scale_factor=1 / 4, mode='bilinear')
      label8 = F.interpolate(label0, scale_factor=1 / 8, mode='bilinear')
    outputs = self.drn(x0, x2, x4, x8)
    clean = outputs[0].detach().cpu()
    sub1 = outputs[1].detach().cpu()
    sub2 = outputs[2].detach().cpu()
    sub4 = outputs[3].detach().cpu()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(clean.numpy(), label0.cpu().numpy())
      writer = get_writer(self.name)
      if writer is not None:
        writer.image('clean', clean)
        writer.image('up2', sub1)
        writer.image('up4', sub2)
        writer.image('up8', sub4)
    return [clean.numpy(), sub1.numpy(), sub2.numpy(), sub4.numpy()], metrics
