#   Copyright (c): Wenyi Tang 2017-2019.
#   Author: Wenyi Tang
#   Email: wenyi.tang@intel.com
#   Update Date: 4/1/19, 7:13 PM

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .Model import SuperResolution
from .Ops.Blocks import RB
from .Ops.Loss import total_variance
from .Ops.Motion import Flownet, STN
from .Ops.Scale import SpaceToDepth, Upsample
from ..Framework.Summary import get_writer
from ..Util import Metrics
from ..Util.Utility import pad_if_divide, upsample

_logger = logging.getLogger("VSR.FRVSR")
_logger.info("LICENSE: FRVSR is proposed by Sajjadi, et. al. "
             "implemented by LoSeall. "
             "@loseall https://github.com/loseall/VideoSuperResolution")


class SRNet(nn.Module):
  def __init__(self, channel, scale, n_rb=10):
    super(SRNet, self).__init__()
    rbs = [RB(64, activation='relu') for _ in range(n_rb)]
    entry = [nn.Conv2d(channel * (scale ** 2 + 1), 64, 3, 1, 1), nn.ReLU(True)]
    up = Upsample(64, scale, method='ps')
    out = nn.Conv2d(64, channel, 3, 1, 1)
    self.body = nn.Sequential(*entry, *rbs, up, out)

  def forward(self, *inputs):
    x = torch.cat(inputs, dim=1)
    return self.body(x)


class FRNet(nn.Module):
  def __init__(self, channel, scale, n_rb):
    super(FRNet, self).__init__()
    self.fnet = Flownet(channel)
    self.warp = STN(padding_mode='border')
    self.snet = SRNet(channel, scale, n_rb)
    self.space_to_depth = SpaceToDepth(scale)
    self.scale = scale

  def forward(self, lr, last_lr, last_sr):
    flow = self.fnet(lr, last_lr, gain=32)
    flow2 = self.scale * upsample(flow, self.scale)
    hw = self.warp(last_sr, flow2[:, 0], flow2[:, 1])
    lw = self.warp(last_lr, flow[:, 0], flow[:, 1])
    hws = self.space_to_depth(hw)
    y = self.snet(hws, lr)
    return y, hw, lw, flow2


class FRVSR(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(FRVSR, self).__init__(scale, channel, **kwargs)
    self.frvsr = FRNet(channel, scale, kwargs.get('n_rb', 10))
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.w = kwargs.get('weights', [1, 1, 1e-3])

  def train(self, inputs, labels, learning_rate=None):
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    if learning_rate:
      for param_group in self.adam.param_groups:
        param_group["lr"] = learning_rate
    total_loss = 0
    flow_loss = 0
    image_loss = 0
    last_lr = frames[0]
    last_sr = upsample(last_lr, self.scale)
    for lr, hr in zip(frames, labels):
      sr, hrw, lrw, flow = self.frvsr(lr, last_lr, last_sr.detach())
      last_lr = lr
      last_sr = sr
      l2_image = F.mse_loss(sr, hr)
      l2_warp = F.mse_loss(lrw, lr)
      tv_flow = total_variance(flow)
      loss = l2_image * self.w[0] + l2_warp * self.w[1] + tv_flow * self.w[2]
      self.adam.zero_grad()
      loss.backward()
      self.adam.step()
      total_loss += loss.detach()
      image_loss += l2_image.detach()
      flow_loss += l2_warp.detach()
    return {
      'total_loss': total_loss.cpu().numpy() / len(frames),
      'image_loss': image_loss.cpu().numpy() / len(frames),
      'flow_loss': flow_loss.cpu().numpy() / len(frames),
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    predicts = []
    last_lr = pad_if_divide(frames[0], 8, 'reflect')
    a = (last_lr.size(2) - frames[0].size(2)) * self.scale
    b = (last_lr.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    last_sr = upsample(last_lr, self.scale)
    for lr in frames:
      lr = pad_if_divide(lr, 8, 'reflect')
      sr, _, _, _ = self.frvsr(lr, last_lr, last_sr)
      last_lr = lr.detach()
      last_sr = sr.detach()
      sr = sr[..., slice_h, slice_w]
      predicts.append(sr.cpu().detach().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      psnr = [Metrics.psnr(x, y) for x, y in zip(predicts, labels)]
      metrics['psnr'] = np.mean(psnr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('clean', sr.clamp(0, 1), step=step)
        writer.image('label', labels[-1], step=step)
    return predicts, metrics
