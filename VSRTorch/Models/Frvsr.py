#   Copyright (c): Wenyi Tang 2017-2019.
#   Author: Wenyi Tang
#   Email: wenyi.tang@intel.com
#   Update Date: 4/1/19, 7:13 PM

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .Arch import SpaceToDepth
from .Loss import total_variance
from .Model import SuperResolution
from .frvsr.ops import FNet, SRNet
from .video.motion import STN
from ..Framework.Summary import get_writer
from ..Util import Metrics
from ..Util.Utility import pad_if_divide


class FRNet(nn.Module):
  def __init__(self, channel, scale, n_rb):
    super(FRNet, self).__init__()
    self.fnet = FNet(channel, gain=32)
    self.warp = STN(padding_mode='border')
    self.snet = SRNet(channel, scale, n_rb)
    self.space_to_depth = SpaceToDepth(scale)
    self.scale = scale

  def forward(self, lr, last_lr, last_sr):
    flow = self.fnet(lr, last_lr)
    flow2 = self.scale * F.interpolate(
      flow, scale_factor=self.scale, mode='bilinear', align_corners=False)
    hw = self.warp(last_sr, flow2[:, 0], flow2[:, 1], normalized=False)
    lw = self.warp(last_lr, flow[:, 0], flow[:, 1], normalized=False)
    hws = self.space_to_depth(hw)
    y = self.snet(hws, lr)
    return y, hw, lw, flow2


class FRVSR(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(FRVSR, self).__init__(scale, channel, **kwargs)
    self.frvsr = FRNet(channel, scale, kwargs.get('n_rb', 10))
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    total_loss = 0
    flow_loss = 0
    image_loss = 0
    last_lr = frames[0]
    last_sr = F.interpolate(
      frames[0], scale_factor=self.scale, mode='bilinear', align_corners=False)
    for lr, hr in zip(frames, labels):
      sr, hrw, lrw, flow = self.frvsr(lr, last_lr, last_sr)
      last_lr = lr.detach()
      last_sr = sr.detach()
      l2_image = F.mse_loss(sr, hr)
      l2_warp = F.mse_loss(lrw, lr)
      tv_flow = total_variance(flow)
      loss = l2_image + l2_warp + 0.001 * tv_flow
      if learning_rate:
        for param_group in self.adam.param_groups:
          param_group["lr"] = learning_rate
      self.adam.zero_grad()
      loss.backward()
      self.adam.step()
      total_loss += loss
      image_loss += l2_image
      flow_loss += l2_warp
    return {
      'total_loss': total_loss.detach().cpu().numpy() / len(frames),
      'image_loss': image_loss.detach().cpu().numpy() / len(frames),
      'flow_loss': flow_loss.detach().cpu().numpy() / len(frames),
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
    last_sr = F.interpolate(
      frames[0], scale_factor=self.scale, mode='bilinear', align_corners=False)
    for lr in frames:
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
        writer.image('clean', sr, step=step)
    return predicts, metrics
