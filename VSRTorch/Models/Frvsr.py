#   Copyright (c): Wenyi Tang 2017-2019.
#   Author: Wenyi Tang
#   Email: wenyi.tang@intel.com
#   Update Date: 4/1/19, 7:13 PM

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from .video.motion import STN
from .frvsr.ops import FNet, SRNet
from .Arch import SpaceToDepth

from .Model import SuperResolution
from ..Framework.Trainer import SRTrainer, to_tensor, from_tensor
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
    self.last_sr = self.last_lr = None
    self.scale = scale

  def reset(self):
    self.last_sr = None
    self.last_lr = None

  def forward(self, x):
    if self.last_sr is None:
      self.last_lr = x.detach()
      self.last_sr = F.interpolate(x.detach(), scale_factor=self.scale)
    flow = self.fnet(self.last_lr, x)
    flow2 = F.interpolate(flow * self.scale, scale_factor=self.scale,
                          mode='bilinear', align_corners=False)
    hw = self.warp(self.last_sr, flow2[:, 0], flow2[:, 1], normalized=False)
    lw = self.warp(self.last_lr, flow[:, 0], flow[:, 1], normalized=False)
    # TODO space-to-depth
    hws = self.space_to_depth(hw)
    y = self.snet(hws, x)
    self.last_lr = x.detach()
    self.last_sr = y.detach()
    return y, hw, lw, flow2


class FRVSR(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(FRVSR, self).__init__(scale, channel, **kwargs)
    self.frvsr = FRNet(channel, scale, kwargs.get('n_rb', 10))
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self._trainer = FRTrainer

  def train(self, inputs, labels, learning_rate=None):
    frames = torch.split(inputs[0], 1, dim=1)
    labels = torch.split(labels[0], 1, dim=1)
    total_loss = 0
    flow_loss = 0
    image_loss = 0
    self.frvsr.reset()
    last_hq = None
    for lq, hq in zip(frames, labels):
      lq = lq.squeeze(1)
      hq = hq.squeeze(1)
      y, hqw, lqw, flow = self.frvsr(lq)
      l2_image = F.mse_loss(hq, y)
      l2_warp = F.mse_loss(lqw, lq)
      if last_hq is not None:
        hqw = self.frvsr.warp(last_hq, flow[:, 0], flow[:, 1], False)
        l2_warp += F.mse_loss(hqw, hq)
        last_hq = hq
      loss = l2_image + l2_warp * 0.5
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
    frames = torch.split(inputs[0], 1, dim=1)
    self.frvsr.reset()
    predicts = []
    for lq in frames:
      lq = lq.squeeze(1)
      lq = pad_if_divide(lq, 8, 'reflect')
      y, hqw, lqw, flow = self.frvsr(lq)
      predicts.append(y.cpu().detach().numpy())
    if labels is not None:
      targets = torch.split(labels[0], 1, dim=1)
      targets = [pad_if_divide(t.squeeze(1), 8 * self.scale) for t in targets]
      psnr = [Metrics.psnr(x, y) for x, y in zip(predicts, targets)]
      metrics['psnr'] = np.mean(psnr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('clean', y, step=step)
        writer.image('warp', lqw, step=step)
    return predicts, metrics


class FRTrainer(SRTrainer):
  def fn_benchmark_each_step(self, label=None, feature=None, name=None,
                             post=None):
    v = self.v
    origin_feat = feature
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    feature = to_tensor(feature, v.cuda)
    label = to_tensor(label, v.cuda)
    outputs, metrics = self.model.eval([feature], [label], epoch=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    outputs = [from_tensor(x) for x in outputs]
    for fn in v.output_callbacks:
      outputs = fn(outputs, input=origin_feat, label=label, name=name,
                   mode=v.color_format, subdir=v.subdir)
