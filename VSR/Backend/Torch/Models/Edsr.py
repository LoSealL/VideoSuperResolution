#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d, MeanShift, RB
from .Ops.Scale import MultiscaleUpsample, Upsample
from ..Util import Metrics

_logger = logging.getLogger("VSR.EDSR")
_logger.info("LICENSE: EDSR is implemented by Bee Lim. "
             "@thstkdgus35 https://github.com/thstkdgus35/EDSR-PyTorch")

url = {
  'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
  'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
  'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
  'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
  'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
  'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt',
  'r16f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr_baseline-a00cab12.pt',
  'r80f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr-4a78bedf.pt'
}


class Edsr(nn.Module):
  def __init__(self, scale, channel, n_resblocks, n_feats, rgb_range):
    super(Edsr, self).__init__()
    self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), True, rgb_range)
    self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), False, rgb_range)
    # define head module
    m_head = [EasyConv2d(channel, n_feats, 3)]
    # define body module
    m_body = [RB(n_feats, n_feats, 3, activation='relu') for _ in
              range(n_resblocks)]
    m_body.append(EasyConv2d(n_feats, n_feats, 3))
    # define tail module
    m_tail = [
      Upsample(n_feats, scale),
      EasyConv2d(n_feats, channel, 3)]
    self.head = nn.Sequential(*m_head)
    self.body = nn.Sequential(*m_body)
    self.tail = nn.Sequential(*m_tail)

  def forward(self, x, **kwargs):
    x = self.sub_mean(x)
    x = self.head(x)
    res = self.body(x) + x
    x = self.tail(res)
    x = self.add_mean(x)
    return x


class Mdsr(nn.Module):
  def __init__(self, scales, channel, n_resblocks, n_feats, rgb_range):
    super(Mdsr, self).__init__()
    self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), True, rgb_range)
    self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), False, rgb_range)
    m_head = [EasyConv2d(channel, n_feats, 3)]
    self.pre_process = nn.ModuleList([
      nn.Sequential(
          RB(n_feats, kernel_size=5, activation='relu'),
          RB(n_feats, kernel_size=5, activation='relu')
      ) for _ in scales
    ])
    m_body = [RB(n_feats, kernel_size=3, activation='relu') for _ in
              range(n_resblocks)]
    m_body.append(EasyConv2d(n_feats, n_feats, 3))
    self.upsample = MultiscaleUpsample(n_feats, scales)
    m_tail = [EasyConv2d(n_feats, channel, 3)]
    self.head = nn.Sequential(*m_head)
    self.body = nn.Sequential(*m_body)
    self.tail = nn.Sequential(*m_tail)

  def forward(self, x, scale):
    x = self.sub_mean(x)
    x = self.head(x)
    x = self.pre_process[scale](x)
    res = self.body(x) + x
    x = self.upsample(res, scale)
    x = self.tail(x)
    x = self.add_mean(x)
    return x


class EDSR(SuperResolution):
  def __init__(self, scale, channel, rgb_range=255, **kwargs):
    super(EDSR, self).__init__(scale, channel)
    self.rgb_range = rgb_range
    self.edsr = Edsr(scale, channel, rgb_range=rgb_range, **kwargs)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    sr = self.edsr(inputs[0] * self.rgb_range) / self.rgb_range
    loss = F.l1_loss(sr, labels[0])
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.edsr(inputs[0] * self.rgb_range) / self.rgb_range
    sr = sr.cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics


class MSDR(SuperResolution):
  def __init__(self, scale, channel, rgb_range=255, **kwargs):
    super(MSDR, self).__init__(scale, channel)
    self.rgb_range = rgb_range
    self.scales = (2, 3, 4)
    self.mdsr = Mdsr(self.scales, channel, rgb_range=rgb_range, **kwargs)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    scale = self.scales[random.randint(0, 3)]
    sr = self.mdsr(inputs[0] * self.rgb_range, scale) / self.rgb_range
    loss = F.l1_loss(sr, labels[0])
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.mdsr(inputs[0] * self.rgb_range, self.scale) / self.rgb_range
    sr = sr.cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
