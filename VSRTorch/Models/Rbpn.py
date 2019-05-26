#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 下午3:24

import torch
import torch.nn.functional as F
from torch import nn

from .Loss import total_variance
from .Model import SuperResolution
from .frvsr.ops import FNet
from .rbpn.ops import Rbpn
from .video.motion import STN
from ..Framework.Summary import get_writer
from ..Util.Metrics import psnr
from ..Util.Utility import pad_if_divide, upsample


class Composer(nn.Module):
  def __init__(self, **kwargs):
    super(Composer, self).__init__()
    self.module = Rbpn(**kwargs)
    self.fnet = FNet(kwargs['num_channels'])
    self.warper = STN(padding_mode='border')

  def forward(self, target, neighbors):
    flows = []
    warps = []
    for i in neighbors:
      flow = self.fnet(target, i)
      warp = self.warper(i, flow[:, 0], flow[:, 1])
      flows.append(flow)
      warps.append(warp)
    sr = self.module(target, neighbors, flows)
    return sr, flows, warps


class RBPN(SuperResolution):
  def __init__(self, scale, channel, depth, residual, **kwargs):
    super(RBPN, self).__init__(scale, channel, **kwargs)
    self.depth = depth
    self.res = residual
    self.w = kwargs.get('weights', [1, 1e-4])
    ops = {
      'num_channels': channel,
      'scale_factor': scale,
      'base_filter': kwargs.get('base_filter', 256),
      'feat': kwargs.get('feat', 64),
      'num_stages': kwargs.get('num_stages', 3),
      'n_resblock': kwargs.get('n_resblock', 5),
      'nFrames': depth
    }
    self.rbpn = Composer(**ops)
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    target = frames.pop(self.depth // 2)
    neighbors = frames
    sr, flows, warps = self.rbpn(target, neighbors)
    if self.res:
      sr = sr + upsample(target, self.scale)

    image_loss = F.l1_loss(sr, labels[self.depth // 2])
    warp_loss = [F.l1_loss(w, target) for w in warps]
    tv_loss = [total_variance(f) for f in flows]
    flow_loss = torch.stack(warp_loss).sum() * self.w[0] + \
                torch.stack(tv_loss).sum() * self.w[1]
    loss = image_loss + flow_loss
    self.adam.zero_grad()
    loss.backward()
    self.adam.step()
    return {
      'flow': flow_loss.detach().cpu().numpy(),
      'image': image_loss.detach().cpu().numpy(),
      'total': loss.detach().cpu().numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    _frames = [pad_if_divide(x, 8, 'reflect') for x in frames]
    target = _frames.pop(self.depth // 2)
    neighbors = _frames
    a = (target.size(2) - frames[0].size(2)) * self.scale
    b = (target.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    sr, _, _ = self.rbpn(target, neighbors)
    if self.res:
      sr = sr + upsample(target, self.scale)
    sr = sr[..., slice_h, slice_w].detach().cpu().numpy()
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      gt = pad_if_divide(labels[self.depth // 2], 8, 'reflect')
      metrics['psnr'] = psnr(sr, gt)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('hr', gt, step=step)
        writer.image('sr', sr.clamp(0, 1), step=step)
    return [sr], metrics
