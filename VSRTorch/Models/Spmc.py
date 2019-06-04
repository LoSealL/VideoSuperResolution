#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 下午12:49

import torch
from torch.nn import functional as F

from .Loss import total_variance
from .Model import SuperResolution
from .spmc.ops import DetailRevealer
from ..Framework.Summary import get_writer
from ..Util.Metrics import psnr
from ..Util.Utility import pad_if_divide, upsample


class SPMC(SuperResolution):
  def __init__(self, scale, channel, stage, lambda1, lambda2, residual,
               **kwargs):
    super(SPMC, self).__init__(scale, channel)
    self.spmc = DetailRevealer(scale, channel, **kwargs)
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.stage = stage
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.residual = residual

  def train(self, inputs, labels, learning_rate=None):
    self.spmc.reset()
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    srs = []
    warps = []
    flows = []
    center = len(frames) // 2
    target = frames[center]
    gt = labels[center]
    for ref in frames:
      sr, flow = self.spmc(target, ref)
      if self.residual:
        sr = sr + upsample(target, self.scale)
      warp = self.spmc.me.warper(ref, flow[:, 0], flow[:, 1])
      srs.append(sr)
      warps.append(warp)
      flows.append(flow)
    losses = [F.mse_loss(x, gt) for x in srs]
    image_loss = torch.stack(losses).sum()
    losses = []
    for w, f in zip(warps, flows):
      losses.append(F.l1_loss(w, target) + total_variance(f) * self.lambda1)
    me_loss = torch.stack(losses).sum()
    if self.stage == 1:
      loss = me_loss
    elif self.stage == 2:
      loss = image_loss
    else:
      loss = image_loss + me_loss * self.lambda2
    self.adam.zero_grad()
    loss.backward()
    self.adam.step()
    return {
      'me': me_loss.detach().cpu().numpy(),
      'image': image_loss.detach().cpu().numpy(),
      'total': loss.detach().cpu().numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    center = len(frames) // 2
    _frames = [pad_if_divide(x, 8, 'reflect') for x in frames]
    target = _frames[center]
    a = (target.size(2) - frames[0].size(2)) * self.scale
    b = (target.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    srs = []
    for ref in frames:
      sr, _ = self.spmc(target, ref)
      if self.residual:
        sr = sr + upsample(target, self.scale)
      srs.append(sr[..., slice_h, slice_w].detach().cpu().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      gt = labels[center]
      gt = pad_if_divide(gt, 8, 'reflect')
      for i, v in enumerate(psnr(x, gt) for x in srs):
        metrics[f'psnr{i}'] = v
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('hr', gt, step=step)
        writer.image('sr', sr.clamp(0, 1), step=step)
    return srs, metrics
