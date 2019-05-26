#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 下午12:49

import torch
from torch.nn import functional as F

from .spmc.ops import DetailRevealer
from .Model import SuperResolution
from .Loss import total_variance
from ..Util.Metrics import psnr


class SPMC(SuperResolution):
  def __init__(self, scale, channel, stage, lambda1, lambda2, **kwargs):
    super(SPMC, self).__init__(scale, channel)
    self.spmc = DetailRevealer(scale, channel, **kwargs)
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.stage = stage
    self.lambda1 = lambda1
    self.lambda2 = lambda2

  def train(self, inputs, labels, learning_rate=None):
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
    for ref in frames:
      sr, flow = self.spmc(target, ref)
      warp = self.spmc.me.warper(ref, flow[:, 0], flow[:, 1])
      srs.append(sr)
      warps.append(warp)
      flows.append(flow)
    losses = []
    for x, y in zip(srs, labels):
      losses.append(F.mse_loss(x, y))
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
    srs = []
    center = len(frames) // 2
    target = frames[center]
    for ref in frames:
      sr, _ = self.spmc(target, ref)
      srs.append(sr.detach().cpu().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      for i, v in enumerate(psnr(x, y) for x, y in zip(srs, labels)):
        metrics[f'psnr{i}'] = v
    return srs, metrics
