#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .vespcn import ops
from ..Framework.Summary import get_writer
from ..Util import Metrics
from ..Util.Utility import pad_if_divide


class VESPCN(SuperResolution):
  def __init__(self, scale, channel, depth=3, **kwargs):
    super(VESPCN, self).__init__(scale, channel, **kwargs)
    self.vespcn = ops.VESPCN(scale, channel, depth)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.depth = depth

  def train(self, inputs, labels, learning_rate=None):
    frames = torch.split(inputs[0], 1, dim=1)
    frames = [f.squeeze(1) for f in frames]
    sr, warps, flows = self.vespcn(*frames)
    targets = torch.split(labels[0], 1, dim=1)
    targets = [t.squeeze(1) for t in targets]
    target = targets[self.depth // 2]
    ref = frames[self.depth // 2]

    loss_content = F.mse_loss(sr, target)
    loss_flow = torch.sum(torch.stack([F.mse_loss(ref, w) for w in warps]))
    loss_tv = torch.sum(torch.stack([Metrics.total_variance(f) for f in flows]))

    loss = loss_content + loss_flow + 0.01 * loss_tv
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {
      'image': loss_content.detach().cpu().numpy(),
      'flow': loss_flow.detach().cpu().numpy(),
      'tv': loss_tv.detach().cpu().numpy(),
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    _frames = [pad_if_divide(x, 4, 'reflect') for x in frames]
    a = (_frames[0].size(2) - frames[0].size(2)) * self.scale
    b = (_frames[0].size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    sr, warps, flows = self.vespcn(*_frames)
    sr = sr[..., slice_h, slice_w].cpu().detach()
    if labels is not None:
      targets = torch.split(labels[0], 1, dim=1)
      targets = [t.squeeze(1) for t in targets]
      hr = targets[self.depth // 2]
      metrics['psnr'] = Metrics.psnr(sr, hr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('clean', sr.clamp(0, 1), step=step)
        writer.image('warp/0', warps[0].clamp(0, 1), step=step)
        writer.image('warp/1', warps[-1].clamp(0, 1), step=step)
    return [sr.numpy()], metrics
