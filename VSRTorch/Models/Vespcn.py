#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/2 上午10:54

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .vespcn import ops
from ..Util import Metrics
from ..Framework.Summary import get_writer


class VESPCN(SuperResolution):
  def __init__(self, scale, channel, depth=3, **kwargs):
    super(VESPCN, self).__init__(scale, channel, **kwargs)
    self.vespcn = ops.VESPCN(scale, channel, depth)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.depth = depth

  def train(self, inputs, labels, learning_rate=None):
    frames = torch.split(inputs[0], 1, dim=1)
    frames = [torch.squeeze(f, dim=1) for f in frames]
    sr, warps, flows = self.vespcn(*frames)
    targets = torch.split(labels[0], 1, dim=1)
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
    frames = torch.split(inputs[0], 1, dim=1)
    frames = [torch.squeeze(f, dim=1) for f in frames]
    sr, warps, flows = self.vespcn(*frames)
    sr = sr.cpu().detach()
    if labels is not None:
      targets = torch.split(labels[0], 1, dim=1)
      hr = targets[self.depth // 2]
      metrics['psnr'] = Metrics.psnr(sr, hr)
      writer = get_writer(self.name)
      if writer is not None:
        writer.image('clean', sr)
        writer.image('warp/0', warps[0])
        writer.image('warp/1', warps[-1])
    return [sr.numpy()], metrics
