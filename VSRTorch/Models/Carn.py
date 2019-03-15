#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 13

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .carn import carn, carn_m
from ..Util import Metrics


class CARN(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(CARN, self).__init__(scale, channel, **kwargs)
    group = kwargs.get('group', 1)
    ms = kwargs.get('multi_scale', 0)
    if group > 1:
      self.carn = carn_m.Net(group=group, scale=scale, multi_scale=ms)
    else:
      self.carn = carn.Net(scale=scale, multi_scale=ms)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    sr = self.carn(inputs[0], self.scale)
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
    sr = self.carn(inputs[0], self.scale).cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
