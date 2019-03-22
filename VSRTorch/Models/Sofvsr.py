#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 22

from .sof.modules import SOFVSR as _SOFVSR

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from ..Util import Metrics


class SOFVSR(SuperResolution):
  """Note: SOF is Y-channel SR with depth=3"""

  def __init__(self, scale, channel, depth=3, **kwargs):
    super(SOFVSR, self).__init__(scale, channel, **kwargs)
    self.sof = _SOFVSR(scale)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    assert depth == 3
    self.center = depth // 2

  def train(self, inputs, labels, learning_rate=None):
    pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
    low_res = torch.cat([pre, cur, nxt], dim=2)
    low_res = torch.squeeze(low_res, dim=1)
    sr = self.sof(low_res)
    hr = labels[0][:, self.center]
    loss = F.l1_loss(sr, hr)
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
    low_res = torch.cat([pre, cur, nxt], dim=2)
    low_res = torch.squeeze(low_res, dim=1)
    sr = self.sof(low_res)
    sr = sr.cpu().detach()
    if labels is not None:
      hr = labels[0][: self.center]
      metrics['psnr'] = Metrics.psnr(sr.numpy(), hr.cpu().numpy())
    return [sr.numpy()], metrics
