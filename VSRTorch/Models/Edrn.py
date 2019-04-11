#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/11 下午7:36

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .edrn import edrn
from ..Util import Metrics
from VSR.Util.Config import Config


class EDRN(SuperResolution):
  """EDRN is one candidate of NTIRE19 RSR"""

  def __init__(self, scale, channel, **kwargs):
    super(EDRN, self).__init__(scale, channel)
    args = Config(kwargs)
    args.scale = [scale]
    args.n_colors = channel
    self.rgb_range = args.rgb_range
    self.edrn = edrn.EDRN(args)
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    sr = self.edrn(inputs[0] * self.rgb_range) / self.rgb_range
    loss = F.l1_loss(sr, labels[0])
    if learning_rate:
      for param_group in self.adam.param_groups:
        param_group["lr"] = learning_rate
    self.adam.zero_grad()
    loss.backward()
    self.adam.step()
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.edrn(inputs[0] * self.rgb_range) / self.rgb_range
    sr = sr.cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
