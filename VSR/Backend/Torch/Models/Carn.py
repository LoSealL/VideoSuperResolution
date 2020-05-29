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
    self.clip = kwargs.get('clip', 10)
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
    torch.nn.utils.clip_grad_norm_(self.carn.parameters(), self.clip)
    self.opt.step()
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.carn(inputs[0], self.scale).cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics

  def export(self, export_dir):
    """An example of how to export ONNX format"""

    # ONNX needs input placeholder to export model!
    # Sounds stupid to set a 48x48 inputs.

    device = list(self.carn.parameters())[0].device
    inputs = torch.randn(1, self.channel, 144, 128, device=device)
    scale = torch.tensor(self.scale, device=device)
    torch.onnx.export(self.carn, (inputs, scale), export_dir / 'carn.onnx')
