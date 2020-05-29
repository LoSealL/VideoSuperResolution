#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .dbpn import dbpn, dbpn_v1, dbpns
from ..Util import Metrics


class DBPNMaker(torch.nn.Module):
  def __init__(self, mode='dbpn', **kwargs):
    super(DBPNMaker, self).__init__()
    _allowed_mode = ('dbpn', 'dbpnll', 'dbpns')
    mode = mode.lower()
    assert mode in _allowed_mode, "mode must in ('DBPN', 'DBPNLL', 'DBPNS)."
    if mode == 'dbpn':
      self.module = dbpn.Net(**kwargs)
    elif mode == 'dbpnll':
      self.module = dbpn_v1.Net(**kwargs)
    elif mode == 'dbpns':
      self.module = dbpns.Net(**kwargs)
    else:
      raise NotImplemented

  def forward(self, x):
    return self.module(x)


class DBPN(SuperResolution):

  def __init__(self, scale, mode='dbpn', **kwargs):
    super(DBPN, self).__init__(scale, 3)
    self.body = DBPNMaker(mode, scale_factor=scale, **kwargs)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    sr = self.body(inputs[0])
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
    sr = self.body(inputs[0]).cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics

  def export(self, export_dir):
    device = list(self.body.parameters())[0].device
    inputs = torch.randn(1, self.channel, 144, 128, device=device)
    torch.onnx.export(self.body, (inputs,), export_dir / 'dbpn.onnx')
