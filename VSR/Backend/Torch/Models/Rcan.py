#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d, MeanShift, Rcab
from .Ops.Scale import Upsample
from ..Util import Metrics

_logger = logging.getLogger("VSR.RCAN")
_logger.info("LICENSE: RCAN is implemented by Yulun Zhang. "
             "@yulunzhang https://github.com/yulunzhang/RCAN.")


class ResidualGroup(nn.Module):
  def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
    super(ResidualGroup, self).__init__()
    modules_body = [Rcab(n_feat, reduction, kernel_size=kernel_size) for _ in
                    range(n_resblocks)]
    modules_body.append(EasyConv2d(n_feat, n_feat, kernel_size))
    self.body = nn.Sequential(*modules_body)

  def forward(self, x):
    res = self.body(x)
    res += x
    return res


class Rcan(nn.Module):
  def __init__(self, channel, scale, n_resgroups, n_resblocks, n_feats,
               reduction, rgb_range):
    super(Rcan, self).__init__()
    # RGB mean for DIV2K
    rgb_mean = (0.4488, 0.4371, 0.4040)
    self.sub_mean = MeanShift(rgb_mean, True, rgb_range)
    # define head module
    modules_head = [EasyConv2d(channel, n_feats, 3)]
    # define body module
    modules_body = [
      ResidualGroup(n_feats, 3, reduction, n_resblocks) for _ in
      range(n_resgroups)]
    modules_body.append(EasyConv2d(n_feats, n_feats, 3))
    # define tail module
    modules_tail = [Upsample(n_feats, scale),
                    EasyConv2d(n_feats, channel, 3)]
    self.add_mean = MeanShift(rgb_mean, False, rgb_range)
    self.head = nn.Sequential(*modules_head)
    self.body = nn.Sequential(*modules_body)
    self.tail = nn.Sequential(*modules_tail)

  def forward(self, x):
    x = self.sub_mean(x)
    x = self.head(x)
    res = self.body(x) + x
    x = self.tail(res)
    x = self.add_mean(x)
    return x


class RCAN(SuperResolution):
  def __init__(self, channel, scale, n_resgroups, n_resblocks, n_feats,
               reduction, **kwargs):
    super(RCAN, self).__init__(scale, channel)
    self.rgb_range = kwargs.get('rgb_range', 255)
    self.rcan = Rcan(channel, scale, n_resgroups, n_resblocks, n_feats,
                     reduction, self.rgb_range)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    sr = self.rcan(inputs[0] * self.rgb_range) / self.rgb_range
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
    sr = self.rcan(inputs[0] * self.rgb_range) / self.rgb_range
    sr = sr.cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
