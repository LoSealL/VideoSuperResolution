#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/16

import torch
import torch.nn.functional as F

from VSR.Util.Config import Config
from .Model import SuperResolution
from .ntire19 import denoise, edrn, frn, ran2
from ..Util import Metrics


class L1Optimizer(SuperResolution):
  def __init__(self, channel, scale=1):
    super(L1Optimizer, self).__init__(scale, channel)

  def fn(self, x):
    raise NotImplementedError

  def train(self, inputs, labels, learning_rate=None):
    sr = self.fn(inputs[0])
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
    sr = self.fn(inputs[0])
    sr = sr.cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics


class EDRN(L1Optimizer):
  """EDRN is one candidate of NTIRE19 RSR"""

  def __init__(self, scale, channel, **kwargs):
    super(EDRN, self).__init__(channel=channel, scale=scale)
    args = Config(kwargs)
    args.scale = [scale]
    args.n_colors = channel
    self.rgb_range = args.rgb_range
    self.edrn = edrn.EDRN(args)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.edrn(x * self.rgb_range) / self.rgb_range


class FRN(L1Optimizer):
  def __init__(self, scale, channel, **kwargs):
    super(FRN, self).__init__(channel=channel, scale=scale)
    args = Config(kwargs)
    args.scale = [scale]
    args.n_colors = channel
    self.rgb_range = args.rgb_range
    self.frn = frn.FRN_UPDOWN(args)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.frn(x * self.rgb_range) / self.rgb_range


class RAN(L1Optimizer):
  def __init__(self, scale, channel, **kwargs):
    super(RAN, self).__init__(channel=channel, scale=scale)
    args = Config(kwargs)
    args.scale = [scale]
    args.n_colors = channel
    self.rgb_range = args.rgb_range
    self.ran = ran2.RAN(args)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.ran(x * self.rgb_range) / self.rgb_range


class DIDN(L1Optimizer):
  def __init__(self, channel, filters, umodule, **kwargs):
    super(DIDN, self).__init__(channel=channel)
    self.didn = denoise.EraserTeam.DIDN(channel, filters, umodule)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.didn(x)


class DHDN(L1Optimizer):
  def __init__(self, channel, filters, **kwargs):
    super(DHDN, self).__init__(channel=channel)
    self.dhdn = denoise.EraserTeam.DHDN(channel, filters)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.dhdn(x)


class GRDN(L1Optimizer):
  def __init__(self, channel, filters, grdb, rdb, **kwargs):
    super(GRDN, self).__init__(channel=channel)
    self.grdn = denoise.DGUTeam.GRDN(channel, filters, grdb, rdb)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.grdn(x)


class ResUNet(L1Optimizer):
  def __init__(self, channel, filters, rb, **kwargs):
    super(ResUNet, self).__init__(channel=channel)
    self.resunet = denoise.HITVPCTeam.ResUNet(channel, filters, rb)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def fn(self, x):
    return self.resunet(x)
