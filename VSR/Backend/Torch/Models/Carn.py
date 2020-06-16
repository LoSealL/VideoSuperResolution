#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 13

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d, MeanShift, RB
from .Ops.Scale import MultiscaleUpsample, Upsample
from ..Util import Metrics

_logger = logging.getLogger("VSR.CARN")
_logger.info("LICENSE: CARN is implemented by Namhyuk Ahn. "
             "@nmhkahn https://github.com/nmhkahn/CARN-pytorch")


class EResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, group):
    super(EResidualBlock, self).__init__()

    self.body = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1, 1, 0),
    )

  def forward(self, x):
    out = self.body(x)
    return out + x


class ResidualBlock(nn.Module):
  def __init__(self,
               in_channels, out_channels):
    super(ResidualBlock, self).__init__()

    self.body = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
    )

  def forward(self, x):
    out = self.body(x)
    out = F.relu(out + x)
    return out


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, group=1):
    """ CARN cascading residual block
    """
    super(Block, self).__init__()
    if group == 1:
      self.b1 = RB(in_channels, out_channels, activation='relu')
      self.b2 = RB(out_channels, out_channels, activation='relu')
      self.b3 = RB(out_channels, out_channels, activation='relu')
    elif group > 1:
      self.b1 = EResidualBlock(64, 64, group=group)
      self.b2 = self.b3 = self.b1
    self.c1 = EasyConv2d(in_channels + out_channels, out_channels, 1,
                         activation='relu')
    self.c2 = EasyConv2d(in_channels + out_channels * 2, out_channels, 1,
                         activation='relu')
    self.c3 = EasyConv2d(in_channels + out_channels * 3, out_channels, 1,
                         activation='relu')

  def forward(self, x):
    c0 = o0 = x

    b1 = F.relu(self.b1(o0))
    c1 = torch.cat([c0, b1], dim=1)
    o1 = self.c1(c1)

    b2 = F.relu(self.b2(o1))
    c2 = torch.cat([c1, b2], dim=1)
    o2 = self.c2(c2)

    b3 = F.relu(self.b3(o2))
    c3 = torch.cat([c2, b3], dim=1)
    o3 = self.c3(c3)

    return o3


class Net(nn.Module):
  def __init__(self, scale, multi_scale=None, group=1):
    super(Net, self).__init__()

    self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
    self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

    self.entry = nn.Conv2d(3, 64, 3, 1, 1)

    self.b1 = Block(64, 64, group=group)
    self.b2 = Block(64, 64, group=group)
    self.b3 = Block(64, 64, group=group)
    self.c1 = EasyConv2d(64 * 2, 64, 1, activation='relu')
    self.c2 = EasyConv2d(64 * 3, 64, 1, activation='relu')
    self.c3 = EasyConv2d(64 * 4, 64, 1, activation='relu')

    if multi_scale:
      self.upsample = MultiscaleUpsample(64, scales=(2, 3, 4), group=group,
                                         activation='relu')
    else:
      self.upsample = Upsample(64, scale=scale, group=group, activation='relu')
    self.exit = nn.Conv2d(64, 3, 3, 1, 1)

  def forward(self, x, scale=None):
    x = self.sub_mean(x)
    x = self.entry(x)
    c0 = o0 = x

    b1 = self.b1(o0)
    c1 = torch.cat([c0, b1], dim=1)
    o1 = self.c1(c1)

    b2 = self.b2(o1)
    c2 = torch.cat([c1, b2], dim=1)
    o2 = self.c2(c2)

    b3 = self.b3(o2)
    c3 = torch.cat([c2, b3], dim=1)
    o3 = self.c3(c3)

    out = self.upsample(o3, scale=scale)

    out = self.exit(out)
    out = self.add_mean(out)

    return out


class CARN(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(CARN, self).__init__(scale, channel, **kwargs)
    group = kwargs.get('group', 1)
    ms = kwargs.get('multi_scale', 0)
    self.clip = kwargs.get('clip', 10)
    self.carn = Net(group=group, scale=scale, multi_scale=ms)
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
