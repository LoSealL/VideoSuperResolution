#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Ops.Blocks import EasyConv2d, MeanShift
from .Ops.Scale import Upsample
from .Optim.SISR import L1Optimizer

_logger = logging.getLogger("VSR.MSRN")
_logger.info("LICENSE: MSRN is implemented by Juncheng Li. "
             "@MIVRC https://github.com/MIVRC/MSRN-PyTorch")


class MSRB(nn.Module):
  def __init__(self, n_feats=64):
    super(MSRB, self).__init__()
    self.conv_3_1 = EasyConv2d(n_feats, n_feats, 3)
    self.conv_3_2 = EasyConv2d(n_feats * 2, n_feats * 2, 3)
    self.conv_5_1 = EasyConv2d(n_feats, n_feats, 5)
    self.conv_5_2 = EasyConv2d(n_feats * 2, n_feats * 2, 5)
    self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)

  def forward(self, x):
    input_1 = x
    output_3_1 = F.relu(self.conv_3_1(input_1))
    output_5_1 = F.relu(self.conv_5_1(input_1))
    input_2 = torch.cat([output_3_1, output_5_1], 1)
    output_3_2 = F.relu(self.conv_3_2(input_2))
    output_5_2 = F.relu(self.conv_5_2(input_2))
    input_3 = torch.cat([output_3_2, output_5_2], 1)
    output = self.confusion(input_3)
    output += x
    return output


class Msrn(nn.Module):
  def __init__(self, channel, scale, n_feats, n_blocks, rgb_range):
    super(Msrn, self).__init__()
    self.n_blocks = n_blocks
    # RGB mean for DIV2K
    rgb_mean = (0.4488, 0.4371, 0.4040)
    self.sub_mean = MeanShift(rgb_mean, True, rgb_range)
    # define head module
    modules_head = [EasyConv2d(channel, n_feats, 3)]
    # define body module
    modules_body = nn.ModuleList()
    for i in range(n_blocks):
      modules_body.append(MSRB(n_feats=n_feats))
    # define tail module
    modules_tail = [
      EasyConv2d(n_feats * (self.n_blocks + 1), n_feats, 1),
      EasyConv2d(n_feats, n_feats, 3),
      Upsample(n_feats, scale),
      EasyConv2d(n_feats, channel, 3)]

    self.add_mean = MeanShift(rgb_mean, False, rgb_range)
    self.head = nn.Sequential(*modules_head)
    self.body = nn.Sequential(*modules_body)
    self.tail = nn.Sequential(*modules_tail)

  def forward(self, x):
    x = self.sub_mean(x)
    x = self.head(x)
    res = x

    MSRB_out = []
    for i in range(self.n_blocks):
      x = self.body[i](x)
      MSRB_out.append(x)
    MSRB_out.append(res)

    res = torch.cat(MSRB_out, 1)
    x = self.tail(res)
    x = self.add_mean(x)
    return x


class MSRN(L1Optimizer):
  def __init__(self, scale, channel, n_feats=64, n_blocks=8, rgb_range=255,
               **kwargs):
    self.rgb_range = rgb_range
    self.msrn = Msrn(channel, scale, n_feats, n_blocks, rgb_range)
    super(MSRN, self).__init__(scale, channel, **kwargs)

  def fn(self, x):
    return self.msrn(x * self.rgb_range) / self.rgb_range
