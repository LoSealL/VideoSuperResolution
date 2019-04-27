#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/27 下午2:37

import torch
from torch import nn

from ..Arch import Upsample, RB


class TecoGenerator(nn.Module):
  """Generator in TecoGAN.

  Note: the flow estimation net `Fnet` shares with FRVSR.

  Args:
    filters: basic filter numbers [default: 64]
    num_rb: number of residual blocks [default: 16]
  """

  def __init__(self, channel, scale, filters, num_rb):
    super(TecoGenerator, self).__init__()
    rbs = []
    for i in range(num_rb):
      rbs.append(RB(filters, 3, 'relu'))
    self.body = nn.Sequential(
      nn.Conv2d(channel * (1 + scale ** 2), filters, 3, 1, 1),
      nn.ReLU(True),
      *rbs,
      Upsample(filters, scale, 'deconv', activation='relu'),
      nn.Conv2d(filters, channel, 3, 1, 1))

  def forward(self, x, prev, residual=None):
    """`residual` is the bicubically upsampled HR images"""
    sr = self.body(torch.cat((x, prev), dim=1))
    if residual is not None:
      sr += residual
    return sr


class TecoDiscriminator(nn.Module):
  def __init__(self):
    super(TecoDiscriminator, self).__init__()
