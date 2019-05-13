#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/27 下午2:37

import torch
from torch import nn

from ..Arch import EasyConv2d, RB, Upsample


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
  def __init__(self, channel, filters, patch_size):
    super(TecoDiscriminator, self).__init__()
    f = filters
    conv0 = EasyConv2d(channel * 8, f, 3, activation='leaky')
    conv1 = EasyConv2d(f, f, 4, 2, activation='leaky', use_bn=True)
    conv11 = EasyConv2d(f, f, 3, 1, activation='leaky', use_bn=True)
    conv2 = EasyConv2d(f, f, 4, 2, activation='leaky', use_bn=True)
    conv21 = EasyConv2d(f, f, 3, 1, activation='leaky', use_bn=True)
    conv3 = EasyConv2d(f, f * 2, 4, 2, activation='leaky', use_bn=True)
    conv31 = EasyConv2d(f * 2, f * 2, 3, 1, activation='leaky', use_bn=True)
    conv4 = EasyConv2d(f * 2, f * 4, 4, 2, activation='leaky', use_bn=True)
    conv41 = EasyConv2d(f * 4, f * 4, 3, 1, activation='leaky', use_bn=True)
    self.body = nn.Sequential(conv0, conv1, conv11, conv2, conv21,
                              conv3, conv31, conv4, conv41)
    self.linear = nn.Sequential(
      nn.Linear(f * 4 * (patch_size // 16) ** 2, 100),
      nn.LeakyReLU(inplace=True),
      nn.Linear(100, 1))

  def forward(self, x):
    """The inputs `x` is the concat of 8 tensors.
      Note that we remove the duplicated gt/yt in paper (9 - 1 = 8).
    """
    y = self.body(x)
    y = self.linear(y.flatten(1))
    return y, None
