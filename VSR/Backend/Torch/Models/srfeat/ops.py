#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/27 下午11:06

import torch
import torch.nn as nn

from ..Arch import RB, Upsample


class Generator(nn.Module):
  """Generator for SRFeat: Single Image Super-Resolution with Feature Discrimination (ECCV 2018)

  """

  def __init__(self, channel, scale, filters, num_rb):
    super(Generator, self).__init__()
    self.head = nn.Conv2d(channel, filters, 9, 1, 4)
    for i in range(num_rb):
      setattr(self, f'rb_{i:02d}', RB(filters, 3, 'lrelu', use_bn=True))
      setattr(self, f'merge_{i:02d}', nn.Conv2d(filters, filters, 1))
    self.tail = nn.Sequential(
      Upsample(filters, scale),
      nn.Conv2d(filters, channel, 3, 1, 1))
    self.num_rb = num_rb

  def forward(self, inputs):
    x = self.head(inputs)
    feat = []
    for i in range(self.num_rb):
      x = getattr(self, f'rb_{i:02d}')(x)
      feat.append(getattr(self, f'merge_{i:02d}')(x))
    x = self.tail(x + torch.stack(feat, dim=0).sum(0).squeeze(0))
    return x
