#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import torch.nn as nn

from .base_networks import *


class Net(nn.Module):
  def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
    super(Net, self).__init__()

    if scale_factor == 2:
      kernel = 6
      stride = 2
      padding = 2
    elif scale_factor == 4:
      kernel = 8
      stride = 4
      padding = 2
    elif scale_factor == 8:
      kernel = 12
      stride = 8
      padding = 2

    # Initial Feature Extraction
    self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu',
                           norm=None)
    self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu',
                           norm=None)
    # Back-projection stages
    self.up1 = UpBlock(base_filter, kernel, stride, padding)
    self.down1 = DownBlock(base_filter, kernel, stride, padding)
    self.up2 = UpBlock(base_filter, kernel, stride, padding)
    # Reconstruction
    self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1,
                                 1, activation=None, norm=None)

    for m in self.modules():
      classname = m.__class__.__name__
      if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          m.bias.data.zero_()
      elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          m.bias.data.zero_()

  def forward(self, x):
    x = self.feat0(x)
    x = self.feat1(x)

    h1 = self.up1(x)
    h2 = self.up2(self.down1(h1))

    x = self.output_conv(torch.cat((h2, h1), 1))

    return x
