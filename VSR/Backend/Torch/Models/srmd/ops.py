#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 11

import torch
import torch.nn as nn

from ..Arch import EasyConv2d


class Net(nn.Module):
  """
  SRMD CNN network. 12 conv layers
  """

  def __init__(self, scale=4, channels=3, layers=12, filters=128,
               pca_length=15):
    super(Net, self).__init__()
    self.pca_length = pca_length
    net = [EasyConv2d(channels + pca_length + 1, filters, 3, activation='relu')]
    net += [EasyConv2d(filters, filters, 3, activation='relu') for _ in
            range(layers - 2)]
    net += [EasyConv2d(filters, channels * scale ** 2, 3),
            nn.PixelShuffle(scale)]
    self.body = nn.Sequential(*net)

  def forward(self, x, kernel, noise):
    # degradation parameter
    degpar = torch.cat([kernel, noise.reshape([-1, 1, 1])], dim=1)
    degpar = degpar.reshape([-1, 1 + self.pca_length, 1, 1])
    degpar = torch.ones_like(x)[:, 0:1] * degpar
    _x = torch.cat([x, degpar], dim=1)
    return self.body(_x)
