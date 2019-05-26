#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 下午3:21

import os
import torch.nn as nn
import torch.optim as optim
from ..dbpn.base_networks import *
from torchvision.transforms import *


class Dbpns(nn.Module):
  def __init__(self, base_filter, feat, num_stages, scale_factor):
    super(Dbpns, self).__init__()

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
    # self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
    self.feat1 = ConvBlock(base_filter, feat, 1, 1, 0, activation='prelu',
                           norm=None)
    # Back-projection stages
    self.up1 = UpBlock(feat, kernel, stride, padding)
    self.down1 = DownBlock(feat, kernel, stride, padding)
    self.up2 = UpBlock(feat, kernel, stride, padding)
    self.down2 = DownBlock(feat, kernel, stride, padding)
    self.up3 = UpBlock(feat, kernel, stride, padding)
    # Reconstruction
    self.output = ConvBlock(num_stages * feat, feat, 1, 1, 0, activation=None,
                            norm=None)

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
    # x = self.feat0(x)
    x = self.feat1(x)

    h1 = self.up1(x)
    h2 = self.up2(self.down1(h1))
    h3 = self.up3(self.down2(h2))

    x = self.output(torch.cat((h3, h2, h1), 1))

    return x


class Rbpn(nn.Module):
  def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock,
               nFrames, scale_factor):
    super(Rbpn, self).__init__()
    # base_filter=256
    # feat=64
    self.nFrames = nFrames

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
    self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1,
                           activation='prelu', norm=None)
    self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='prelu',
                           norm=None)

    ###DBPNS
    self.DBPN = Dbpns(base_filter, feat, num_stages, scale_factor)

    # Res-Block1
    modules_body1 = [
      ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True,
                  activation='prelu', norm=None) \
      for _ in range(n_resblock)]
    modules_body1.append(
      DeconvBlock(base_filter, feat, kernel, stride, padding,
                  activation='prelu', norm=None))
    self.res_feat1 = nn.Sequential(*modules_body1)

    # Res-Block2
    modules_body2 = [
      ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True,
                  activation='prelu', norm=None) \
      for _ in range(n_resblock)]
    modules_body2.append(
      ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
    self.res_feat2 = nn.Sequential(*modules_body2)

    # Res-Block3
    modules_body3 = [
      ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True,
                  activation='prelu', norm=None) \
      for _ in range(n_resblock)]
    modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding,
                                   activation='prelu', norm=None))
    self.res_feat3 = nn.Sequential(*modules_body3)

    # Reconstruction
    self.output = ConvBlock((nFrames - 1) * feat, num_channels, 3, 1, 1,
                            activation=None, norm=None)

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

  def forward(self, x, neigbor, flow):
    ### initial feature extraction
    feat_input = self.feat0(x)
    feat_frame = []
    for j in range(len(neigbor)):
      feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]), 1)))

    ####Projection
    Ht = []
    for j in range(len(neigbor)):
      h0 = self.DBPN(feat_input)
      h1 = self.res_feat1(feat_frame[j])

      e = h0 - h1
      e = self.res_feat2(e)
      h = h0 + e
      Ht.append(h)
      feat_input = self.res_feat3(h)

    ####Reconstruction
    out = torch.cat(Ht, 1)
    output = self.output(out)

    return output
