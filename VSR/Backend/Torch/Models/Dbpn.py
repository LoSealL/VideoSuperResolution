"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-3-15

DBPN is implemented by Haris.
alterzero https://github.com/alterzero/DBPN-Pytorch
"""
import logging

import torch
import torch.nn as nn

from .Ops.Blocks import EasyConv2d
from .Optim.SISR import L1Optimizer

_logger = logging.getLogger("VSR.DBPN")
_logger.info("LICENSE: DBPN is implemented by Haris. "
             "@alterzero https://github.com/alterzero/DBPN-Pytorch")


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, activation='prelu'):
        super(UpBlock, self).__init__()
        self.up_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                   activation=activation, transposed=True)
        self.up_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                   activation=activation)
        self.up_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                   activation=activation, transposed=True)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, activation='prelu'):
        super(DownBlock, self).__init__()
        self.down_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                     activation=activation)
        self.down_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                     activation=activation, transposed=True)
        self.down_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                     activation=activation)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, num_stages=1,
                 activation='prelu'):
        super(D_UpBlock, self).__init__()
        self.conv = EasyConv2d(num_filter * num_stages, num_filter, 1,
                               activation=activation)
        self.up_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                   activation=activation, transposed=True)
        self.up_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                   activation=activation)
        self.up_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                   activation=activation, transposed=True)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, num_stages=1,
                 activation='prelu'):
        super(D_DownBlock, self).__init__()
        self.conv = EasyConv2d(num_filter * num_stages, num_filter, 1,
                               activation=activation)
        self.down_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                     activation=activation)
        self.down_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                     activation=activation, transposed=True)
        self.down_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride,
                                     activation=activation)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class Dbpn(nn.Module):
    def __init__(self, channels, scale, base_filter=64, feat=256, num_stages=7):
        super(Dbpn, self).__init__()
        kernel, stride = self.get_kernel_stride(scale)

        # Initial Feature Extraction
        self.feat0 = EasyConv2d(channels, feat, 3, activation='prelu')
        self.feat1 = EasyConv2d(feat, base_filter, 1, activation='prelu')
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride)
        self.down1 = DownBlock(base_filter, kernel, stride)
        self.up2 = UpBlock(base_filter, kernel, stride)
        for i in range(2, num_stages):
            self.__setattr__(f'down{i}', D_DownBlock(
                base_filter, kernel, stride, i))
            self.__setattr__(
                f'up{i + 1}', D_UpBlock(base_filter, kernel, stride, i))
        self.num_stages = num_stages
        # Reconstruction
        self.output_conv = EasyConv2d(num_stages * base_filter, channels, 3)

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        h = h2
        concat_h = h1
        concat_l = l1
        for i in range(2, self.num_stages):
            concat_h = torch.cat((h, concat_h), 1)
            l = self.__getattr__(f'down{i}')(concat_h)
            concat_l = torch.cat((l, concat_l), 1)
            h = self.__getattr__(f'up{i + 1}')(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        return x

    @staticmethod
    def get_kernel_stride(scale):
        if scale == 2:
            return 6, 2
        elif scale == 4:
            return 8, 4
        elif scale == 8:
            return 12, 8


class DBPN(L1Optimizer):
    def __init__(self, channel, scale, base_filter=64, feat=256, num_stages=7,
                 **kwargs):
        self.body = Dbpn(channel, scale, base_filter, feat, num_stages)
        super(DBPN, self).__init__(scale, channel, **kwargs)

    def fn(self, x):
        return self.body(x)
