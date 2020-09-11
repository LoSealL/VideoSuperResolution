"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-3-15

RRDB/ESRGAN
"""
import logging

import numpy as np
import torch.nn as nn

from .Ops.Blocks import Activation, EasyConv2d, Rrdb
from .Ops.Discriminator import DCGAN
from .Ops.Scale import Upsample
from .Optim.SISR import PerceptualOptimizer

_logger = logging.getLogger("VSR.ESRGAN")
_logger.info("LICENSE: ESRGAN is implemented by Xintao Wang. "
             "@xinntao https://github.com/xinntao/ESRGAN")


class RRDB_Net(nn.Module):
    def __init__(self, channel, scale, nf, nb, gc=32):
        super(RRDB_Net, self).__init__()
        self.head = EasyConv2d(channel, nf, kernel_size=3)
        rb_blocks = [
            Rrdb(nf, gc, 5, 0.2, kernel_size=3,
                 activation=Activation('lrelu', negative_slope=0.2))
            for _ in range(nb)]
        LR_conv = EasyConv2d(nf, nf, kernel_size=3)
        upsampler = [Upsample(nf, scale, 'nearest',
                              activation=Activation('lrelu', negative_slope=0.2))]
        HR_conv0 = EasyConv2d(nf, nf, kernel_size=3, activation='lrelu',
                              negative_slope=0.2)
        HR_conv1 = EasyConv2d(nf, channel, kernel_size=3)
        self.body = nn.Sequential(*rb_blocks, LR_conv)
        self.tail = nn.Sequential(*upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x


class ESRGAN(PerceptualOptimizer):
    def __init__(self, channel, scale, patch_size=128, weights=(0.01, 1, 5e-3),
                 nf=64, nb=23, gc=32, **kwargs):
        self.rrdb = RRDB_Net(channel, scale, nf, nb, gc)
        super(ESRGAN, self).__init__(scale, channel,
                                     discriminator=DCGAN,
                                     discriminator_kwargs={
                                         'channel': channel,
                                         'scale': scale,
                                         'num_layers': np.log2(patch_size // 4) * 2,
                                         'norm': 'BN'
                                     },
                                     image_weight=weights[0],
                                     feature_weight=weights[1],
                                     gan_weight=weights[2], **kwargs)

    def fn(self, x):
        return self.rrdb(x)
