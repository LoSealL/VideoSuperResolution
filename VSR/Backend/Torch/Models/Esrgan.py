#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import SuperResolution
from .Ops.Blocks import Activation, EasyConv2d, Rrdb
from .Ops.Discriminator import DCGAN
from .Ops.Loss import VggFeatureLoss, gan_bce_loss
from .Ops.Scale import Upsample
from ..Framework.Summary import get_writer
from ..Util import Metrics

_logger = logging.getLogger("VSR.ESRGAN")
_logger.info("LICENSE: ESRGAN is implemented by Xintao Wang. "
             "@xinntao https://github.com/xinntao/ESRGAN")


class RRDB_Net(nn.Module):
  def __init__(self, channel, scale, nf, nb, gc=32):
    super(RRDB_Net, self).__init__()
    self.head = EasyConv2d(channel, nf, kernel_size=3)
    rb_blocks = [
      Rrdb(nf, gc, 5, 0.2, kernel_size=3, activation=Activation('lrelu', 0.2))
      for _ in range(nb)]
    LR_conv = EasyConv2d(nf, nf, kernel_size=3)
    upsampler = [Upsample(nf, scale, 'nearest',
                          activation=Activation('lrelu', 0.2))]
    HR_conv0 = EasyConv2d(nf, nf, kernel_size=3, activation='lrelu')
    HR_conv1 = EasyConv2d(nf, channel, kernel_size=3)
    self.body = nn.Sequential(*rb_blocks, LR_conv)
    self.tail = nn.Sequential(*upsampler, HR_conv0, HR_conv1)

  def forward(self, x):
    x = self.head(x)
    x = self.body(x) + x
    x = self.tail(x)
    return x


class ESRGAN(SuperResolution):
  def __init__(self, channel, scale, patch_size=128, weights=(0.01, 1, 5e-3),
               nf=64, nb=23, gc=32, **kwargs):
    super(ESRGAN, self).__init__(scale, channel)
    self.use_vgg = weights[1] > 0
    self.use_gan = weights[2] > 0
    if self.use_gan:
      self.dnet = DCGAN(3, np.log2(patch_size // 4) * 2, 'bn')
      self.optd = torch.optim.Adam(self.trainable_variables('dnet'), 1e-4)
    self.rrdb = RRDB_Net(channel, scale, nf, nb, gc)
    self.optg = torch.optim.Adam(self.trainable_variables('rrdb'), 1e-4)
    if self.use_vgg:
      self.vgg = [VggFeatureLoss(['block5_conv4'], True)]
    # image, vgg, gan
    self.w = weights

  def cuda(self):
    super(ESRGAN, self).cuda()
    if self.use_vgg:
      self.vgg[0].cuda()

  def train(self, inputs, labels, learning_rate=None):
    sr = self.rrdb(inputs[0])
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    image_loss = F.l1_loss(sr, labels[0])
    loss = image_loss * self.w[0]
    if self.use_vgg:
      feature_loss = F.l1_loss(self.vgg[0](sr)[0], self.vgg[0](labels[0])[0])
      loss += feature_loss * self.w[1]
    if self.use_gan:
      # update G
      self.optg.zero_grad()
      fake = self.dnet(sr)
      gan_loss_g = gan_bce_loss(fake, True)
      loss += gan_loss_g * self.w[2]
      loss.backward()
      self.optg.step()
      # update D
      self.optd.zero_grad()
      real = self.dnet(labels[0])
      fake = self.dnet(sr.detach())
      loss_d = gan_bce_loss(real, True) + gan_bce_loss(fake, False)
      loss_d.backward()
      self.optd.step()
      return {
        'loss': loss.detach().cpu().numpy(),
        'image': image_loss.detach().cpu().numpy(),
        'loss_g': gan_loss_g.detach().cpu().numpy(),
        'loss_d': loss_d.detach().cpu().numpy()
      }
    else:
      self.optg.zero_grad()
      loss.backward()
      self.optg.step()
      return {
        'loss': loss.detach().cpu().numpy(),
        'image': image_loss.detach().cpu().numpy()
      }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.rrdb(inputs[0]).cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs.get('epoch')
        writer.image('sr', sr.clamp(0, 1), step=step)
        writer.image('lr', inputs[0], step=step)
        writer.image('hr', labels[0], step=step)
    return [sr.numpy()], metrics

  def export(self, export_dir):
    device = list(self.rrdb.parameters())[0].device
    inputs = torch.randn(1, self.channel, 144, 128, device=device)
    torch.onnx.export(self.rrdb, (inputs,), export_dir / 'rrdb.onnx')
