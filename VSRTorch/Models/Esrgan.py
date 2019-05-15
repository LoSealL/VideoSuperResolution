#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import numpy as np
import torch
import torch.nn.functional as F

from . import Discriminator as disc
from .Model import SuperResolution
from .Loss import gan_bce_loss, VggFeatureLoss
from .esrgan import architecture as arch
from ..Util import Metrics
from ..Framework.Summary import get_writer


class ESRGAN(SuperResolution):
  def __init__(self, scale, patch_size=128, weights=(0.01, 1, 5e-3), **kwargs):
    super(ESRGAN, self).__init__(scale, 3)
    self.use_vgg = weights[1] > 0
    self.use_gan = weights[2] > 0
    if self.use_gan:
      self.dnet = disc.DCGAN(3, np.log2(patch_size // 4) * 2, 'bn')
      self.optd = torch.optim.Adam(self.trainable_variables('dnet'), 1e-4)
    self.rrdb = arch.RRDB_Net(upscale=scale, **kwargs)
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
