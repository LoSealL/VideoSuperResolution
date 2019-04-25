#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import numpy as np
import torch
import torch.nn.functional as F

from . import Discriminator as disc
from .Model import SuperResolution
from .esrgan import architecture as arch
from ..Util import Metrics


class ESRGAN(SuperResolution):
  def __init__(self, scale, patch_size=128, use_gan=None, weights=(1, 1),
               **kwargs):
    super(ESRGAN, self).__init__(scale, 3)
    self.rrdb = arch.RRDB_Net(upscale=scale, **kwargs)
    if use_gan:
      self.dnet = disc.DCGAN(3, np.log2(patch_size // 4) * 2, 'bn')
      self.use_gan = True
    self.optg = torch.optim.Adam(self.trainable_variables('rrdb'), 1e-4)
    if use_gan:
      self.optd = torch.optim.Adam(self.trainable_variables('dnet'), 1e-4)
    self.w = weights

  def train(self, inputs, labels, learning_rate=None):
    sr = self.rrdb(inputs[0])
    if learning_rate:
      for param_group in self.optg.param_groups:
        param_group["lr"] = learning_rate
    loss = F.l1_loss(sr, labels[0]) * self.w[0]
    if self.use_gan:
      # update G
      fake = self.dnet(sr)
      loss += F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake)) * \
              self.w[1]
      self.optg.zero_grad()
      loss.backward()
      self.optg.step()
      # update D
      if learning_rate:
        for param_group in self.optd.param_groups:
          param_group["lr"] = learning_rate
      real = self.dnet(labels[0])
      fake = self.dnet(sr.detach())
      loss_d = F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + \
               F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
      self.optd.zero_grad()
      loss_d.backward()
      self.optd.step()
      return {
        'loss_g': loss.detach().cpu().numpy(),
        'loss_d': loss_d.detach().cpu().numpy()
      }
    else:
      self.optg.zero_grad()
      loss.backward()
      self.optg.step()
      return {'loss': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.rrdb(inputs[0]).cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
