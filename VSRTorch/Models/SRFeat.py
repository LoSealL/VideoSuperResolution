#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/27 下午11:26

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import numpy as np
import torch
import torch.nn.functional as F

from . import Discriminator as disc
from .Model import SuperResolution
from .srfeat import ops
from ..Util import Metrics
from .Loss import VggFeatureLoss


class SRFEAT(SuperResolution):
  def __init__(self, channel, scale, patch_size=64, use_gan=None,
               weights=(1, 1, 1), **kwargs):
    super(SRFEAT, self).__init__(scale, channel)
    n_rb = kwargs.get('num_residualblocks', 16)
    f = kwargs.get('filters', 64)
    self.srfeat = ops.Generator(channel, scale, f, n_rb)
    self.vgg = [VggFeatureLoss(['block3_conv1'], True)]
    if use_gan:
      # vanilla image
      self.dnet1 = disc.DCGAN(channel, np.log2(patch_size // 4) * 2, 'bn')
      # vgg feature
      self.dnet2 = disc.DCGAN(256, np.log2(patch_size // 16) * 2, 'bn')
      self.use_gan = True
    self.gopt = torch.optim.Adam(self.trainable_variables('srfeat'), 1e-4)
    if use_gan:
      self.dopt1 = torch.optim.Adam(self.trainable_variables('dnet1'), 1e-4)
      self.dopt2 = torch.optim.Adam(self.trainable_variables('dnet2'), 1e-4)
    self.w = weights

  def cuda(self):
    super(SRFEAT, self).cuda()
    self.vgg[0].cuda()

  def train(self, inputs, labels, learning_rate=None):
    sr = self.srfeat(self.norm(inputs[0]))
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    fake_feature = self.vgg[0](self.denorm(sr))[0]
    real_feature = self.vgg[0](labels[0])[0]
    loss_p = F.mse_loss(fake_feature, real_feature)
    loss = loss_p * self.w[0]
    if self.use_gan:
      # update G
      fake_prob_image = self.dnet1(sr)
      fake_prob_feat = self.dnet2(fake_feature)
      loss_g_image = F.binary_cross_entropy_with_logits(
        fake_prob_image, torch.ones_like(fake_prob_image))
      loss_g_feat = F.binary_cross_entropy_with_logits(
        fake_prob_feat, torch.ones_like(fake_prob_feat))
      loss += (loss_g_image + loss_g_feat) * self.w[1]
      self.gopt.zero_grad()
      loss.backward()
      self.gopt.step()
      # update D
      real_prob_image = self.dnet1(self.norm(labels[0]))
      real_prob_feat = self.dnet2(real_feature.detach())
      fake_prob_image = self.dnet1(sr.detach())
      fake_prob_feat = self.dnet2(fake_feature.detach())
      loss_d_image = \
        F.binary_cross_entropy_with_logits(
          real_prob_image, torch.ones_like(real_prob_image)) + \
        F.binary_cross_entropy_with_logits(
          fake_prob_image, torch.zeros_like(fake_prob_image))
      loss_d_feat = \
        F.binary_cross_entropy_with_logits(
          real_prob_feat, torch.ones_like(real_prob_feat)) + \
        F.binary_cross_entropy_with_logits(
          fake_prob_feat, torch.zeros_like(fake_prob_feat))
      self.dopt1.zero_grad()
      loss_d_image.backward()
      self.dopt1.step()
      self.dopt2.zero_grad()
      loss_d_feat.backward()
      self.dopt2.step()
      return {
        'g': loss.detach().cpu().numpy(),
        'd_image': loss_d_image.detach().cpu().numpy(),
        'd_feat': loss_d_feat.detach().cpu().numpy(),
      }
    else:
      self.gopt.zero_grad()
      loss.backward()
      self.gopt.step()
      return {'loss': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.srfeat(self.norm(inputs[0])).cpu().detach()
    sr = self.denorm(sr)
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics

  @staticmethod
  def norm(x):
    return x * 2 - 1

  @staticmethod
  def denorm(x):
    return x / 2 + 0.5
