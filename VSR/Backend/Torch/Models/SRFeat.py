#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/27 下午11:26

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 15

import logging

import numpy as np
import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d, RB
from .Ops.Scale import Upsample
from .Ops.Discriminator import DCGAN
from .Ops.Loss import VggFeatureLoss, gan_bce_loss
from ..Framework.Summary import get_writer
from ..Util import Metrics

_logger = logging.getLogger("VSR.SRFEAT")
_logger.info("LICENSE: SRFeat is proposed by S. Park, et. al. "
             "Implemented via PyTorch by @LoSealL.")


class Generator(torch.nn.Module):
  """ Generator for SRFeat:
  Single Image Super-Resolution with Feature Discrimination (ECCV 2018)
  """

  def __init__(self, channel, scale, filters, num_rb):
    super(Generator, self).__init__()
    self.head = EasyConv2d(channel, filters, 9)
    for i in range(num_rb):
      setattr(self, f'rb_{i:02d}', RB(filters, 3, 'lrelu', use_bn=True))
      setattr(self, f'merge_{i:02d}', EasyConv2d(filters, filters, 1))
    self.tail = torch.nn.Sequential(
        Upsample(filters, scale), EasyConv2d(filters, channel, 3))
    self.num_rb = num_rb

  def forward(self, inputs):
    x = self.head(inputs)
    feat = []
    for i in range(self.num_rb):
      x = getattr(self, f'rb_{i:02d}')(x)
      feat.append(getattr(self, f'merge_{i:02d}')(x))
    x = self.tail(x + torch.stack(feat, dim=0).sum(0).squeeze(0))
    return x


class SRFEAT(SuperResolution):
  def __init__(self, channel, scale, patch_size=64, weights=(1, 0.01, 0.01),
               **kwargs):
    super(SRFEAT, self).__init__(scale, channel)
    n_rb = kwargs.get('num_residualblocks', 16)
    f = kwargs.get('filters', 64)
    self.use_gan = weights[1] > 0
    self.use_feat_gan = weights[2] > 0
    self.srfeat = Generator(channel, scale, f, n_rb)
    self.gopt = torch.optim.Adam(self.trainable_variables('srfeat'), 1e-4)
    if self.use_gan:
      # vanilla image
      self.dnet1 = DCGAN(channel, np.log2(patch_size // 4) * 2, scale, 'bn')
      self.dopt1 = torch.optim.Adam(self.trainable_variables('dnet1'), 1e-4)
    if self.use_feat_gan:
      # vgg feature
      self.dnet2 = DCGAN(256, np.log2(patch_size // 16) * 2, scale, 'bn')
      self.dopt2 = torch.optim.Adam(self.trainable_variables('dnet2'), 1e-4)
    self.vgg = [VggFeatureLoss(['block3_conv1'], True)]
    self.w = weights

  def cuda(self):
    super(SRFEAT, self).cuda()
    self.vgg[0].cuda()

  def train(self, inputs, labels, learning_rate=None):
    metrics = {}
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
      fake_prob_image = self.dnet1(sr)
      loss_g_image = gan_bce_loss(fake_prob_image, True)
      loss += loss_g_image * self.w[1]
    if self.use_feat_gan:
      fake_prob_feat = self.dnet2(fake_feature)
      loss_g_feat = gan_bce_loss(fake_prob_feat, True)
      loss += loss_g_feat * self.w[2]
    # update G
    self.gopt.zero_grad()
    loss.backward()
    self.gopt.step()
    metrics['vgg'] = loss_p.detach().cpu().numpy()
    metrics['g_loss'] = loss.detach().cpu().numpy()
    if self.use_gan:
      # update D
      real_prob_image = self.dnet1(self.norm(labels[0]))
      fake_prob_image = self.dnet1(sr.detach())
      loss_d_image = gan_bce_loss(real_prob_image, True) + \
                     gan_bce_loss(fake_prob_image, False)
      self.dopt1.zero_grad()
      loss_d_image.backward()
      self.dopt1.step()
      metrics['d_image'] = loss_d_image.detach().cpu().numpy()
    if self.use_feat_gan:
      real_prob_feat = self.dnet2(real_feature.detach())
      fake_prob_feat = self.dnet2(fake_feature.detach())
      loss_d_feat = gan_bce_loss(real_prob_feat, True) + \
                    gan_bce_loss(fake_prob_feat, False)
      self.dopt2.zero_grad()
      loss_d_feat.backward()
      self.dopt2.step()
      metrics['d_feat'] = loss_d_feat.detach().cpu().numpy()
    return metrics

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.srfeat(self.norm(inputs[0])).cpu().detach()
    sr = self.denorm(sr)
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs.get('epoch')
        writer.image('sr', sr, step=step)
    return [sr.numpy()], metrics

  @staticmethod
  def norm(x):
    return x * 2 - 1

  @staticmethod
  def denorm(x):
    return x / 2 + 0.5
