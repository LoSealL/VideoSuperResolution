#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/7 下午5:21

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .Arch import SpaceToDepth
from .Loss import VggFeatureLoss, gan_bce_loss, ragan_bce_loss
from .Model import SuperResolution
from .frvsr.ops import FNet
from .teco.ops import TecoDiscriminator, TecoGenerator
from .video.motion import STN
from ..Framework.Summary import get_writer
from ..Util import Metrics
from ..Util.Utility import pad_if_divide, upsample


class Composer(nn.Module):
  def __init__(self, scale, channel, gain=24, filters=64, n_rb=16):
    super(Composer, self).__init__()
    self.fnet = FNet(channel, gain=gain)
    self.gnet = TecoGenerator(channel, scale, filters, n_rb)
    self.warpper = STN(padding_mode='border')
    self.spd = SpaceToDepth(scale)
    self.scale = scale

  def forward(self, lr, lr_pre, sr_pre, detach_fnet=None):
    """
    Args:
       lr: t_1 lr frame
       lr_pre: t_0 lr frame
       sr_pre: t_0 sr frame
       detach_fnet: detach BP to fnet
    """
    flow = self.fnet(lr, lr_pre)
    flow_up = self.scale * upsample(flow, self.scale)
    u, v = [x.squeeze(1) for x in flow_up.split(1, dim=1)]
    sr_warp = self.warpper(sr_pre, u, v, False)
    bi = upsample(lr, self.scale)
    if detach_fnet:
      sr = self.gnet(lr, self.spd(sr_warp.detach()), bi)
    else:
      sr = self.gnet(lr, self.spd(sr_warp), bi)
    return sr, sr_warp, flow, flow_up


class TeCoGAN(SuperResolution):
  """Temporally Coherent GANs for Video Super-Resolution.

  WARNING: Training now is experimental.

  Args:
    scale: scale factor
    channel: input channel number
    weights: a list of 4 integers representing weights of
      [Image, Flow, Ping-Pong, GAN]
    vgg_layers: a list of string representing VGG layers to extract
    vgg_layer_weights: a list of integers the same number as `vgg_layers`,
      weights for each VGG layers.
    gan_layer_weights: a list of 4 integers representing weights for each layer
      in the discriminator.
  """

  def __init__(self, scale, channel, weights, vgg_layers, vgg_layer_weights,
               gan_layer_weights, patch_size, **kwargs):
    super(TeCoGAN, self).__init__(scale, channel, **kwargs)
    filters = kwargs.get('filters', 64)  # default filter number
    gain = kwargs.get('max_displacement', 24)  # max movement of optical flow
    n_rb = kwargs.get('num_residualblocks', 16)  # default RB numbers
    self.use_vgg = vgg_layers != []
    self.use_gan = weights[3] > 0
    self.debug = kwargs.get('debug', {})
    self.gnet = Composer(scale, channel, gain, filters, n_rb)
    self.gopt = torch.optim.Adam(self.trainable_variables('gnet'), 5e-5)
    if self.use_vgg:
      # put into list to avoid saving VGG weights
      self.vgg = [VggFeatureLoss(vgg_layers, True)]
      self.vgg_weights = vgg_layer_weights
    if self.use_gan:
      self.dnet = TecoDiscriminator(channel, filters, patch_size)
      self.dopt = torch.optim.Adam(self.trainable_variables('dnet'), 5e-5)
      self.gan_weights = gan_layer_weights
    self.w = weights  # [L2, flow, ping-pong, gan]

  def cuda(self):
    super(TeCoGAN, self).cuda()
    if self.use_vgg:
      self.vgg[0].cuda()

  @staticmethod
  def shave_border_pixel(x, border=16):
    x = x[..., border:-border, border:-border]
    return F.pad(x, [border, border, border, border])

  def train(self, inputs, labels, learning_rate=None):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    # For ping-pong loss
    frames_rev = frames.copy()
    frames_rev.reverse()
    last_lr = frames_rev[0]
    last_sr = upsample(last_lr, self.scale)
    back_sr = []
    for lr in frames_rev:
      sr_rev, _, _, _ = self.gnet(lr, last_lr, last_sr)
      back_sr.append(sr_rev.detach())
      last_lr = lr
      last_sr = sr_rev
    back_sr.reverse()
    total_loss = 0
    pp_loss = 0
    # Generator graph
    last_lr = frames[0]
    last_lr2 = last_lr
    last_sr = upsample(last_lr, self.scale)
    last_sr2 = last_sr
    last_hr = labels[0]
    last_hr2 = last_hr
    for i in range(len(frames)):
      self.gopt.zero_grad()
      lr, hr, bk = frames[i], labels[i], back_sr[i]
      sr, sr_warp, flow, _ = self.gnet(lr, last_lr, last_sr.detach())
      lr_warp = self.gnet.warpper(last_lr, flow[:, 0], flow[:, 1], False)
      l2_image = F.mse_loss(sr, hr)
      l2_warp = F.mse_loss(lr_warp, lr)
      l2_pp = F.mse_loss(sr, bk)
      loss = l2_image * self.w[0] + l2_warp * self.w[1] + l2_pp * self.w[2]
      pp_loss += l2_pp.detach()
      metrics['image'] = l2_image.detach().cpu().numpy()
      metrics['flow'] = l2_warp.detach().cpu().numpy()
      if self.use_vgg:
        real_feature = self.vgg[0](hr)
        fake_feature = self.vgg[0](sr)
        l2_vgg = 0
        for x, y, w in zip(real_feature, fake_feature, self.vgg_weights):
          l2_vgg += F.mse_loss(x, y) * w
        metrics['vgg'] = l2_vgg.detach().cpu().numpy()
        loss += l2_vgg
      if self.use_gan:
        # last_bi2 = upsample(last_lr2, self.scale)
        # last_bi = upsample(last_lr, self.scale)
        # bi = upsample(lr, self.scale)
        flow20 = self.scale * upsample(self.gnet.fnet(lr, last_lr2), self.scale)
        flow10 = self.scale * upsample(self.gnet.fnet(lr, last_lr), self.scale)
        hr20 = self.gnet.warpper(last_hr2, flow20[:, 0], flow20[:, 1], False)
        hr10 = self.gnet.warpper(last_hr, flow10[:, 0], flow10[:, 1], False)
        sr20 = self.gnet.warpper(last_sr2, flow20[:, 0], flow20[:, 1], False)
        sr10 = self.gnet.warpper(last_sr, flow10[:, 0], flow10[:, 1], False)
        d_input_fake = torch.cat(
          (  # last_bi2, last_bi, bi,
            last_sr2, last_sr, sr, sr20, sr10), dim=1)
        d_input_real = torch.cat(
          (  # last_bi2, last_bi, bi,
            last_hr2, last_hr, hr, hr20, hr10), dim=1)
        # Padding border pixels to zero
        # d_input_fake = self.shave_border_pixel(d_input_fake, 16)
        # d_input_real = self.shave_border_pixel(d_input_real, 16)
        fake, fake_d_feature = self.dnet(self.norm(d_input_fake))
        real, real_d_feature = self.dnet(self.norm(d_input_real).detach())
        l2_d_feature = 0
        for x, y, w in zip(fake_d_feature, real_d_feature, self.gan_weights):
          l2_d_feature += F.mse_loss(x, y) * w
        gloss = ragan_bce_loss(fake, real) * self.w[3]
        gloss.backward(retain_graph=True)
        self.gnet.fnet.zero_grad()
        loss += l2_d_feature
        metrics['dfeat'] = l2_d_feature.detach().cpu().numpy()
        metrics['g'] = gloss.detach().cpu().numpy()
      loss.backward()
      self.gopt.step()
      total_loss += loss.detach()
      if self.use_gan:
        self.dopt.zero_grad()
        fake, _ = self.dnet(self.norm(d_input_fake).detach())
        real, _ = self.dnet(self.norm(d_input_real).detach())
        dloss = ragan_bce_loss(real, fake)
        dloss.backward()
        self.dopt.step()
        metrics['d'] = dloss.detach().cpu().numpy()
      last_lr2 = last_lr.detach()
      last_sr2 = last_sr.detach()
      last_hr2 = last_hr.detach()
      last_lr = lr.detach()
      last_sr = sr.detach()
      last_hr = hr.detach()
    metrics['pp'] = pp_loss.detach().cpu().numpy() / len(frames)
    metrics['loss'] = total_loss.detach().cpu().numpy() / len(frames)
    return metrics

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    predicts = []
    last_lr = pad_if_divide(frames[0], 8, 'reflect')
    a = (last_lr.size(2) - frames[0].size(2)) * self.scale
    b = (last_lr.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    last_sr = upsample(last_lr, self.scale)
    for lr in frames:
      lr = pad_if_divide(lr, 8, 'reflect')
      sr, warp, _, _ = self.gnet(lr, last_lr, last_sr)
      last_lr = lr.detach()
      last_sr = sr.detach()
      sr = sr[..., slice_h, slice_w]
      warp = warp[..., slice_h, slice_w]
      predicts.append(sr.cpu().detach().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      psnr = [Metrics.psnr(x, y) for x, y in zip(predicts, labels)]
      metrics['psnr'] = np.mean(psnr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('hr', labels[-1], step=step)
        writer.image('sr', sr.clamp(0, 1), step=step)
        writer.image('warp', warp.clamp(0, 1), step=step)
    return predicts, metrics

  @staticmethod
  def norm(x):
    return x * 2.0 - 1

  @staticmethod
  def denorm(x):
    return x / 2.0 + 0.5
