#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/7 下午5:21

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .Arch import SpaceToDepth
from .Loss import VggFeatureLoss, gan_bce_loss
from .Model import SuperResolution
from .frvsr.ops import FNet
from .teco.ops import TecoDiscriminator, TecoGenerator
from .video.motion import STN
from ..Framework.Summary import get_writer
from ..Util import Metrics
from ..Util.Utility import pad_if_divide


class Composer(nn.Module):
  def __init__(self, scale, channel, gain=24, filters=64, n_rb=16):
    super(Composer, self).__init__()
    self.fnet = FNet(channel, gain=gain)
    self.gnet = TecoGenerator(channel, scale, filters, n_rb)
    self.warpper = STN(padding_mode='border')
    self.spd = SpaceToDepth(scale)
    self.scale = scale

  def forward(self, lr, lr_pre, sr_pre):
    """
    Args:
       lr: t_1 lr frame
       lr_pre: t_0 lr frame
       sr_pre: t_0 sr frame
    """
    flow = self.fnet(lr, lr_pre)
    flow_up = self.scale * F.interpolate(
      flow, scale_factor=self.scale, mode='bicubic', align_corners=False)
    u, v = [x.squeeze(1) for x in flow_up.split(1, dim=1)]
    sr_warp = self.warpper(sr_pre, u, v, False)
    bi = F.interpolate(
      lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
    sr = self.gnet(lr, self.spd(sr_warp), bi)
    return sr, sr_warp, flow, flow_up, bi


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
    frames = [self.norm(x.squeeze(1)) for x in inputs[0].split(1, dim=1)]
    labels = [self.norm(x.squeeze(1)) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    # For ping-pong loss
    frames_rev = frames.copy()
    frames_rev.reverse()
    last_lr = frames_rev[0]
    last_sr = F.interpolate(
      last_lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
    back_sr = []
    for lr in frames_rev:
      sr_rev, _, _, _, _ = self.gnet(lr, last_lr, last_sr)
      # TODO detach or not?
      back_sr.append(sr_rev)
      last_lr = lr.detach()
      last_sr = sr_rev.detach()
    back_sr.reverse()
    total_loss = 0
    pp_loss = 0
    # Generator graph
    last_lr = frames[0]
    last_sr = F.interpolate(
      last_lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
    forward_sr = []
    bicubic_lr = []
    for lr, hr, bk in zip(frames, labels, back_sr):
      sr, sr_warp, flow, _, bi = self.gnet(lr, last_lr, last_sr)
      lr_warp = self.gnet.warpper(last_lr, flow[:, 0], flow[:, 1], False)
      l2_image = F.mse_loss(sr, hr)
      l2_warp = F.mse_loss(lr_warp, lr)
      l2_pingpong = F.mse_loss(sr, bk)
      total_loss += l2_image * self.w[0] + l2_warp * self.w[1] + \
                    l2_pingpong * self.w[2]
      pp_loss += l2_pingpong.detach()
      metrics['image'] = l2_image.detach().cpu().numpy()
      metrics['flow'] = l2_warp.detach().cpu().numpy()
      last_lr = lr  # .detach()
      last_sr = sr  # .detach()
      forward_sr.append(sr)
      bicubic_lr.append(bi.detach())
      if self.use_vgg:
        real_feature = self.vgg[0](hr)
        fake_feature = self.vgg[0](sr)
        l2_vgg = 0
        for x, y, w in zip(real_feature, fake_feature, self.vgg_weights):
          l2_vgg += F.mse_loss(x, y) * w
        metrics['vgg'] = l2_vgg.detach().cpu().numpy()
        total_loss += l2_vgg
    metrics['pp'] = pp_loss.detach().cpu().numpy() / len(frames)
    if self.use_gan:
      # Discriminator graph
      disc_loss = 0
      for i in range(1, len(forward_sr) - 1):
        bi_p, bi_c, bi_n = bicubic_lr[i - 1:i + 2]
        lr_p, lr_c, lr_n = frames[i - 1:i + 2]
        hr_p, hr_c, hr_n = labels[i - 1:i + 2]
        sr_p, sr_c, sr_n = forward_sr[i - 1:i + 2]
        flow_forward = self.scale * F.interpolate(
          self.gnet.fnet(lr_c, lr_p), scale_factor=self.scale,
          mode='bicubic', align_corners=False)
        flow_backward = self.scale * F.interpolate(
          self.gnet.fnet(lr_c, lr_n), scale_factor=self.scale,
          mode='bicubic', align_corners=False)
        hr_w1 = self.gnet.warpper(
          hr_p, flow_forward[:, 0], flow_forward[:, 1], False)
        hr_w2 = self.gnet.warpper(
          hr_n, flow_backward[:, 0], flow_backward[:, 1], False)
        sr_w1 = self.gnet.warpper(
          sr_p, flow_forward[:, 0], flow_forward[:, 1], False)
        sr_w2 = self.gnet.warpper(
          sr_n, flow_backward[:, 0], flow_backward[:, 1], False)
        # Stop BP to Fnet
        d_input_fake = torch.cat(
          (bi_p, bi_c, bi_n, sr_p, sr_c, sr_n, sr_w1.detach(), sr_w2.detach()),
          dim=1)
        d_input_real = torch.cat(
          (bi_p, bi_c, bi_n, hr_p, hr_c, hr_n, hr_w1.detach(), hr_w2.detach()),
          dim=1)
        # Padding border pixels to zero
        d_input_fake = self.shave_border_pixel(d_input_fake, 16)
        d_input_real = self.shave_border_pixel(d_input_real, 16)
        # BP to generator
        fake, fake_d_feature = self.dnet(d_input_fake)
        real, real_d_feature = self.dnet(d_input_real)
        loss_g = gan_bce_loss(fake, True)
        # l2_d_feature = 0
        # for x, y, w in zip(fake_d_feature, real_d_feature, self.gan_weights):
        #   l2_d_feature += F.mse_loss(x, y) * w
        total_loss += loss_g * self.w[3]
        # metrics['dfeat'] = l2_d_feature.detach().cpu().numpy()
        # Now avoid BP to generator
        fake, _ = self.dnet(d_input_fake.detach())
        disc_loss += gan_bce_loss(real, True) + gan_bce_loss(fake, False)
      metrics['dloss'] = disc_loss.detach().cpu().numpy() / (len(frames) - 2)
      self.dopt.zero_grad()
      disc_loss.backward(retain_graph=True)
      self.dopt.step()
    self.gopt.zero_grad()
    total_loss.backward()
    self.gopt.step()
    metrics['loss'] = total_loss.detach().cpu().numpy() / len(frames)
    return metrics

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [self.norm(x.squeeze(1)) for x in inputs[0].split(1, dim=1)]
    predicts = []
    last_lr = pad_if_divide(frames[0], 8, 'reflect')
    a = (last_lr.size(2) - frames[0].size(2)) * self.scale
    b = (last_lr.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    last_sr = F.interpolate(
      last_lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
    for lr in frames:
      lr = pad_if_divide(lr, 8, 'reflect')
      sr, _, _, _, _ = self.gnet(lr, last_lr, last_sr)
      last_lr = lr.detach()
      last_sr = sr.detach()
      sr = sr[..., slice_h, slice_w]
      predicts.append(self.denorm(sr).cpu().detach().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      psnr = [Metrics.psnr(x, y) for x, y in zip(predicts, labels)]
      metrics['psnr'] = np.mean(psnr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('sr', self.denorm(sr).clamp(0, 1), step=step)
    return predicts, metrics

  @staticmethod
  def norm(x):
    return x * 2.0 - 1

  @staticmethod
  def denorm(x):
    return x / 2.0 + 0.5
