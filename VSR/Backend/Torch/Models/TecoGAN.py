#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/7 下午5:21

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d, RB
from .Ops.Loss import VggFeatureLoss, ragan_bce_loss
from .Ops.Motion import Flownet, STN
from .Ops.Scale import SpaceToDepth, Upsample
from ..Framework.Summary import get_writer
from ..Util import Metrics
from ..Util.Utility import pad_if_divide, upsample

_logger = logging.getLogger("VSR.TecoGAN")
_logger.info("LICENSE: TecoGAN is implemented by Mengyu Chu, et. al. "
             "@rachelchu https://github.com/rachelchu/TecoGAN")
_logger.warning("Training of TecoGAN hasn't been verified!!")


class TecoGenerator(nn.Module):
  """Generator in TecoGAN.

  Note: the flow estimation net `Fnet` shares with FRVSR.

  Args:
    filters: basic filter numbers [default: 64]
    num_rb: number of residual blocks [default: 16]
  """

  def __init__(self, channel, scale, filters, num_rb):
    super(TecoGenerator, self).__init__()
    rbs = []
    for i in range(num_rb):
      rbs.append(RB(filters, filters, 3, 'relu'))
    self.body = nn.Sequential(
        EasyConv2d(channel * (1 + scale ** 2), filters, 3, activation='relu'),
        *rbs,
        Upsample(filters, scale, 'nearest', activation='relu'),
        EasyConv2d(filters, channel, 3))

  def forward(self, x, prev, residual=None):
    """`residual` is the bicubically upsampled HR images"""
    sr = self.body(torch.cat((x, prev), dim=1))
    if residual is not None:
      sr += residual
    return sr


class TecoDiscriminator(nn.Module):
  def __init__(self, channel, filters, patch_size):
    super(TecoDiscriminator, self).__init__()
    f = filters
    self.conv0 = EasyConv2d(channel * 6, f, 3, activation='leaky')
    self.conv1 = EasyConv2d(f, f, 4, 2, activation='leaky', use_bn=True)
    self.conv2 = EasyConv2d(f, f, 4, 2, activation='leaky', use_bn=True)
    self.conv3 = EasyConv2d(f, f * 2, 4, 2, activation='leaky', use_bn=True)
    self.conv4 = EasyConv2d(f * 2, f * 4, 4, 2, activation='leaky', use_bn=True)
    # self.pool = nn.AdaptiveAvgPool2d(1)
    self.linear = nn.Linear(f * 4 * (patch_size // 16) ** 2, 1)

  def forward(self, x):
    """The inputs `x` is the concat of 8 tensors.
      Note that we remove the duplicated gt/yt in paper (9 - 1 = 8).
    """
    l0 = self.conv0(x)
    l1 = self.conv1(l0)
    l2 = self.conv2(l1)
    l3 = self.conv3(l2)
    l4 = self.conv4(l3)
    # y = self.pool(l4)
    y = self.linear(l4.flatten(1))
    return y, (l1, l2, l3, l4)


class Composer(nn.Module):
  def __init__(self, scale, channel, gain=24, filters=64, n_rb=16):
    super(Composer, self).__init__()
    self.fnet = Flownet(channel)
    self.gnet = TecoGenerator(channel, scale, filters, n_rb)
    self.warpper = STN(padding_mode='border')
    self.spd = SpaceToDepth(scale)
    self.scale = scale
    self.gain = gain

  def forward(self, lr, lr_pre, sr_pre, detach_fnet=None):
    """
    Args:
       lr: t_1 lr frame
       lr_pre: t_0 lr frame
       sr_pre: t_0 sr frame
       detach_fnet: detach BP to fnet
    """
    flow = self.fnet(lr, lr_pre, gain=self.gain)
    flow_up = self.scale * upsample(flow, self.scale)
    u, v = [x.squeeze(1) for x in flow_up.split(1, dim=1)]
    sr_warp = self.warpper(sr_pre, u, v)
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
    self.weights = weights  # [L2, flow, ping-pong, gan]

  def cuda(self):
    super(TeCoGAN, self).cuda()
    if self.use_vgg:
      self.vgg[0].cuda()

  @staticmethod
  def shave_border_pixel(x, border=16):
    x = x[..., border:-border, border:-border]
    return F.pad(x, [border, border, border, border])

  def gen_sr_clips(self, frames):
    """generate a video clip"""
    last_lr = frames[0]
    sr = [upsample(last_lr, self.scale)]
    flow = []
    for lr in frames:
      _sr, _, _f, _ = self.gnet(lr, last_lr, sr[-1].detach())
      sr.append(_sr)
      flow.append(_f)
    return sr[1:], flow

  def prepare_dnet_input(self, frames):
    # For now inputs don't take LR into account
    cube = []
    for i in range(1, len(frames) - 1):
      pre, cur, nex = frames[i - 1:i + 2]
      with torch.no_grad():
        ff = self.gnet.fnet(cur, pre)
        bf = self.gnet.fnet(cur, nex)
        warpf = self.gnet.warpper(pre, ff[:, 0], ff[:, 1])
        warpb = self.gnet.warpper(nex, bf[:, 0], bf[:, 1])
        cube.append(torch.cat((pre, cur, nex, warpf, cur, warpb), dim=1))
    return cube

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
    back_sr, _ = self.gen_sr_clips(frames_rev)
    back_sr.reverse()
    sr, ff = self.gen_sr_clips(frames)
    # Generator loss
    # 1. Image MSE
    loss_image_mse = torch.stack([F.mse_loss(x, y) for x, y in zip(sr, labels)])
    # 2. Ping-pong loss
    loss_pp = torch.stack([F.mse_loss(x, y) for x, y in zip(sr, back_sr)])
    # 3. FlowNet loss
    loss_flow = []
    for i in range(len(sr)):
      last_lr = frames[i] if i == 0 else frames[i - 1]
      flow = ff[i]
      lr_warp = self.gnet.warpper(last_lr, flow[:, 0], flow[:, 1])
      loss_flow.append(F.mse_loss(frames[i], lr_warp))
    loss_flow = torch.stack(loss_flow)
    w = self.weights
    loss_image_mse = loss_image_mse.mean() * w[0]
    loss_flow = loss_flow.mean() * w[1]
    loss_pp = loss_pp.mean() * w[2]
    loss = loss_image_mse + loss_flow + loss_pp
    # recording
    metrics['image'] = loss_image_mse.detach().cpu().numpy()
    metrics['flow'] = loss_flow.detach().cpu().numpy()
    metrics['pp'] = loss_pp.detach().cpu().numpy()
    # 4. Vgg feature loss
    if self.use_vgg:
      loss_vgg = []
      for x, y in zip(sr, labels):
        real_feature = self.vgg[0](y)
        fake_feature = self.vgg[0](x)
        loss_vgg += [F.mse_loss(x, y) * w for x, y, w in
                     zip(real_feature, fake_feature, self.vgg_weights)]
      loss_vgg = torch.stack(loss_vgg).mean()
      loss += loss_vgg
      metrics['vgg'] = loss_vgg.detach().cpu().numpy()
    # 5. GAN loss g
    if self.use_gan:
      fake_inputs = self.prepare_dnet_input(sr)
      real_inputs = self.prepare_dnet_input(labels)
      loss_d_feature = []
      loss_g = []
      for x, y in zip(fake_inputs, real_inputs):
        fake, fake_d_feats = self.dnet(self.norm(x))
        real, real_d_feats = self.dnet(self.norm(y.detach()))
        loss_d_feature += [F.mse_loss(x, y) * w for x, y, w in
                           zip(real_d_feats, fake_d_feats, self.gan_weights)]
        loss_g += [ragan_bce_loss(fake, real)]
      loss_d_feature = torch.stack(loss_d_feature).mean()
      loss_g = torch.stack(loss_g).mean() * w[3]
      loss += loss_d_feature
      metrics['df'] = loss_d_feature.detach().cpu().numpy()
      metrics['g'] = loss_g.detach().cpu().numpy()
    # Discriminator loss
    if self.use_gan:
      loss_d = []
      for x, y in zip(fake_inputs, real_inputs):
        fake, _ = self.dnet(self.norm(x.detach()))
        real, _ = self.dnet(self.norm(y.detach()))
        loss_d.append(ragan_bce_loss(real, fake))
      loss_d = torch.stack(loss_d).mean()
      metrics['d'] = loss_d.detach().cpu().numpy()
    metrics['total'] = loss.detach().cpu().numpy()
    # Optimize
    self.gopt.zero_grad()
    if self.use_gan:
      loss_g.backward(retain_graph=True)
      self.gnet.fnet.zero_grad()
    loss.backward()
    self.gopt.step()
    if self.use_gan:
      self.dopt.zero_grad()
      loss_d.backward()
      self.dopt.step()
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
