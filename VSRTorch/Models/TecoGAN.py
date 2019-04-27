#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/27 下午3:15

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .Arch import SpaceToDepth
from .Classic import Scale
from .Model import SuperResolution
from .frvsr.ops import FNet
from .teco.ops import TecoDiscriminator, TecoGenerator
from .video.motion import STN
from .Loss import VggFeatureLoss
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
    self.bicubic = torchvision.transforms.Compose([
      torchvision.transforms.ToPILImage(),
      Scale(scale),
      torchvision.transforms.ToTensor()])
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
      flow, scale_factor=self.scale, mode='bilinear', align_corners=False)
    u, v = [x.squeeze(1) for x in flow_up.split(1, dim=1)]
    sr_warp = self.warpper(sr_pre, u, v, False)
    device = lr.device
    bi = [self.bicubic(img) for img in lr.cpu()]
    bi = torch.stack(bi, dim=0).to(device)
    sr = self.gnet(lr, self.spd(sr_warp), bi)
    return sr, sr_warp, flow, flow_up, bi


class TeCoGAN(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(TeCoGAN, self).__init__(scale, channel, **kwargs)
    filters = kwargs.get('filters', 64)
    gain = kwargs.get('max_displacement', 24)
    n_rb = kwargs.get('num_residualblocks', 16)
    self.debug = kwargs.get('debug', {})
    self.gnet = Composer(scale, channel, gain, filters, n_rb)
    self.dnet = TecoDiscriminator()
    self.vgg = [VggFeatureLoss(['block5_conv4'], True)]
    self.gopt = torch.optim.Adam(self.trainable_variables('gnet'), 1e-4)
    self.dopt = torch.optim.Adam(self.trainable_variables('dnet'), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    # For ping-pong loss
    frames_rev = frames.copy()
    labels_rev = labels.copy()
    last_lr = frames_rev[0]
    last_sr = torch.zeros_like(labels_rev[0])
    back_sr = []
    for lr, sr in zip(frames_rev, labels_rev):
      sr_rev, _, _, _ = self.gnet(lr, sr, last_lr, last_sr)
      # TODO detach or not?
      back_sr.append(sr_rev.detach())
    total_loss = 0
    last_lr = frames[0]
    last_sr = F.interpolate(
      frames[0], scale_factor=self.scale, mode='bilinear', align_corners=False)
    forward_sr = []
    bicubic_lr = []
    for lr, hr, bk in zip(frames, labels, back_sr):
      sr, sr_warp, flow, _, bi = self.gnet(lr, last_lr, last_sr)
      last_lr = lr.detach()
      last_sr = sr.detach()
      forward_sr.append(sr.detach())
      bicubic_lr.append(bi.detach())
      lr_warp = self.gnet.warpper(last_lr, flow[:, 0], flow[:, 1], False)
      real_feature = self.vgg[0](hr)[0]
      fake_feature = self.vgg[0](sr)[0]
      l2_image = F.mse_loss(sr, hr)
      l2_warp = F.mse_loss(lr_warp, lr)
      l2_vgg = F.mse_loss(fake_feature, real_feature)
      l2_pingpong = F.mse_loss(sr, bk)
      total_loss += l2_image + l2_warp + l2_vgg + l2_pingpong
    disc_loss = 0
    for i in range(1, len(forward_sr) - 1):
      bi_p, bi_c, bi_n = bicubic_lr[i - 1:i + 2]
      lr_p, lr_c, lr_n = frames[i - 1:i + 2]
      hr_p, hr_c, hr_n = labels[i - 1:i + 2]
      sr_p, sr_c, sr_n = forward_sr[i - 1:i + 2]
      flow_forward = self.scale * F.interpolate(
        self.gnet.fnet(lr_c, lr_p), scale_factor=self.scale,
        mode='bilinear', align_corners=False)
      flow_backward = self.scale * F.interpolate(
        self.gnet.fnet(lr_c, lr_n), scale_factor=self.scale,
        mode='bilinear', align_corners=False)
      hr_w1 = self.gnet.warpper(
        hr_p, flow_forward[:, 0], flow_forward[:, 1], False)
      hr_w2 = self.gnet.warpper(
        hr_n, flow_backward[:, 0], flow_backward[:, 1], False)
      sr_w1 = self.gnet.warpper(
        sr_p, flow_forward[:, 0], flow_forward[:, 1], False)
      sr_w2 = self.gnet.warpper(
        sr_n, flow_backward[:, 0], flow_backward[:, 1], False)
      d_input_fake = torch.cat(
        (bi_p, bi_c, bi_n, sr_p, sr_c, sr_n, sr_w1, sr_w2), dim=1)
      d_input_real = torch.cat(
        (bi_p, bi_c, bi_n, hr_p, hr_c, hr_n, hr_w1, hr_w2), dim=1)
      fake_prob = self.dnet(d_input_fake)
      real_prob = self.dnet(d_input_real)
      disc_loss += F.binary_cross_entropy_with_logits(
        real_prob, torch.ones_like(real_prob))
      disc_loss += F.binary_cross_entropy_with_logits(
        fake_prob, torch.zeros_like(fake_prob))
      loss_g = F.binary_cross_entropy_with_logits(
        fake_prob, torch.ones_like(fake_prob))
      total_loss += loss_g
    self.gopt.zero_grad()
    total_loss.backward()
    self.gopt.step()
    self.dopt.zero_grad()
    disc_loss.backward()
    self.dopt.step()
    return {
      'total_loss': total_loss.detach().cpu().numpy() / len(frames),
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    predicts = []
    last_lr = pad_if_divide(frames[0], 8, 'reflect')
    a = (last_lr.size(2) - frames[0].size(2)) * self.scale
    b = (last_lr.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    last_sr = F.interpolate(
      frames[0], scale_factor=self.scale, mode='bilinear', align_corners=False)
    for lr in frames:
      sr, _, _, _ = self.gnet(lr, last_lr, last_sr)
      last_lr = lr.detach()
      last_sr = sr.detach()
      sr = sr[..., slice_h, slice_w]
      predicts.append(sr.cpu().detach().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      psnr = [Metrics.psnr(x, y) for x, y in zip(predicts, labels)]
      metrics['psnr'] = np.mean(psnr)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('sr', sr, step=step)
        # writer.image('warp', lqw, step=step)
    return predicts, metrics
