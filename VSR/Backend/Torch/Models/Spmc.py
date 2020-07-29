#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 下午12:49

import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from .Model import SuperResolution
from .Ops.Blocks import Conv2dLSTMCell, EasyConv2d
from .Ops.Loss import total_variance
from .Ops.Motion import CoarseFineFlownet, STN
from ..Framework.Summary import get_writer
from ..Util.Metrics import psnr
from ..Util.Utility import pad_if_divide, upsample

_logger = logging.getLogger("VSR.SPMC")
_logger.info("LICENSE: SPMC is proposed by X. Tao, et. al. "
             "Implemented via PyTorch by @LoSealL.")
_logger.info("LICENSE: ConvLSTM is implemented by @Kaixhin.")


class ZeroUpsample(nn.Module):
  def __init__(self, scale_factor):
    super(ZeroUpsample, self).__init__()
    self.ps = nn.PixelShuffle(scale_factor)
    self.scale = scale_factor

  def forward(self, x):
    z = torch.zeros_like(x).repeat_interleave(self.scale ** 2 - 1, dim=1)
    x = torch.cat((x, z), dim=1)
    return self.ps(x)


class SubPixelMotionCompensation(nn.Module):
  def __init__(self, scale):
    super(SubPixelMotionCompensation, self).__init__()
    self.zero_up = ZeroUpsample(scale)
    self.warper = STN()
    self.scale = scale

  def forward(self, x, u=0, v=0, flow=None):
    if flow is not None:
      u = flow[:, 0]
      v = flow[:, 1]
    x2 = self.zero_up(x)
    u2 = self.zero_up(u.unsqueeze(1)) * self.scale
    v2 = self.zero_up(v.unsqueeze(1)) * self.scale
    return self.warper(x2, u2.squeeze(1), v2.squeeze(1))


class MotionEstimation(nn.Module):
  def __init__(self, channel, gain=32):
    super(MotionEstimation, self).__init__()
    self.gain = gain
    self.flownet = CoarseFineFlownet(channel)

  def forward(self, target, ref, to_tuple=None):
    flow = self.flownet(target, ref, self.gain)
    if to_tuple:
      return flow[:, 0], flow[:, 1]
    return flow


class DetailFusion(nn.Module):
  def __init__(self, channel, base_filter):
    super(DetailFusion, self).__init__()
    f = base_filter
    self.enc1 = EasyConv2d(channel, f, 5, activation='relu')
    self.enc2 = nn.Sequential(
        EasyConv2d(f, f * 2, 3, 2, activation='relu'),
        EasyConv2d(f * 2, f * 2, 3, activation='relu'))
    self.enc3 = EasyConv2d(f * 2, f * 4, 3, 2, activation='relu')
    self.lstm = Conv2dLSTMCell(f * 4, f * 4, 3, 1, 1)
    self.dec1 = nn.Sequential(
        EasyConv2d(f * 4, f * 4, 3, activation='relu'),
        nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1),
        nn.ReLU(True))
    self.dec2 = nn.Sequential(
        EasyConv2d(f * 2, f * 2, 3, activation='relu'),
        nn.ConvTranspose2d(f * 2, f, 4, 2, 1),
        nn.ReLU(True))
    self.dec3 = nn.Sequential(
        EasyConv2d(f, f, 3, activation='relu'),
        EasyConv2d(f, channel, 5))

  def forward(self, x, hx):
    add1 = self.enc1(x)
    add2 = self.enc2(add1)
    h0 = self.enc3(add2)
    x, hx = self.lstm(h0, hx)
    x = self.dec1(x)
    x = self.dec2(x + add2)
    x = self.dec3(x + add1)
    return x, hx


class DetailRevealer(nn.Module):
  def __init__(self, scale, channel, **kwargs):
    super(DetailRevealer, self).__init__()
    self.base_filter = kwargs.get('base_filter', 32)
    self.me = MotionEstimation(channel, gain=kwargs.get('gain', 32))
    self.spmc = SubPixelMotionCompensation(scale)
    self.vsr = DetailFusion(channel, self.base_filter)
    self.scale = scale
    self.hidden_state = None

  def reset(self):
    self.hidden_state = None

  def forward(self, target, ref):
    flow = self.me(target, ref)
    hr_ref = self.spmc(ref, flow=flow)
    hr_target = upsample(target, self.scale)
    if self.hidden_state is None:
      batch, _, height, width = hr_ref.shape
      hidden_shape = (batch, self.base_filter * 4, height // 4, width // 4)
      hx = (torch.zeros(hidden_shape, device=ref.device),
            torch.zeros(hidden_shape, device=ref.device))
    else:
      hx = self.hidden_state
    res, hx = self.vsr(hr_ref, hx)
    sr = hr_target + res
    self.hidden_state = hx
    return sr, flow


class SPMC(SuperResolution):
  def __init__(self, scale, channel, stage, lambda1, lambda2, residual,
               **kwargs):
    super(SPMC, self).__init__(scale, channel)
    self.spmc = DetailRevealer(scale, channel, **kwargs)
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.stage = stage
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.residual = residual

  def train(self, inputs, labels, learning_rate=None):
    self.spmc.reset()
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    srs = []
    warps = []
    flows = []
    center = len(frames) // 2
    target = frames[center]
    gt = labels[center]
    for ref in frames:
      sr, flow = self.spmc(target, ref)
      if self.residual:
        sr = sr + upsample(target, self.scale)
      warp = self.spmc.me.warper(ref, flow[:, 0], flow[:, 1])
      srs.append(sr)
      warps.append(warp)
      flows.append(flow)
    losses = [F.mse_loss(x, gt) for x in srs]
    image_loss = torch.stack(losses).sum()
    losses = []
    for w, f in zip(warps, flows):
      losses.append(F.l1_loss(w, target) + total_variance(f) * self.lambda1)
    me_loss = torch.stack(losses).sum()
    if self.stage == 1:
      loss = me_loss
    elif self.stage == 2:
      loss = image_loss
    else:
      loss = image_loss + me_loss * self.lambda2
    self.adam.zero_grad()
    loss.backward()
    self.adam.step()
    return {
      'me': me_loss.detach().cpu().numpy(),
      'image': image_loss.detach().cpu().numpy(),
      'total': loss.detach().cpu().numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    self.spmc.reset()
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    center = len(frames) // 2
    _frames = [pad_if_divide(x, 12, 'reflect') for x in frames]
    target = _frames[center]
    a = (target.size(2) - frames[0].size(2)) * self.scale
    b = (target.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    srs = []
    for ref in _frames:
      sr, _ = self.spmc(target, ref)
      if self.residual:
        sr = sr + upsample(target, self.scale)
      srs.append(sr[..., slice_h, slice_w].detach().cpu().numpy())
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      gt = labels[center]
      for i, v in enumerate(psnr(x, gt) for x in srs):
        metrics[f'psnr{i}'] = v
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('hr', gt, step=step)
        writer.image('sr', sr.clamp(0, 1), step=step)
    return srs, metrics
