#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 下午3:24

import logging

import torch
import torch.nn.functional as F
from torch import nn

from .Dbpn import Dbpn, DownBlock, UpBlock
from .Model import SuperResolution
from .Ops.Blocks import EasyConv2d, RB
from .Ops.Loss import total_variance
from .Ops.Motion import Flownet, STN
from ..Framework.Summary import get_writer
from ..Util.Metrics import psnr
from ..Util.Utility import pad_if_divide, upsample

_logger = logging.getLogger("VSR.RBPN")
_logger.info("LICENSE: RBPN is implemented by M. Haris, et. al. @alterzero")
_logger.warning(
    "I use unsupervised flownet to estimate optical flow, rather than pyflow module.")


class DbpnS(nn.Module):
  def __init__(self, scale, base_filter, feat, num_stages):
    super(DbpnS, self).__init__()
    kernel, stride = Dbpn.get_kernel_stride(scale)
    # Initial Feature Extraction
    self.feat1 = EasyConv2d(base_filter, feat, 1, activation='prelu')
    # Back-projection stages
    for i in range(num_stages):
      self.__setattr__(f'up{i}', UpBlock(feat, kernel, stride))
      if i < num_stages - 1:
        # not the last layer
        self.__setattr__(f'down{i}', DownBlock(feat, kernel, stride))
    self.num_stages = num_stages
    # Reconstruction
    self.output_conv = EasyConv2d(feat * num_stages, feat, 1)

  def forward(self, x):
    x = self.feat1(x)
    h1 = [self.__getattr__('up0')(x)]
    d1 = []
    for i in range(self.num_stages):
      d1.append(self.__getattr__(f'down{i}')(h1[-1]))
      h1.append(self.__getattr__(f'up{i + 1}')(d1[-1]))
    x = self.output_conv(torch.cat(h1, 1))
    return x


class Rbpn(nn.Module):
  def __init__(self, channel, scale, base_filter, feat, n_resblock,
               nFrames):
    super(Rbpn, self).__init__()
    self.nFrames = nFrames
    kernel, stride = Dbpn.get_kernel_stride(scale)
    # Initial Feature Extraction
    self.feat0 = EasyConv2d(channel, base_filter, 3, activation='prelu')
    self.feat1 = EasyConv2d(8, base_filter, 3, activation='prelu')
    ###DBPNS
    self.DBPN = DbpnS(scale, base_filter, feat, 3)
    # Res-Block1
    modules_body1 = [RB(base_filter, kernel_size=3, activation='prelu') for _ in
                     range(n_resblock)]
    modules_body1.append(
        EasyConv2d(base_filter, feat, kernel, stride, activation='prelu',
                   transposed=True))
    self.res_feat1 = nn.Sequential(*modules_body1)
    # Res-Block2
    modules_body2 = [RB(feat, kernel_size=3, activation='prelu') for _ in
                     range(n_resblock)]
    modules_body2.append(EasyConv2d(feat, feat, 3, activation='prelu'))
    self.res_feat2 = nn.Sequential(*modules_body2)
    # Res-Block3
    modules_body3 = [RB(feat, kernel_size=3, activation='prelu') for _ in
                     range(n_resblock)]
    modules_body3.append(EasyConv2d(feat, base_filter, kernel, stride,
                                    activation='prelu'))
    self.res_feat3 = nn.Sequential(*modules_body3)
    # Reconstruction
    self.output = EasyConv2d((nFrames - 1) * feat, channel, 3)

  def forward(self, x, neigbor, flow):
    ### initial feature extraction
    feat_input = self.feat0(x)
    feat_frame = []
    for j in range(len(neigbor)):
      feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]), 1)))

    ####Projection
    Ht = []
    for j in range(len(neigbor)):
      h0 = self.DBPN(feat_input)
      h1 = self.res_feat1(feat_frame[j])

      e = h0 - h1
      e = self.res_feat2(e)
      h = h0 + e
      Ht.append(h)
      feat_input = self.res_feat3(h)

    ####Reconstruction
    out = torch.cat(Ht, 1)
    output = self.output(out)

    return output


class Composer(nn.Module):
  def __init__(self, **kwargs):
    super(Composer, self).__init__()
    self.module = Rbpn(**kwargs)
    self.fnet = Flownet(kwargs['num_channels'])
    self.warper = STN(padding_mode='border')

  def forward(self, target, neighbors):
    flows = []
    warps = []
    for i in neighbors:
      flow = self.fnet(target, i, gain=32)
      warp = self.warper(i, flow[:, 0], flow[:, 1])
      flows.append(flow)
      warps.append(warp)
    sr = self.module(target, neighbors, flows)
    return sr, flows, warps


class RBPN(SuperResolution):
  def __init__(self, scale, channel, depth, residual, **kwargs):
    super(RBPN, self).__init__(scale, channel, **kwargs)
    self.depth = depth
    self.res = residual
    self.w = kwargs.get('weights', [1, 1e-4])
    ops = {
      'num_channels': channel,
      'scale_factor': scale,
      'base_filter': kwargs.get('base_filter', 256),
      'feat': kwargs.get('feat', 64),
      'num_stages': kwargs.get('num_stages', 3),
      'n_resblock': kwargs.get('n_resblock', 5),
      'nFrames': depth
    }
    self.rbpn = Composer(**ops)
    self.adam = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    target = frames.pop(self.depth // 2)
    neighbors = frames
    sr, flows, warps = self.rbpn(target, neighbors)
    if self.res:
      sr = sr + upsample(target, self.scale)

    image_loss = F.l1_loss(sr, labels[self.depth // 2])
    warp_loss = [F.l1_loss(w, target) for w in warps]
    tv_loss = [total_variance(f) for f in flows]
    flow_loss = torch.stack(warp_loss).sum() * self.w[0] + \
                torch.stack(tv_loss).sum() * self.w[1]
    loss = image_loss + flow_loss
    self.adam.zero_grad()
    loss.backward()
    self.adam.step()
    return {
      'flow': flow_loss.detach().cpu().numpy(),
      'image': image_loss.detach().cpu().numpy(),
      'total': loss.detach().cpu().numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    _frames = [pad_if_divide(x, 8, 'reflect') for x in frames]
    target = _frames.pop(self.depth // 2)
    neighbors = _frames
    a = (target.size(2) - frames[0].size(2)) * self.scale
    b = (target.size(3) - frames[0].size(3)) * self.scale
    slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    sr, _, warps = self.rbpn(target, neighbors)
    if self.res:
      sr = sr + upsample(target, self.scale)
    sr = sr[..., slice_h, slice_w].detach().cpu()
    if labels is not None:
      labels = [x.squeeze(1) for x in labels[0].split(1, dim=1)]
      gt = labels[self.depth // 2]
      metrics['psnr'] = psnr(sr, gt)
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs['epoch']
        writer.image('hr', gt, step=step)
        writer.image('sr', sr.clamp(0, 1), step=step)
        writer.image('warp/0', warps[0].clamp(0, 1), step=step)
        writer.image('warp/1', warps[1].clamp(0, 1), step=step)
    return [sr.numpy()], metrics
