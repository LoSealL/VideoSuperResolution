#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
from torch import nn
from torch.nn import functional as F
from ..video.motion import STN


class RB(nn.Module):
  def __init__(self, inchannels, outchannels):
    super(RB, self).__init__()
    self.conv1 = nn.Conv2d(inchannels, 64, 3, 1, 1)
    self.conv2 = nn.Conv2d(64, outchannels, 3, 1, 1)
    if inchannels != outchannels:
      self.sc = nn.Conv2d(inchannels, outchannels, 1)

  def forward(self, inputs):
    x = F.relu(inputs)
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    if hasattr(self, 'sc'):
      sc = self.sc(inputs)
    else:
      sc = inputs
    return x + sc


class MotionCompensation(nn.Module):
  def __init__(self, channel, gain=32):
    super(MotionCompensation, self).__init__()
    self.gain = gain
    in_c = channel * 2
    # Coarse Flow
    conv1 = nn.Sequential(nn.Conv2d(in_c, 24, 5, 2, 2), nn.ReLU(True))
    conv2 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
    conv3 = nn.Sequential(nn.Conv2d(24, 24, 5, 2, 2), nn.ReLU(True))
    conv4 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
    conv5 = nn.Sequential(nn.Conv2d(24, 32, 3, 1, 1), nn.Tanh())
    up1 = nn.PixelShuffle(4)
    self.coarse_flow = nn.Sequential(conv1, conv2, conv3, conv4, conv5, up1)
    # Fine Flow
    in_c = channel * 3 + 2
    conv1 = nn.Sequential(nn.Conv2d(in_c, 24, 5, 2, 2), nn.ReLU(True))
    conv2 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
    conv3 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
    conv4 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
    conv5 = nn.Sequential(nn.Conv2d(24, 8, 3, 1, 1), nn.Tanh())
    up2 = nn.PixelShuffle(2)
    self.fine_flow = nn.Sequential(conv1, conv2, conv3, conv4, conv5, up2)
    self.warp1 = STN(padding_mode='border')
    self.warp2 = STN(padding_mode='border')

  def forward(self, target, ref):
    flow0 = self.coarse_flow(torch.cat([ref, target], 1))
    flow0 *= self.gain
    w0 = self.warp1(ref, flow0[:, 0], flow0[:, 1])
    flow1 = self.fine_flow(torch.cat([ref, target, flow0, w0], 1))
    flow1 *= self.gain
    flow1 += flow0
    w1 = self.warp2(ref, flow1[:, 0], flow1[:, 1])
    return w1, flow1


class SRNet(nn.Module):
  def __init__(self, scale, channel, depth):
    super(SRNet, self).__init__()
    self.entry = nn.Conv2d(channel * depth, 64, 3, 1, 1)
    self.exit = nn.Conv2d(64, channel, 3, 1, 1)
    self.body = nn.Sequential(RB(64, 64), RB(64, 64), RB(64, 64), nn.ReLU(True))
    self.conv = nn.Conv2d(64, 64 * scale ** 2, 3, 1, 1)
    self.up = nn.PixelShuffle(scale)

  def forward(self, inputs):
    x = self.entry(inputs)
    y = self.body(x) + x
    y = self.conv(y)
    y = self.up(y)
    y = self.exit(y)
    return y


class VESPCN(nn.Module):
  def __init__(self, scale, channel, depth):
    super(VESPCN, self).__init__()
    self.sr = SRNet(scale, channel, depth)
    self.mc = MotionCompensation(channel)
    self.depth = depth

  def forward(self, *inputs):
    center = self.depth // 2
    target = inputs[center]
    refs = inputs[:center] + inputs[center + 1:]
    warps = []
    flows = []
    for r in refs:
      warp, flow = self.mc(target, r)
      warps.append(warp)
      flows.append(flow)
    warps.append(target)
    x = torch.cat(warps, 1)
    sr = self.sr(x)
    return sr, warps[:-1], flows
