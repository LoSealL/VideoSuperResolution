#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/26 上午11:39

import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from ..Arch import EasyConv2d
from ..video.motion import STN
from ...Util.Utility import upsample


class Conv2dLSTMCell(nn.Module):
  """ConvLSTM cell.
  Copied from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81
  Special thanks to @Kaixhin
  """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True):

    super(Conv2dLSTMCell, self).__init__()
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.padding_h = tuple(
      k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
    self.dilation = dilation
    self.groups = groups
    self.weight_ih = Parameter(
      torch.Tensor(4 * out_channels, in_channels // groups, *kernel_size))
    self.weight_hh = Parameter(
      torch.Tensor(4 * out_channels, out_channels // groups, *kernel_size))
    self.weight_ch = Parameter(
      torch.Tensor(3 * out_channels, out_channels // groups, *kernel_size))
    if bias:
      self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
      self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
      self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
    else:
      self.register_parameter('bias_ih', None)
      self.register_parameter('bias_hh', None)
      self.register_parameter('bias_ch', None)
    self.register_buffer('wc_blank', torch.zeros(out_channels))
    self.reset_parameters()

  def reset_parameters(self):
    n = 4 * self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.weight_ih.data.uniform_(-stdv, stdv)
    self.weight_hh.data.uniform_(-stdv, stdv)
    self.weight_ch.data.uniform_(-stdv, stdv)
    if self.bias_ih is not None:
      self.bias_ih.data.uniform_(-stdv, stdv)
      self.bias_hh.data.uniform_(-stdv, stdv)
      self.bias_ch.data.uniform_(-stdv, stdv)

  def forward(self, input, hx):
    h_0, c_0 = hx

    wx = F.conv2d(input, self.weight_ih, self.bias_ih, self.stride,
                  self.padding, self.dilation, self.groups)
    wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                  self.padding_h, self.dilation, self.groups)
    # Cell uses a Hadamard product instead of a convolution?
    wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                  self.padding_h, self.dilation, self.groups)
    v = Variable(self.wc_blank).reshape((1, -1, 1, 1))
    wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels],
                                v.expand(wc.size(0), wc.size(1) // 3,
                                         wc.size(2), wc.size(3)),
                                wc[:, 2 * self.out_channels:]), 1)

    i = torch.sigmoid(wxhc[:, :self.out_channels])
    f = torch.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
    g = torch.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
    o = torch.sigmoid(wxhc[:, 3 * self.out_channels:])

    c_1 = f * c_0 + i * g
    h_1 = o * torch.tanh(c_1)
    return h_1, (h_1, c_1)


class ZeroUpsample(nn.Module):
  def __init__(self, scale_factor):
    super(ZeroUpsample, self).__init__()
    self.ps = nn.PixelShuffle(scale_factor)
    self.scale = scale_factor

  def forward(self, x):
    z = torch.zeros_like(x).repeat_interleave(self.scale ** 2 - 1, dim=1)
    x = torch.cat((x, z), dim=1)
    return self.ps(x)


class SPMC(nn.Module):
  def __init__(self, scale):
    super(SPMC, self).__init__()
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
    self.warper = STN(padding_mode='border')

  def forward(self, target, ref, to_tuple=None):
    flow0 = self.coarse_flow(torch.cat((ref, target), dim=1))
    w0 = self.warper(ref, flow0[:, 0], flow0[:, 1])
    flow_res = self.fine_flow(torch.cat((ref, target, flow0, w0), dim=1))
    flow1 = (flow_res + flow0) * self.gain
    if to_tuple:
      return flow1[:, 0], flow1[:, 1]
    return flow1


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
    self.spmc = SPMC(scale)
    self.vsr = DetailFusion(channel, self.base_filter)
    self.scale = scale
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
    self.hidden_state = [x.detach() for x in hx]
    return sr, flow
