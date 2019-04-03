#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
from torch import nn
from torch.nn import functional as F

from ...Util.Utility import irtranspose, nd_meshgrid, transpose


class SpatioTemporalFlow(nn.Module):
  """Estimate spatio-temporal flow. Output shape (t, u, v).
    The output is normalized via `tanh` to [-1, 1]. Set `gain` to scale to
    un-normalized value.

  Args:
    gain: if `normalized` is False, `gain` means max displacement of the pixel.
    normalized: whether the output is normalized or absolute value.
  """

  def __init__(self, in_channels, gain=1, normalized=False):
    super(SpatioTemporalFlow, self).__init__()
    if normalized:
      self.gain = 1
    else:
      self.gain = gain
    self.conv0 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1),
                               nn.ReLU(True))
    self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True))
    self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True))
    self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True))
    self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True))
    self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(True))
    self.conv6 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(True))
    self.conv7 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(True))
    self.conv8 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(True))
    self.conv9 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(True))
    self.conv10 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True))
    self.conv11 = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())

  def forward(self, inputs: torch.Tensor):
    assert inputs.dim() == 5
    b, t, c, h, w = inputs.shape
    inputs = inputs.view([b, t * c, h, w])
    x = self.conv0(inputs)
    x = self.conv1(x)
    c2 = self.conv2(x)
    x = self.conv3(c2)
    c4 = self.conv4(x)
    x = self.conv5(c4)
    x = self.conv6(x)
    x = F.interpolate(x, scale_factor=2)
    x = self.conv7(torch.cat([x, c4], dim=1))
    x = self.conv8(x)
    x = F.interpolate(x, scale_factor=2)
    x = self.conv9(torch.cat([x, c2], dim=1))
    x = self.conv10(x)
    x = self.conv11(x)

    return x * self.gain


class STN(nn.Module):
  """Spatial transformer network.
    For optical flow based frame warping.
  """

  def __init__(self, mode='bilinear', padding_mode='zeros'):
    super(STN, self).__init__()
    self.mode = mode
    self.padding_mode = padding_mode

  def forward(self, inputs, u, v, normalized=True):
    batch = inputs.shape[0]
    device = inputs.device
    mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
    mesh = torch.stack([torch.Tensor(mesh)] * batch)
    # add flow to mesh
    _u, _v = u, v
    if not normalized:
      # flow needs to normalize to [-1, 1]
      h, w = inputs.shape[-2:]
      _u = u / w * 2
      _v = v / h * 2
    flow = torch.stack([_u, _v], dim=-1)
    assert flow.shape == mesh.shape
    mesh = mesh.to(device)
    mesh += flow
    return F.grid_sample(inputs, mesh,
                         mode=self.mode, padding_mode=self.padding_mode)


class STTN(nn.Module):
  """Spatio-temporal transformer network. (ECCV 2018)"""

  def __init__(self, transpose_ncthw=(0, 1, 2, 3, 4),
               mode='bilinear', padding_mode='zeros'):
    super(STTN, self).__init__()
    self.mode = mode
    self.padding_mode = padding_mode
    self.t = transpose_ncthw

  def forward(self, inputs, d, u, v, normalized=True):
    _error_msg = "STTN only works for 5D tensor but got {}D input!"
    if inputs.dim() != 5:
      raise ValueError(_error_msg.format(inputs.dim()))
    device = inputs.device
    batch, channel, t, h, w = (inputs.shape[i] for i in self.t)
    mesh = nd_meshgrid(t, h, w, permute=[2, 1, 0])
    mesh = torch.stack([torch.Tensor(mesh)] * batch)
    if not normalized:
      d /= t
      u /= w
      v /= h
    st_flow = torch.stack([d, u, v], dim=-1)
    st_flow = torch.stack([st_flow] * t, dim=1)
    assert st_flow.shape == mesh.shape
    mesh = mesh.to(device)
    mesh += st_flow
    inputs = transpose(inputs, self.t)
    warp = F.grid_sample(inputs, mesh, mode=self.mode,
                         padding_mode=self.padding_mode)
    # STTN warps into a single frame
    warp = warp[:, :, 0:1]
    return irtranspose(warp, self.t)
