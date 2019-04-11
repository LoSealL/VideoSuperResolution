#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
from torch import nn
from torch.nn import functional as F

from ..Arch import Rdb, SpaceToDepth, Upsample
from ...Util.Utility import irtranspose, nd_meshgrid, transpose


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
               normalize=False, mode='bilinear', padding_mode='zeros'):
    super(STTN, self).__init__()
    self.normalized = normalize
    self.mode = mode
    self.padding_mode = padding_mode
    self.t = transpose_ncthw

  def forward(self, inputs, d, u, v):
    _error_msg = "STTN only works for 5D tensor but got {}D input!"
    if inputs.dim() != 5:
      raise ValueError(_error_msg.format(inputs.dim()))
    device = inputs.device
    batch, channel, t, h, w = (inputs.shape[i] for i in self.t)
    mesh = nd_meshgrid(t, h, w, permute=[2, 1, 0])
    mesh = torch.stack([torch.Tensor(mesh)] * batch)
    _d, _u, _v = d, u, v
    if not self.normalized:
      _d = d / t * 2
      _u = u / w * 2
      _v = v / h * 2
    st_flow = torch.stack([u, v, d], dim=-1)
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


class Fnet(nn.Module):
  def __init__(self, channel, L=2, gain=64):
    super(Fnet, self).__init__()
    self.lq_entry = nn.Sequential(
      nn.Conv2d(channel * (L + 1), 16, 3, 1, 1),
      SpaceToDepth(4),
      nn.Conv2d(256, 64, 1, 1, 0),
      Rdb(64), Rdb(64))
    self.hq_entry = nn.Sequential(
      nn.Conv2d(channel * L, 16, 3, 1, 1),
      SpaceToDepth(4),
      nn.Conv2d(256, 64, 1, 1, 0),
      Rdb(64), Rdb(64))
    self.flownet = nn.Sequential(
      nn.Conv2d(128, 64, 1, 1, 0),
      Rdb(64), Rdb(64), Upsample(64, 4),
      nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())
    gain = torch.as_tensor([L, gain, gain], dtype=torch.float32)
    self.gain = gain.reshape(1, 3, 1, 1)

  def forward(self, lq, hq):
    x = torch.cat(lq, dim=1)
    y = torch.cat(hq, dim=1)
    x = self.lq_entry(x)
    y = self.hq_entry(y)
    z = torch.cat([x, y], dim=1)
    flow = self.flownet(z)
    gain = self.gain.to(flow.device)
    return flow * gain
