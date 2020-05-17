#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import torch
from torch import nn
from torch.nn import functional as F

from VSR.Util.Math import nd_meshgrid
from ...Util.Utility import irtranspose, transpose


class STN(nn.Module):
  """Spatial transformer network.
    For optical flow based frame warping.

  Args:
    mode: sampling interpolation mode of `grid_sample`
    padding_mode: can be `zeros` | `borders`
    normalized: flow value is normalized to [-1, 1] or absolute value
  """

  def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
    super(STN, self).__init__()
    self.mode = mode
    self.padding_mode = padding_mode
    self.norm = normalize

  def forward(self, inputs, u, v):
    batch = inputs.size(0)
    device = inputs.device
    mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
    mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
    mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
    # add flow to mesh
    _u, _v = u, v
    if not self.norm:
      # flow needs to normalize to [-1, 1]
      h, w = inputs.shape[-2:]
      _u = u / w * 2
      _v = v / h * 2
    flow = torch.stack([_u, _v], dim=-1)
    assert flow.shape == mesh.shape, \
      f"Shape mis-match: {flow.shape} != {mesh.shape}"
    mesh = mesh + flow
    return F.grid_sample(inputs, mesh,
                         mode=self.mode, padding_mode=self.padding_mode)


class STTN(nn.Module):
  """Spatio-temporal transformer network. (ECCV 2018)

  Args:
    transpose_ncthw: how input tensor be transposed to format NCTHW
    mode: sampling interpolation mode of `grid_sample`
    padding_mode: can be `zeros` | `borders`
    normalize: flow value is normalized to [-1, 1] or absolute value
  """

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
    mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
    mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
    _d, _u, _v = d, u, v
    if not self.normalized:
      _d = d / t * 2
      _u = u / w * 2
      _v = v / h * 2
    st_flow = torch.stack([_u, _v, _d], dim=-1)
    st_flow = st_flow.unsqueeze(1).repeat_interleave(t, dim=1)
    assert st_flow.shape == mesh.shape, \
      f"Shape mis-match: {st_flow.shape} != {mesh.shape}"
    mesh = mesh + st_flow
    inputs = transpose(inputs, self.t)
    warp = F.grid_sample(inputs, mesh, mode=self.mode,
                         padding_mode=self.padding_mode)
    # STTN warps into a single frame
    warp = warp[:, :, 0:1]
    return irtranspose(warp, self.t)
