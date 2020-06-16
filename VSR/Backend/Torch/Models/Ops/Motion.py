#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 15

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

  def forward(self, inputs, u, v=None, gain=1):
    batch = inputs.size(0)
    device = inputs.device
    mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
    mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
    mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
    # add flow to mesh
    if v is None:
      assert u.shape[1] == 2, "optical flow must have 2 channels"
      _u, _v = u[:, 0], u[:, 1]
    else:
      _u, _v = u, v
    if not self.norm:
      # flow needs to normalize to [-1, 1]
      h, w = inputs.shape[-2:]
      _u = _u / w * 2
      _v = _v / h * 2
    flow = torch.stack([_u, _v], dim=-1) * gain
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


class CoarseFineFlownet(nn.Module):
  def __init__(self, channel):
    """Coarse to fine flow estimation network

    Originally from paper "Real-Time Video Super-Resolution with Spatio-Temporal
    Networks and Motion Compensation".
    See Vespcn.py
    """

    super(CoarseFineFlownet, self).__init__()
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
    self.warp_c = STN(padding_mode='border')

  def forward(self, target, ref, gain=1):
    """Estimate optical flow from `ref` frame to `target` frame"""

    flow_c = self.coarse_flow(torch.cat((ref, target), 1))
    wc = self.warp_c(ref, flow_c[:, 0], flow_c[:, 1])
    flow_f = self.fine_flow(torch.cat((ref, target, flow_c, wc), 1)) + flow_c
    flow_f *= gain
    return flow_f


class Flownet(nn.Module):
  def __init__(self, channel):
    """Flow estimation network

    Originally from paper "FlowNet: Learning Optical Flow with Convolutional
    Networks" and adapted according to paper "Frame-Recurrent Video
    Super-Resolution".
    See Frvsr.py

    Args:
      channel: input channels of each sequential frame
    """

    super(Flownet, self).__init__()
    f = 32
    layers = []
    in_c = channel * 2
    for i in range(3):
      layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.MaxPool2d(2)]
      in_c = f
      f *= 2
    for i in range(3):
      layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
      in_c = f
      f //= 2
    layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
    layers += [nn.Conv2d(f, 2, 3, 1, 1), nn.Tanh()]
    self.body = nn.Sequential(*layers)

  def forward(self, target, ref, gain=1):
    """Estimate densely optical flow from `ref` to `target`

    Args:
      target: frame A
      ref: frame B
      gain: a scalar multiplied to final flow map
    """

    x = torch.cat((target, ref), 1)
    x = self.body(x) * gain
    return x
