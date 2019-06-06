#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/2 上午10:54

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from .sof.modules import SOFVSR as _SOFVSR
from .sof.modules import optical_flow_warp
from ..Util import Metrics
from ..Util.Metrics import total_variance


class SOFVSR(SuperResolution):
  """Note: SOF is Y-channel SR with depth=3"""

  def __init__(self, scale, channel, depth=3, **kwargs):
    super(SOFVSR, self).__init__(scale, channel, **kwargs)
    self.sof = _SOFVSR(scale, channel, depth)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    assert depth == 3
    self.center = depth // 2

  def train(self, inputs, labels, learning_rate=None):
    pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
    pre = torch.squeeze(pre, dim=1)
    cur = torch.squeeze(cur, dim=1)
    nxt = torch.squeeze(nxt, dim=1)
    low_res = torch.cat([pre, cur, nxt], dim=1)
    sr, flow01, flow21 = self.sof(low_res)
    hrp, hr, hrn = torch.split(labels[0], 1, dim=1)
    hrp = torch.squeeze(hrp, dim=1)
    hr = torch.squeeze(hr, dim=1)
    hrn = torch.squeeze(hrn, dim=1)
    loss_sr = F.mse_loss(sr, hr)
    pre_d = F.avg_pool2d(pre, 2)
    cur_d = F.avg_pool2d(cur, 2)
    nxt_d = F.avg_pool2d(nxt, 2)

    pre_d_warp = optical_flow_warp(pre_d, flow01[2])
    pre_warp = optical_flow_warp(pre, flow01[1])
    hrp_warp = optical_flow_warp(hrp, flow01[0])
    nxt_d_warp = optical_flow_warp(nxt_d, flow21[2])
    nxt_warp = optical_flow_warp(nxt, flow21[1])
    hrn_warp = optical_flow_warp(hrn, flow21[0])

    loss_lvl1 = F.mse_loss(pre_d_warp, cur_d) + F.mse_loss(nxt_d_warp, cur_d) + \
                0.01 * (total_variance(flow01[2]) + total_variance(flow21[2]))
    loss_lvl2 = F.mse_loss(pre_warp, cur) + F.mse_loss(nxt_warp, cur) + \
                0.01 * (total_variance(flow01[1]) + total_variance(flow21[1]))
    loss_lvl3 = F.mse_loss(hrp_warp, hr) + F.mse_loss(hrn_warp, hr) + \
                0.01 * (total_variance(flow01[0]) + total_variance(flow21[0]))
    loss = loss_sr + 0.01 * (loss_lvl3 + 0.25 * loss_lvl2 + 0.125 * loss_lvl1)
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {
      'image': loss_sr.detach().cpu().numpy(),
      'flow/lvl1': loss_lvl1.detach().cpu().numpy(),
      'flow/lvl2': loss_lvl2.detach().cpu().numpy(),
      'flow/lvl3': loss_lvl3.detach().cpu().numpy(),
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
    low_res = torch.cat([pre, cur, nxt], dim=2)
    low_res = torch.squeeze(low_res, dim=1)
    sr, _, _ = self.sof(low_res)
    sr = sr.cpu().detach()
    if labels is not None:
      hr = labels[0][:, self.center]
      metrics['psnr'] = Metrics.psnr(sr, hr)
    return [sr.numpy()], metrics
