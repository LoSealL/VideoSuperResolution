#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 28

import torch
import torch.nn.functional as F

from .Contrib.ntire20.xiaozhong.ops import define_D, define_F, define_G
from .Model import SuperResolution
from ..Util import Metrics


class RealSR(SuperResolution):
  """
  RealSR proposed by Xiaozhong Ji
  See (NTIRE report, not full paper): https://arxiv.org/pdf/2005.01996.pdf
  """

  def __init__(self, channel=3, scale=4, nf=64, nb=23, **kwargs):
    super(RealSR, self).__init__(channel=channel, scale=scale)
    self.weights = [
      kwargs.get('pixel_weight', 1),
      kwargs.get('feature_weight', 0),
      kwargs.get('gan_weight', 0)
    ]
    self.realsr_g = define_G(in_nc=channel, out_nc=channel, nf=nf, nb=nb)
    self.opt_g = torch.optim.Adam(self.trainable_variables('realsr_g'), 1e-4,
                                  betas=(0.5, 0.999))
    if self.weights[1] > 0:
      self.feature_net = define_F()  # for feature loss
    if self.weights[2] > 0:
      self.realsr_d = define_D(in_nc=channel, nf=nf, nlayers=3)
      self.opt_d = torch.optim.Adam(self.trainable_variables('realsr_d'), 1e-4,
                                    betas=(0.5, 0.999))

  def train(self, inputs, labels, learning_rate=None):
    sr = self.realsr_g(inputs[0])
    pixel_loss = F.l1_loss(sr, labels[0]) * self.weights[0]
    loss = pixel_loss
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt_g.zero_grad()
    loss.backward()
    self.opt_g.step()
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.realsr_g(inputs[0]).cpu().detach()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
