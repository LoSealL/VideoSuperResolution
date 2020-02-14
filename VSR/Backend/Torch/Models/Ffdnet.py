#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/27 下午5:22

import torch
import torch.nn.functional as F
from torch import nn

from .Arch import EasyConv2d, SpaceToDepth, Upsample
from .Model import SuperResolution
from ..Framework.Summary import get_writer
from ..Util import Metrics


class Net(nn.Module):
  def __init__(self, channel, layers, bn, filters=64):
    super(Net, self).__init__()
    self.spd = SpaceToDepth(2)
    body = [EasyConv2d(channel * 4 + 1, filters, 3, activation='relu')]
    for i in range(1, layers):
      body.append(EasyConv2d(filters, filters, 3, activation='relu', use_bn=bn))
    body += [
      Upsample(filters, 2),
      EasyConv2d(filters, channel, 3)
    ]
    self.body = nn.Sequential(*body)

  def forward(self, x, sigma):
    x = self.spd(x)
    sig = torch.ones_like(x)[:, 0:1] * sigma
    return self.body(torch.cat((x, sig), dim=1))


class FFDNET(SuperResolution):
  def __init__(self, scale, channel, n_layers, level, training, **kwargs):
    super(FFDNET, self).__init__(scale, channel)
    self.ffdnet = Net(channel, n_layers, True)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)
    self.level = level / 255
    self.is_training = training

  def train(self, inputs, labels, learning_rate=None):
    for opt in self.opts.values():
      if learning_rate:
        for param_group in opt.param_groups:
          param_group["lr"] = learning_rate
    lr = inputs[0]
    sigma = torch.rand(1, device=lr.device) * 75 / 255
    noise = torch.randn_like(lr) * sigma
    hr = self.ffdnet((lr + noise).clamp(0, 1), sigma)
    loss = F.l1_loss(hr, labels[0])
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {
      'loss': loss.detach().cpu().numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    lr = inputs[0]
    if self.is_training:
      sigma = torch.rand(1, device=lr.device) * 75 / 255
      noise = torch.randn_like(lr) * sigma
    else:
      sigma = self.level
      noise = torch.zeros_like(lr)
    hr = self.ffdnet((lr + noise).clamp(0, 1), sigma).detach().cpu()
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(hr, labels[0])
      writer = get_writer(self.name)
      if writer is not None:
        step = kwargs.get('epoch')
        writer.image('gt', labels[0], step=step)
        writer.image('clean', hr.clamp(0, 1), step=step)
    return [hr.numpy()], metrics
