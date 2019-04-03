#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 21

import torch
import torch.nn.functional as F

from .Model import SuperResolution
from ..Util import Metrics


class Net(torch.nn.Module):
  def __init__(self, channel, scale):
    super(Net, self).__init__()
    conv1 = torch.nn.Conv2d(channel, 64, 5, 1, 2)
    conv2 = torch.nn.Conv2d(64, 32, 3, 1, 1)
    conv3 = torch.nn.Conv2d(32, channel*scale*scale, 3, 1, 1)
    ps = torch.nn.PixelShuffle(scale)
    self.body = torch.nn.Sequential(conv1, torch.nn.Tanh(),
                                    conv2, torch.nn.Tanh(),
                                    conv3, torch.nn.Tanh(), ps)

  def forward(self, x):
    return self.body(x)


class ESPCN(SuperResolution):
  def __init__(self, scale, channel, **kwargs):
    super(ESPCN, self).__init__(scale, channel, **kwargs)
    self.espcn = Net(channel, scale)
    self.opt = torch.optim.Adam(self.trainable_variables(), 1e-4)

  def train(self, inputs, labels, learning_rate=None):
    x = inputs[0] * 2 - 1
    sr = self.espcn(x)
    sr = (sr + 1) / 2
    loss = F.mse_loss(sr, labels[0])
    if learning_rate:
      for param_group in self.opt.param_groups:
        param_group["lr"] = learning_rate
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return {'l2': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    x = inputs[0] * 2 - 1
    sr = self.espcn(x).cpu().detach()
    sr = (sr + 1) / 2
    if labels is not None:
      metrics['psnr'] = Metrics.psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
