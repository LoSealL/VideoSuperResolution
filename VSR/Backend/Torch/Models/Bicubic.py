#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 17

# Non-trainable bicubic, for performance benchmarking and debugging

import torch
import torch.nn as nn
import torchvision as tv

from .Model import SuperResolution
from ..Util.Metrics import psnr


class Cubic(nn.Module):
  def __init__(self, scale):
    super(Cubic, self).__init__()
    self.to_pil = tv.transforms.ToPILImage()
    self.to_tensor = tv.transforms.ToTensor()
    self.scale = scale

  def forward(self, x):
    if self.scale == 1:
      return x
    ret = []
    for img in [i[0] for i in x.split(1, dim=0)]:
      img = self.to_pil(img.cpu())
      w = img.width
      h = img.height
      img = img.resize([w * self.scale, h * self.scale], 3)
      img = self.to_tensor(img)
      ret.append(img)
    return torch.stack(ret).to(x.device)


class BICUBIC(SuperResolution):
  def __init__(self, scale=4, channel=3, **kwargs):
    super(BICUBIC, self).__init__(scale, channel, **kwargs)
    self.cubic = Cubic(scale)
    self.cri = nn.L1Loss()

  def train(self, inputs, labels, learning_rate=None):
    sr = self.cubic(inputs[0])
    loss = self.cri(sr, labels[0])
    return {'l1': loss.detach().cpu().numpy()}

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    sr = self.cubic(inputs[0]).cpu().detach()
    if labels is not None:
      metrics['psnr'] = psnr(sr.numpy(), labels[0].cpu().numpy())
    return [sr.numpy()], metrics
