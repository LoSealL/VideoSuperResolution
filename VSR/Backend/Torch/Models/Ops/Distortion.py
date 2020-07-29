#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 15

import random

import torch
import torch.nn as nn

from VSR.Util.Math import gaussian_kernel
from ...Util.Utility import gaussian_noise, imfilter, poisson_noise


class Distorter(nn.Module):
  """Randomly add the noise and blur of an image.

  Args:
      gaussian_noise_std (float or tuple of float (min, max)): How much to
          additive gaussian white noise. gaussian_noise_std is chosen uniformly
          from [0, std] or the given [min, max]. Should be non negative numbers.
      poisson_noise_std (float or tuple of float (min, max)): How much to
          poisson noise. poisson_noise_std is chosen uniformly from [0, std] or
          the given [min, max]. Should be non negative numbers.
      gaussian_blur_std (float or tuple of float (min, max)): How much to
          blur kernel. gaussian_blur_std is chosen uniformly from [0, std] or
          the given [min, max]. Should be non negative numbers.
  """

  def __init__(self,
               gaussian_noise_std=0,
               poisson_noise_std=0,
               gaussian_blur_std=0):
    super(Distorter, self).__init__()
    self.awgn = self._check_input(gaussian_noise_std, 'awgn', center=0,
                                  bound=(0, 75 / 255), clip_first_on_zero=True)
    self.poisson = self._check_input(poisson_noise_std, 'poisson', center=0,
                                     bound=(0, 50 / 255),
                                     clip_first_on_zero=True)
    self.blur = self._check_input(gaussian_blur_std, 'blur', center=0)
    self.blur_padding = nn.ReflectionPad2d(7)

  def _check_input(self, value, name, center=1, bound=(0, float('inf')),
                   clip_first_on_zero=True):
    if isinstance(value, (tuple, list)) and len(value) == 2:
      if not bound[0] <= value[0] <= value[1] <= bound[1]:
        raise ValueError("{} values should be between {}".format(name, bound))
    else:
      if value < 0:
        raise ValueError(
            "If {} is a single number, it must be non negative.".format(name))
      value = [center - value, center + value]
      if clip_first_on_zero:
        value[0] = max(value[0], 0)
    # if value is 0 or (1., 1.) for brightness/contrast/saturation
    # or (0., 0.) for hue, do nothing
    if value[0] == value[1] == center:
      value = None
    return value

  def forward(self, img):
    factors = []
    # noise & blur
    blur_factor = 0
    if self.blur is not None:
      blur_factor = random.uniform(*self.blur)
      img = imfilter(
          img,
          torch.tensor(gaussian_kernel(15, blur_factor),
                       device=img.device),
          self.blur_padding)
    awgn_factor = (0, 0, 0)
    if self.awgn is not None:
      _r = random.uniform(*self.awgn)
      _g = random.uniform(*self.awgn)
      _b = random.uniform(*self.awgn)
      img += gaussian_noise(img, stddev=(_r, _g, _b))
      awgn_factor = (_r, _g, _b)
    poisson_factor = (_r, _g, _b)
    if self.poisson is not None:
      _r = random.uniform(*self.poisson)
      _g = random.uniform(*self.poisson)
      _b = random.uniform(*self.poisson)
      img += poisson_noise(img, stddev=(_r, _g, _b))
      poisson_factor = (_r, _g, _b)
    fac = [blur_factor, *awgn_factor, *poisson_factor]
    factors.append(torch.tensor(fac))
    img = img.clamp(0, 1)
    return img, torch.stack(factors).to(img.device)
