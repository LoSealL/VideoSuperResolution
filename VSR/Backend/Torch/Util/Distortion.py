#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 6 - 9

import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor

from VSR.Util.Math import gaussian_kernel
from ..Util.Utility import gaussian_noise, imfilter, poisson_noise


class Distortion(nn.Module):
  """Randomly change the brightness, contrast and saturation of an image.

  Args:
      brightness (float or tuple of float (min, max)): How much to jitter brightness.
          brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
          or the given [min, max]. Should be non negative numbers.
      contrast (float or tuple of float (min, max)): How much to jitter contrast.
          contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
          or the given [min, max]. Should be non negative numbers.
      saturation (float or tuple of float (min, max)): How much to jitter saturation.
          saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
          or the given [min, max]. Should be non negative numbers.
      hue (float or tuple of float (min, max)): How much to jitter hue.
          hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
          Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
  """

  def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,
               gaussian_noise_std=0, poisson_noise_std=0, gaussian_blur_std=0):
    super(Distortion, self).__init__()
    self.brightness = self._check_input(brightness, 'brightness')
    self.contrast = self._check_input(contrast, 'contrast')
    self.saturation = self._check_input(saturation, 'saturation')
    self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                 clip_first_on_zero=False)
    self.awgn = self._check_input(gaussian_noise_std, 'awgn', center=0,
                                  bound=(0, 0.75), clip_first_on_zero=True)
    self.poisson = None
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

  @staticmethod
  def get_params(brightness, contrast, saturation, hue, awgn, poisson, blur):
    """Get a randomized transform to be applied on image.

    Arguments are same as that of __init__.

    Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
    """
    transforms = []

    brightness_factor = 0
    if brightness is not None:
      brightness_factor = random.uniform(brightness[0], brightness[1])
      transforms.append(
          Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
    contrast_factor = 0
    if contrast is not None:
      contrast_factor = random.uniform(contrast[0], contrast[1])
      transforms.append(
          Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
    saturation_factor = 0
    if saturation is not None:
      saturation_factor = random.uniform(saturation[0], saturation[1])
      transforms.append(
          Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
    hue_factor = 0
    if hue is not None:
      hue_factor = random.uniform(hue[0], hue[1])
      transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

    random.shuffle(transforms)
    transform = Compose([
      ToPILImage('RGB'),
      *transforms,
      ToTensor()
    ])
    factors = [
      brightness_factor, contrast_factor, saturation_factor, hue_factor
    ]
    return transform, factors

  def forward(self, x):
    img = [x_[0].cpu() for x_ in torch.split(x, 1, dim=0)]
    factors = []
    for i in range(len(img)):
      # color jitter
      transform, fac = self.get_params(self.brightness, self.contrast,
                                       self.saturation, self.hue, self.awgn,
                                       self.poisson, self.blur)
      img[i] = transform(img[i])
      # noise & blur
      blur_factor = 0
      if self.blur is not None:
        blur_factor = random.uniform(*self.blur)
        img[i] = imfilter(
            img[i],
            torch.tensor(gaussian_kernel(15, blur_factor),
                         device=img[i].device),
            self.blur_padding)[0]
      awgn_factor = 0
      if self.awgn is not None:
        awgn_factor = random.uniform(*self.awgn)
        img[i] += gaussian_noise(img[i], stddev=awgn_factor, channel_wise=False)
      poisson_factor = 0
      if self.poisson is not None:
        poisson_factor = random.uniform(*self.poisson)
        img[i] += poisson_noise(img[i], stddev=poisson_factor)
      fac += [awgn_factor, poisson_factor, blur_factor]
      factors.append(torch.tensor(fac))
      img[i] = img[i].clamp(0, 1)
    return torch.stack(img).to(x.device), torch.stack(factors).to(x.device)
