#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


class Transformer(object):
  """Image transformer.

  Args:
      value: the parameter for each transform function.
      random: if specify 'uniform', generate value sampled from 0 to `+value`;
              if specify 'normal', generate value N~(mean=0, std=value)
  """

  def __init__(self, value=1, random=None):
    self._v = value
    self._r = random

  @property
  def value(self):
    if self._r == 'uniform':
      return np.random.uniform(0, self._v)
    elif self._r == 'normal':
      return np.random.normal(0, self._v)
    else:
      return self._v


class _Transformer1(Transformer):
  def __call__(self, img: Image.Image):
    assert isinstance(img, Image.Image)
    return self.call(img)

  def call(self, img):
    raise NotImplementedError


class Tidy(_Transformer1):
  def call(self, img: Image.Image):
    scale = self.value
    shape = np.array((img.width, img.height))
    shape -= shape % scale
    return img.crop([0, 0, *shape.tolist()])


class Bicubic(_Transformer1):
  def call(self, img: Image.Image):
    scale = self.value
    shape = np.array((img.width, img.height))
    if scale < 1:
      rscale = int(1 / scale)
      if np.any(shape % rscale):
        raise ValueError(f"Image size is not divisible by {rscale}.")
      return img.resize(shape // rscale, resample=Image.BICUBIC)
    else:
      return img.resize((shape * scale).astype('int32'), resample=Image.BICUBIC)


class Brightness(_Transformer1):
  def call(self, img: Image.Image):
    brightness = max(0, self.value)
    return ImageEnhance.Brightness(img).enhance(brightness)


class Contrast(_Transformer1):
  def call(self, img: Image.Image):
    contrast = self.value
    return ImageEnhance.Contrast(img).enhance(contrast)


class Sharpness(_Transformer1):
  def call(self, img):
    sharp = min(max(0, self.value), 2)
    return ImageEnhance.Sharpness(img).enhance(sharp)


class GaussianBlur(_Transformer1):
  def call(self, img):
    radius = self.value
    return ImageFilter.GaussianBlur(radius).filter(img)


class _Transformer2(Transformer):
  def __call__(self, img: np.ndarray):
    assert isinstance(img, np.ndarray)
    return self.call(img)

  def call(self, img):
    raise NotImplementedError


class GaussianWhiteNoise(_Transformer2):
  def call(self, img):
    shape = img.shape
    noise = np.random.normal(0, self.value, shape)
    noise += img.astype('float32')
    return np.clip(np.round(noise), 0, 255).astype('uint8')


class FixedVideoLengthBatch(_Transformer2):
  def call(self, img):
    assert img.ndim == 5, f"img is not 5D, which is {img.ndim}"
    depth = int(self.value)
    shape = img.shape
    if shape[1] <= depth:
      return img
    ret = []
    for i in range(shape[1] - depth + 1):
      ret.append(img[:, i * depth: (i + 1) * depth])
    return np.stack(ret, 1).reshape([-1, depth, *shape[-3:]])
