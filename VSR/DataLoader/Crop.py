#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import numpy as np
from ..Backend import DATA_FORMAT


class Cropper(object):
  def __init__(self, scale):
    self.scale = scale

  def __call__(self, img_pair: tuple, shape: list):
    assert len(img_pair) >= 2, \
      f"Pair must contain more than 2 elements, which is {img_pair}"
    for img in img_pair:
      assert img.ndim == len(shape), \
        f"Shape mis-match: {img.ndim} != {len(shape)}"

    return self.call(img_pair, shape)

  def call(self, img: tuple, shape: (list, tuple)) -> tuple:
    raise NotImplementedError


class RandomCrop(Cropper):
  def call(self, img: tuple, shape: (list, tuple)) -> tuple:
    hr, lr = img
    if lr.shape[-2] < shape[-2]:
      raise ValueError(
        f"Batch shape is larger than data: {lr.shape} vs {shape}")
    ind = [np.random.randint(nd + 1) for nd in lr.shape - np.array(shape)]
    slc1 = [slice(n, n + s) for n, s in zip(ind, shape)]
    slc2 = slc1.copy()
    if DATA_FORMAT == 'channels_last':
      slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
      slc2[-3] = slice(ind[-3] * self.scale,
                       (ind[-3] + shape[-3]) * self.scale)
    else:
      slc2[-1] = slice(ind[-1] * self.scale,
                       (ind[-1] + shape[-1]) * self.scale)
      slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
    return hr[tuple(slc2)], lr[tuple(slc1)]


class CenterCrop(Cropper):
  def call(self, img: tuple, shape: (list, tuple)) -> tuple:
    hr, lr = img
    ind = [nd // 2 for nd in hr.shape - np.array(shape)]
    slc1 = [slice(n, n + s) for n, s in zip(ind, shape)]
    slc2 = slc1.copy()
    if DATA_FORMAT == 'channels_last':
      slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
      slc2[-3] = slice(ind[-3] * self.scale,
                       (ind[-3] + shape[-3]) * self.scale)
    else:
      slc2[-1] = slice(ind[-1] * self.scale,
                       (ind[-1] + shape[-1]) * self.scale)
      slc2[-2] = slice(ind[-2] * self.scale,
                       (ind[-2] + shape[-2]) * self.scale)
    return hr[tuple(slc2)], lr[tuple(slc1)]
