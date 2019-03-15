#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import torch
import torch.nn.functional as F


def pad_if_divide(x, value, mode='constant'):
  """pad tensor if its width and height couldn't be divided by `value`.

  Args:
      x: a tensor at least has 3 dimensions.
      value: value to divide width and height.
      mode: a string, representing padding mode.
  Return:
      padded tensor.
  """

  shape = x.shape
  assert 3 <= x.dim() <= 4
  h = shape[-2]
  w = shape[-1]
  dh = h + (value - h % value) % value - h
  dw = w + (value - w % value) % value - w
  pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
  return F.pad(x, pad, mode)


def shave_if_divide(x, value):
  """crop tensor if its width and height couldn't be divided by `value`.

  Args:
      x: a tensor at least has 3 dimensions.
      value: value to divide width and height.
  Return:
      cropped tensor.
  """

  shape = x.shape
  h = shape[-2]
  w = shape[-1]
  dh = h % value
  dw = w % value
  return x[..., dh // 2:h - dh // 2, dw // 2:w - dw // 2]
