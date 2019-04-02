#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/2 上午10:54

import torch
import torch.nn.functional as F
import numpy as np


def pad_if_divide(x: torch.Tensor, value, mode='constant'):
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


def shave_if_divide(x: torch.Tensor, value):
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


def transpose(x: torch.Tensor, dims):
  """transpose like numpy and tensorflow"""
  _dims = list(dims)
  for i in range(len(_dims)):
    if _dims[i] != i:
      x = x.transpose(i, _dims[i])
      j = _dims.index(i)
      _dims[i], _dims[j] = i, _dims[i]
  return x


def irtranspose(x: torch.Tensor, dims):
  """back transpose.
    `x = irtranspose(transpose(x, d), d)`
  """

  _dims = list(dims)
  _ir_dims = [_dims.index(i) for i in range(len(_dims))]
  return transpose(x, _ir_dims)


def nd_meshgrid(*size, permute=None):
  _error_msg = ("Permute index must match mesh dimensions, "
                "should have {} indexes but got {}")
  size = np.array(size)
  ranges = []
  for x in size:
    ranges.append(np.linspace(-1, 1, x))
  mesh = np.stack(np.meshgrid(*ranges, indexing='ij'))
  if permute is not None:
    if len(permute) != len(size):
      raise ValueError(_error_msg.format(len(size), len(permute)))
    mesh = mesh[permute]
  return mesh.transpose(*range(1, mesh.ndim), 0)
