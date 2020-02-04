#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import numpy as np
import torch
import torch.nn.functional as F


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
  pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
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


def _bicubic_filter(x, a=-0.5):
  # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  if x < 0:
    x = -x
  if x < 1:
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
  if x < 2:
    return (((x - 5) * x + 8) * x - 4) * a
  return 0


def list_rshift(l, s):
  for _ in range(s):
    l.insert(0, l.pop(-1))
  return l


def _weights_downsample(scale_factor):
  if scale_factor < 1:
    ss = int(1 / scale_factor + 0.5)
  else:
    ss = int(scale_factor + 0.5)
  support = 2 * ss
  ksize = support * 2 + 1
  weights = []
  for lambd in range(ksize):
    dist = -2 + (2 * lambd + 1) / support
    weights.append(_bicubic_filter(dist))
  h = np.array([weights])
  h /= h.sum()
  v = h.transpose()
  kernel = np.matmul(v, h)
  assert kernel.shape == (ksize, ksize), f"{kernel.shape} != [{ksize}]"
  return kernel, ss


def _weights_upsample(scale_factor):
  if scale_factor < 1:
    ss = int(1 / scale_factor + 0.5)
  else:
    ss = int(scale_factor + 0.5)
  support = 2
  ksize = support * 2 + 1
  weights = [[] for _ in range(ss)]
  for i in range(ss):
    for lambd in range(ksize):
      dist = int((1 + ss + 2 * i) / 2 / ss) + lambd - 1.5 - (2 * i + 1) / 2 / ss
      weights[i].append(_bicubic_filter(dist))
  w = [np.array([i]) / np.sum(i) for i in weights]
  w = list_rshift(w, ss - ss // 2)
  kernels = []
  for i in range(len(w)):
    for j in range(len(w)):
      kernels.append(np.matmul(w[i].transpose(), w[j]))
  return kernels, ss


def _push_shape_4d(x):
  dim = x.dim()
  if dim == 2:
    return x.unsqueeze(0).unsqueeze(1), 2
  elif dim == 3:
    return x.unsqueeze(0), 3
  elif dim == 4:
    return x, 4
  else:
    raise ValueError("Unsupported tensor! Must be 2D/3D/4D")


def _pop_shape(x, shape):
  if shape == 2:
    return x[0, ..., 0]
  elif shape == 3:
    return x[0]
  elif shape == 4:
    return x
  else:
    raise ValueError("Unsupported shape! Must be 2/3/4")


def downsample(img, scale, border='reflect'):
  """Bicubical downsample via **CONV2D**. Using PIL's kernel.

  Args:
    img: a tf tensor of 2/3/4-D.
    scale: n or 1/n. `n` must be integer >= 2.
    border: padding mode. Recommend to 'REFLECT'.
  """
  device = img.device
  kernel, s = _weights_downsample(scale)
  if s == 1:
    return img  # bypass
  kernel = kernel.astype('float32')
  kernel = torch.from_numpy(kernel)
  p1 = int(s * 3 / 2)
  p2 = 4 * s - int(s * 3 / 2)
  img, shape = _push_shape_4d(img)
  img_ex = F.pad(img, [p1, p2, p1, p2], mode=border)
  c = img_ex.shape[1]
  assert c is not None, "img must define channel number"
  c = int(c)
  filters = torch.reshape(torch.eye(c, c), [c, c, 1, 1]) * kernel
  img_s = F.conv2d(img_ex, filters.to(device), stride=s)
  img_s = _pop_shape(img_s, shape)
  return img_s


def upsample(img, scale, border='reflect'):
  """Bicubical upsample via **CONV2D**. Using PIL's kernel.

  Args:
    img: a tf tensor of 2/3/4-D.
    scale: must be integer >= 2.
    border: padding mode. Recommend to 'REFLECT'.
  """
  device = img.device
  kernels, s = _weights_upsample(scale)
  if s == 1:
    return img  # bypass
  kernels = [k.astype('float32') for k in kernels]
  kernels = [torch.from_numpy(k) for k in kernels]
  p1 = 1 + s // 2
  p2 = 3
  img, shape = _push_shape_4d(img)
  img_ex = F.pad(img, [p1, p2, p1, p2], mode=border)
  c = img_ex.shape[1]
  assert c is not None, "img must define channel number"
  c = int(c)
  filters = [torch.reshape(torch.eye(c, c), [c, c, 1, 1]) * k for k in kernels]
  weights = torch.stack(filters, dim=0).transpose(0, 1).reshape([-1, c, 5, 5])
  img_s = F.conv2d(img_ex, weights.to(device))
  img_s = F.pixel_shuffle(img_s, s)
  more = s // 2 * s
  crop = slice(more - s // 2, - (s // 2))
  img_s = _pop_shape(img_s[..., crop, crop], shape)
  return img_s


def bicubic_resize(img, scale, border='reflect'):
  if scale > 1:
    return upsample(img, scale, border)
  elif 0 < scale < 1:
    return downsample(img, scale, border)
  elif scale == 1:
    return img  # bypass
  else:
    raise ValueError("Wrong scale factor!")
