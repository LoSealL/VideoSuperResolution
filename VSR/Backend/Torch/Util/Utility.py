#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import torch
import torch.nn.functional as F

from VSR.Backend import DATA_FORMAT
from VSR.Util.Math import weights_downsample, weights_upsample


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
  assert 3 <= x.dim() <= 4, f"Dim of x is not 3 or 4, which is {x.dim()}"
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
  kernel, s = weights_downsample(scale)
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
  kernels, s = weights_upsample(scale)
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


def imfilter(image: torch.Tensor, kernel: torch.Tensor, padding=None):
  with torch.no_grad():
    if image.dim() == 3:
      image = image.unsqueeze(0)
    assert image.dim() == 4, f"Dim of image must be 4, but is {image.dim()}"
    if kernel.dtype != image.dtype:
      kernel = kernel.to(dtype=image.dtype)
    if kernel.dim() == 2:
      kernel = kernel.unsqueeze(0)
      kernel = torch.cat([kernel] * image.shape[0])
    assert kernel.dim() == 3, f"Dim of kernel must be 3, but is {kernel.dim()}"

    ret = []
    for i, k in zip(image.split(1), kernel.split(1)):
      _c = i.shape[1]
      _k = k.unsqueeze(0)
      _p = torch.zeros_like(_k)
      _m = []
      for j in range(_c):
        t = [_p] * _c
        t[j] = _k
        _m.append(torch.cat(t, dim=1))
      _k = torch.cat(_m, dim=0)
      if padding is None:
        ret.append(F.conv2d(i, _k, padding=[x // 2 for x in kernel.shape[1:]]))
      elif callable(padding):
        ret.append(F.conv2d(padding(i), _k))
      else:
        raise ValueError("Wrong padding value!")
    return torch.cat(ret)


def poisson_noise(inputs: torch.Tensor, stddev=None, sigma_max=0.16,
                  channel_wise=1):
  """Add poisson noise to inputs."""

  if stddev is None:
    stddev = torch.rand(channel_wise) * sigma_max
  stddev = torch.tensor(stddev, device=inputs.device)
  if DATA_FORMAT == 'channels_first':
    stddev = stddev.reshape([1, -1] + [1] * (inputs.ndim - 2))
  else:
    stddev = stddev.reshape([1] * (inputs.ndim - 1) + [-1])
  sigma_map = (1 - inputs) * stddev
  return torch.randn_like(inputs) * sigma_map


def gaussian_noise(inputs: torch.Tensor, stddev=None, sigma_max=0.06,
                   channel_wise=1):
  """Add channel wise gaussian noise."""

  if stddev is None:
    stddev = torch.rand(channel_wise) * sigma_max
  stddev = torch.tensor(stddev, device=inputs.device)
  if DATA_FORMAT == 'channels_first':
    stddev = stddev.reshape([1, -1] + [1] * (inputs.ndim - 2))
  else:
    stddev = stddev.reshape([1] * (inputs.ndim - 1) + [-1])
  noise_map = torch.randn_like(inputs) * stddev
  return noise_map


def gaussian_poisson_noise(inputs, stddev_s=None, stddev_c=None,
                           max_s=0.16, max_c=0.06):
  noise = poisson_noise(inputs, stddev_s, max_s)
  return noise + gaussian_noise(inputs, stddev_c, max_c)
