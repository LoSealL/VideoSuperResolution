"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: July 31st 2018

custom feature callback.
Usage:
    pass the function name to `run.py` args `--add_custom_callbacks func1 --add_custom_callbacks func2`.
    During training and testing, the `Trainer` will call the given functions to process input data

Note:
    functions must keep `kwargs` to ignore un-parsed arguments
"""

import numpy as np
from PIL import Image, ImageFilter


def color_inverse(inputs, **kwargs):
  """Invert color"""
  type_old = inputs.dtype
  inputs = inputs.astype('float')
  return np.clip(255 - inputs, 0, 255).astype(type_old)


def color_shift(inputs, value=0, **kwargs):
  """Shift color by `value`"""
  type_old = inputs.dtype
  inputs = inputs.astype('float')
  return np.clip(inputs + value, 0, 255).astype(type_old)


def red(inputs, value=255, **kwargs):
  """Generate pure red channel"""
  x = np.zeros_like(inputs)
  x[..., 0] = value
  return x


def green(inputs, value=255, **kwargs):
  """Generate pure green channel"""
  x = np.zeros_like(inputs)
  x[..., 1] = value
  return x


def blue(inputs, value=255, **kwargs):
  """Generate pure blue channel"""
  x = np.zeros_like(inputs)
  x[..., 2] = value
  return x


def chessboard(inputs, **kwargs):
  """Generate a chessboard picture"""
  x = np.zeros_like(inputs)
  c = np.random.randint(0, 255, [1, 1, 1, 3])
  for i in range(32):
    for j in range(32):
      x[:, i::64, j::64, :] = c
  c = np.random.randint(0, 255, [1, 1, 1, 3])
  for i in range(32):
    for j in range(32):
      x[:, i + 32::64, j::64, :] = c
  c = np.random.randint(0, 255, [1, 1, 1, 3])
  for i in range(32):
    for j in range(32):
      x[:, i::64, j + 32::64, :] = c
  c = np.random.randint(0, 255, [1, 1, 1, 3])
  for i in range(32):
    for j in range(32):
      x[:, i + 32::64, j + 32::64, :] = c
  return x


def noisy(inputs, sigma=15, **kwargs):
  """Generate random noise"""
  shape = inputs.shape
  return np.random.normal(0, sigma, shape)


def add_noise(inputs, sigma=15, **kwargs):
  """Add gaussian noise to inputs"""
  noise = noisy(inputs, sigma)
  noise += inputs.astype('float32')
  return np.clip(np.round(noise), 0, 255).astype('uint8')


def add_random_noise(inputs, sigma_max=50, **kwargs):
  """Add random gaussian noise to inputs"""
  sigma = np.random.randint(0, int(sigma_max))
  noise = np.random.normal(0, sigma, inputs.shape)
  noise += inputs.astype('float32')
  return np.clip(np.round(noise), 0, 255).astype('uint8')


def shave(inputs, div=64, **kwargs):
  """Crop borders"""
  div = int(div)
  h, w = inputs.shape[-3:-1]
  h_mod = h - h % div
  w_mod = w - w % div
  return inputs[..., :h_mod, :w_mod, :]


def pad(inputs, div=64, **kwargs):
  """Pad borders"""
  div = int(div)
  h, w = inputs.shape[-3:-1]
  ph = div - h % div
  pw = div - w % div
  if ph == div: ph = 0
  if pw == div: pw = 0
  ph = [ph // 2, ph - ph // 2]
  pw = [pw // 2, pw - pw // 2]
  if np.ndim(inputs) == 5:
    inputs = np.pad(inputs, [[0, 0], [0, 0], ph, pw, [0, 0]], 'edge')
  else:
    inputs = np.pad(inputs, [[0, 0], ph, pw, [0, 0]], 'edge')
  return inputs


def blur(inputs, width=2, **kwargs):
  """Apply blur kernel to images"""
  k = ImageFilter.GaussianBlur(float(width))
  shape = inputs.shape
  inputs = list(inputs.reshape([-1, *shape[-3:]]))
  for i, img in enumerate(inputs):
    assert img.dtype == 'uint8'
    inputs[i] = np.asarray(Image.fromarray(img, 'RGB').filter(k), 'uint8')
  inputs = np.stack(inputs)
  return inputs.reshape(shape)


def upsample(inputs, r=4, **kwargs):
  """Use PIL.Image.resize(resample=CUBIC) to upsample inputs"""
  r = float(r)
  res = []
  for img in inputs:
    h, w, c = img.shape
    if c == 3:
      img = Image.fromarray(img, 'RGB')
    elif c == 1:
      img = Image.fromarray(img[..., 0], 'L')
    else:
      raise ValueError
    img = img.resize([int(w * r), int(h * r)], resample=Image.BICUBIC)
    res.append(np.array(img))
  res = np.stack(res)
  if np.ndim(res) < 4:
    res = np.expand_dims(res, axis=-1)
  return res


scale=upsample