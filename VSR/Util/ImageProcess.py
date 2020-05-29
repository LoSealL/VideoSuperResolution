#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import numpy as np
from PIL import Image

from ..Backend import DATA_FORMAT


def array_to_img(x: np.ndarray, mode='RGB', data_format=None, min_val=0,
                 max_val=255):
  """Convert an ndarray to PIL Image."""

  x = np.squeeze(x).astype('float32')
  x = (x - min_val) / (max_val - min_val)
  x = x.clip(0, 1) * 255
  if data_format not in ('channels_first', 'channels_last'):
    data_format = DATA_FORMAT
  if np.ndim(x) == 2:
    return Image.fromarray(x.astype('uint8'), mode='L').convert(mode)
  elif np.ndim(x) == 3:
    if data_format == 'channels_first':
      x = x.transpose([1, 2, 0])
    return Image.fromarray(x.astype('uint8'), mode=mode)
  elif np.ndim(x) == 4:
    if data_format == 'channels_first':
      x = x.transpose([0, 2, 3, 1])
    ret = [Image.fromarray(np.round(i).astype('uint8'), mode=mode) for i in x]
    return ret.pop() if len(ret) is 1 else ret
  elif np.ndim(x) >= 5:
    raise ValueError(f"Dimension of x must <= 4. Got {np.ndim(x)}.")


def img_to_array(img, data_format=None):
  """Converts a PIL Image instance to a Numpy array.

    !!Copy from Keras!!

    Assure the array's ndim is 3
  """
  if not isinstance(img, Image.Image):
    return img
  if data_format is None:
    data_format = DATA_FORMAT
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ', data_format)
  # Numpy array x has format (height, width, channel)
  # or (channel, height, width)
  # but original PIL image1 has format (width, height, channel)
  x = np.asarray(img, dtype=np.uint8)
  if len(x.shape) == 3:
    if data_format == 'channels_first':
      x = x.transpose([2, 0, 1])
  elif len(x.shape) == 2:
    if data_format == 'channels_first':
      x = x.reshape((1, x.shape[0], x.shape[1]))
    else:
      x = x.reshape((x.shape[0], x.shape[1], 1))
  else:
    raise ValueError('Unsupported image1 shape: ', x.shape)
  return x


def imresize(image, scale, size=None, mode=None, resample=None):
  """Image resize using simple cubic provided in PIL"""

  def _resample(name: str):
    if 'cubic' in name:
      return Image.BICUBIC
    if 'linear' in name:
      return Image.BILINEAR
    if 'nearest' in name:
      return Image.NEAREST
    return 0

  dtype = Image.Image
  if isinstance(image, np.ndarray):
    dtype = np.ndarray
    mode = 'RGB' or mode
    image = array_to_img(image, mode)
  if size is None:
    size = (np.array(image.size) * scale).astype(int)
  if image.mode in ('RGB', 'BGR'):
    image = image.convert('YCbCr')
  mode = image.mode if not mode else mode
  if isinstance(resample, str):
    resample = _resample(resample)
  if not resample:
    resample = Image.BICUBIC
  image = image.resize(size, resample=resample).convert(mode)
  if dtype is np.ndarray:
    return img_to_array(image, DATA_FORMAT)
  return image


def imread(url, mode='RGB'):
  """Read image from file to ndarray"""

  img = Image.open(url)
  return img_to_array(img.convert(mode))


_Y601 = (0.299, 0.587, 0.114)
_Y709 = (0.2126, 0.7152, 0.0722)
_UMAX = 0.436
_VMAX = 0.615
_U601 = (_Y601[0] / (_Y601[2] - 1), _Y601[1] / (_Y601[2] - 1), 1.0)
_V601 = (1.0, _Y601[1] / (_Y601[0] - 1), _Y601[2] / (_Y601[0] - 1))
_Y601 = np.array(_Y601, dtype=np.float32)
_U601 = np.array(_U601, dtype=np.float32) * _UMAX
_V601 = np.array(_V601, dtype=np.float32) * _VMAX
_U709 = (_Y709[0] / (_Y709[2] - 1), _Y709[1] / (_Y709[2] - 1), 1.0)
_V709 = (1.0, _Y709[1] / (_Y709[0] - 1), _Y709[2] / (_Y709[0] - 1))
_Y709 = np.array(_Y709, dtype=np.float32)
_U709 = np.array(_U709, dtype=np.float32) * _UMAX
_V709 = np.array(_V709, dtype=np.float32) * _VMAX
_T601 = np.stack([_Y601, _U601, _V601])
_T709 = np.stack([_Y709, _U709, _V709])


def rgb_to_yuv(img, max_val=1.0, standard='bt601'):
  """convert rgb to yuv

  There are plenty of rgb2yuv functions in python modules, but here we want to
  make things more clearly.

  Usually there are two standards: BT.601 and BT.709. While bt601 is the most
  widely used (PIL, opencv, matlab's rgb2gray and also in the lovely
  tf.image), somehow bt709 is not found used in any libs.
  However, matlab's rgb2ycbcr uses different weights, which come from
  C.A. Poynton. Most SR papers use matlab's rgb2ycbcr in benchmark,
  because it gets the highest PSNR :)

  Args:
       img: a 3-D numpy array. If `dtype=uint8`, it ranges from [0, 255], if
         `dtype=float`, it ranges from [0, 1]
       max_val: a scalar representing range of the image value
       standard: a string, should be one of ('bt601', 'bt709', 'matlab')

  Return:
      yuv image
  """
  _standard = standard.lower()
  if _standard not in ('bt601', 'bt709', 'matlab'):
    raise ValueError('Not known standard:', standard)
  if DATA_FORMAT == 'channels_first':
    img = img.transpose([1, 2, 0])
  if img.shape[-1] != 3:
    return img
  """ matrix used in matlab
    yuv = _trans * rgb + _bias
  """
  _trans = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112],
                     [112, -93.786, -18.214]], dtype=np.float32)
  _bias = np.array([16, 128, 128], dtype=np.float32)
  _trans /= 255
  _bias /= 255
  _img = img.reshape([-1, 3]) / max_val
  if _standard == 'bt601':
    _yuv = np.matmul(_T601, _img.transpose())
    _yuv = _yuv.transpose() + np.array([0, 0.5, 0.5])
    _yuv = _yuv.reshape(img.shape)
  elif _standard == 'bt709':
    _yuv = np.matmul(_T709, _img.transpose())
    _yuv = _yuv.transpose() + np.array([0, 0.5, 0.5])
    _yuv = _yuv.reshape(img.shape)
  else:
    _yuv = np.matmul(_trans, _img.transpose())
    _yuv = _yuv.transpose() + _bias
    _yuv = _yuv.reshape(img.shape)
  _yuv = np.clip(_yuv, 0, 1) * max_val
  if DATA_FORMAT == 'channels_first':
    _yuv = _yuv.transpose([2, 0, 1])
  return _yuv.astype(img.dtype)
