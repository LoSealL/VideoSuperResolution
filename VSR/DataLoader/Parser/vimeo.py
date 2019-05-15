#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/7 上午10:49

from pathlib import Path

import h5py
import numpy as np

from . import _logger
from ...Util.ImageProcess import array_to_img
from ...Util.ImageProcess import imresize
from ...Util.ImageProcess import shrink_to_multiple_scale


class Parser(object):
  r"""Vimeo HDF5 Data parser.
  Only vimeo HDF5 file is acceptable, it has following meta-data:
    - data_format: `channels_first` or `channels_last`.
  """

  def __init__(self, dataset, config):
    url = dataset.get(config.method, [])
    if not url or not Path(url[0]).exists():
      self.index = []
      return
    self.scale = config.scale
    self.depth = config.depth
    self.method = config.method
    self.modcrop = config.modcrop
    self.resample = config.resample
    if config.convert_to.lower() in ('gray', 'l'):
      self.color_format = 'L'
    elif config.convert_to.lower() in ('yuv', 'ycbcr'):
      self.color_format = 'YCbCr'
    elif config.convert_to.lower() in ('rgb',):
      self.color_format = 'RGB'
    else:
      _logger.warning('Use grayscale by default. '
                      'Unknown format {}'.format(config.convert_to))
      self.color_format = 'L'
    self.fd = fd = h5py.File(url[0], 'r')  # keep `fd` open
    self.data = fd[list(fd.keys())[0]]
    _logger.debug(self.data.shape)
    if self.depth < 0:
      self.depth = self.data.shape[1]  # max length of vimeo sep
    self.data_format = fd.attrs['data_format']

  def __getitem__(self, index):
    if isinstance(index, slice):
      hr = self.data[index]
    else:
      hr = [self.data[index]]
    ret = []
    for i in hr:
      ret += self.gen_frames(i)
    return ret

  def __len__(self):
    return self.data.shape[0]

  @property
  def capacity(self):
    return np.prod(self.data.shape, dtype='uint64') * 3

  def gen_frames(self, hr):
    length = hr.shape[0]
    seq = 0
    if self.depth < length:
      seq = np.random.randint(length - self.depth)
      hr = hr[seq:seq + self.depth]
    if self.data_format == 'channels_first':
      hr = hr.transpose([0, 2, 3, 1])
    hr = [array_to_img(i, 'RGB') for i in hr]
    hr = [shrink_to_multiple_scale(img, self.scale) if self.modcrop else img for
          img in hr]
    lr = [imresize(img, np.reciprocal(self.scale, dtype='float32'),
                   resample=self.resample) for img in hr]
    hr = [img.convert(self.color_format) for img in hr]
    lr = [img.convert(self.color_format) for img in lr]
    return [(hr, lr, ('vimeo', seq, length))]
