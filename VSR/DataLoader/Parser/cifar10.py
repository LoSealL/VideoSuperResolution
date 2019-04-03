#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import numpy as np
import tensorflow as tf

from . import _logger
from ...Util.ImageProcess import array_to_img
from ...Util.ImageProcess import imresize


class Parser(object):
  def __init__(self, dataset, config):
    (x, _), (y, _) = tf.keras.datasets.cifar10.load_data()
    self.data = x if config.method == 'train' else y
    self.scale = config.scale
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

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for d in self.data[index]:
        ret += self.gen_frames(d)
      return ret
    else:
      d = self.data[index]
      return self.gen_frames(d)

  def __len__(self):
    return self.data.shape[0]

  def gen_frames(self, data):
    hr = [array_to_img(data, 'RGB')]
    lr = [imresize(img, np.reciprocal(self.scale, dtype='float32'),
                   resample=self.resample) for img in hr]
    hr = [img.convert(self.color_format) for img in hr]
    lr = [img.convert(self.color_format) for img in lr]
    return [(hr, lr, ('cifar10', 0, 0))]
