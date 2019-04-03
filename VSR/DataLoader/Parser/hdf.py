#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

from functools import partial
from pathlib import Path

import h5py
import numpy as np

from . import _logger, parse_index
from ...Util.ImageProcess import array_to_img
from ...Util.ImageProcess import imresize
from ...Util.ImageProcess import shrink_to_multiple_scale


class Parser(object):
  r"""HDF5 Data parser.
  Any HDF5 file is acceptable, as long as it has following meta-data:
    - frames_info: a list of integers, tells length of each dataset (N of the NCHW or NHWC)
    - data_format: `channels_first` or `channels_last`.
  """

  def __init__(self, dataset, config):
    url = dataset.get(config.method, [])
    if not url or not Path(url[0]).exists():
      self.index = []
      return
    self.fd = h5py.File(url[0], 'r')
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
    if self.depth < 0:
      self.depth = 2 ** 31 - 1
    self.keys = list(self.fd.keys())
    frames = []
    # calculate index range
    for _f in self.fd.attrs['frames_info']:
      if _f < self.depth:
        frames.append(1)
      else:
        frames.append(_f - self.depth + 1)
    # make random index
    index = np.arange(int(np.sum(frames)))
    self.index = [parse_index(i, frames) for i in index]
    # capacity
    cap = []

    def _visit(key, item, cap):
      if isinstance(item, h5py.Dataset):
        _logger.debug(f"Indexing {key}...")
        cap.append(np.prod(item.shape, dtype='uint64'))

    self.fd.visititems(partial(_visit, cap=cap))
    self.capacity = np.sum(cap) * 3

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for n_key, n_seq in self.index[index]:
        key = self.keys[n_key]
        ret += self.gen_frames(key, n_seq)
      return ret
    else:
      n_key, n_seq = self.index[index]
      key = self.keys[n_key]
      return self.gen_frames(key, n_seq)

  def __len__(self):
    return len(self.index)

  def gen_frames(self, key, seq):
    _logger.debug(f'Prefetching {key} @{seq}')
    clip = self.fd[key]
    depth = self.depth
    depth = min(depth, clip.shape[0])
    hr = clip[seq:seq + depth]
    if self.fd.attrs['data_format'] != 'channels_last':
      hr = hr.transpose([0, 2, 3, 1])
    hr = [array_to_img(i, 'RGB') for i in hr]
    hr = [shrink_to_multiple_scale(img, self.scale) if self.modcrop else img for
          img in hr]
    lr = [imresize(img, np.reciprocal(self.scale, dtype='float32'),
                   resample=self.resample) for img in hr]
    hr = [img.convert(self.color_format) for img in hr]
    lr = [img.convert(self.color_format) for img in lr]
    return [(hr, lr, (key, seq, clip.shape[0]))]
