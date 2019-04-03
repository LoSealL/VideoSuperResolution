#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import copy

import numpy as np

from . import _logger, parse_index
from ..VirtualFile import RawFile
from ...Util.ImageProcess import imresize, shrink_to_multiple_scale


class Parser(object):
  def __init__(self, dataset, config):
    urls = dataset.get(config.method, [])
    self.file_objects = [
      RawFile(fp, dataset.mode, (dataset.width, dataset.height)) for fp in urls]
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
    # calculate index range
    n_frames = []
    for _f in self.file_objects:
      l = _f.frames
      if l < self.depth:
        n_frames.append(1)
      else:
        n_frames.append(l - self.depth + 1)
    index = np.arange(int(np.sum(n_frames)))
    self.index = [parse_index(i, n_frames) for i in index]

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for key, seq in self.index[index]:
        vf = self.file_objects[key]
        ret += self.gen_frames(copy.deepcopy(vf), seq)
      return ret
    else:
      key, seq = self.index[index]
      vf = self.file_objects[key]
      return self.gen_frames(copy.deepcopy(vf), seq)

  def __len__(self):
    return len(self.index)

  def gen_frames(self, vf, index):
    assert isinstance(vf, RawFile)

    _logger.debug(f'Prefetching {vf.name} @{index}')
    vf.reopen()
    depth = self.depth
    depth = min(depth, vf.frames)
    vf.seek(index)
    hr = [shrink_to_multiple_scale(img, self.scale)
          if self.modcrop else img for img in vf.read_frame(depth)]
    lr = [imresize(img,
                   np.reciprocal(self.scale, dtype='float32'),
                   resample=self.resample)
          for img in hr]
    hr = [img.convert(self.color_format) for img in hr]
    lr = [img.convert(self.color_format) for img in lr]
    return [(hr, lr, (vf.name, index, vf.frames))]

  @property
  def capacity(self):
    # bytes per pixel
    bpp = 1.5 * (1 + np.reciprocal(self.scale, dtype='float32'))
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                   for vf in self.file_objects])
