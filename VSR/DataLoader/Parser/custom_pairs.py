#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import copy

import numpy as np

from . import _logger, parse_index
from ..VirtualFile import ImageFile


class Parser(object):
  def __init__(self, dataset, config):
    urls = dataset.get(config.method, [])
    pair = getattr(dataset, '{}_pair'.format(config.method))
    if not pair:
      pair = []
    urls = sorted(urls)
    pair = sorted(pair)
    self.files = [ImageFile(fp).attach_pair(p) for fp, p in zip(urls, pair)]
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
    # calculate index range
    depth = self.depth
    if self.method in ('test', 'infer') and depth > 1:
      # padding head and tail
      for vf in self.files:
        vf.pad([depth // 2, depth // 2])
        vf.pair.pad([depth // 2, depth // 2])
    if depth < 0:
      depth = 2 ** 31 - 1
    n_frames = []
    for _f in self.files:
      l = _f.frames
      if l < depth:
        n_frames.append(1)
      else:
        n_frames.append(l - depth + 1)
    index = np.arange(int(np.sum(n_frames)))
    self.index = [parse_index(i, n_frames) for i in index]

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for key, seq in self.index[index]:
        vf = self.files[key]
        ret += self.gen_frames(copy.deepcopy(vf), seq)
      return ret
    else:
      key, seq = self.index[index]
      vf = self.files[key]
      return self.gen_frames(copy.deepcopy(vf), seq)

  def __len__(self):
    return len(self.index)

  def gen_frames(self, vf, index):
    assert isinstance(vf, ImageFile)

    _logger.debug('Prefetching ' + vf.name)
    # read all frames if depth is set to -1
    depth = self.depth if self.depth > 0 else vf.frames
    depth = min(depth, vf.frames)
    vf.seek(index)
    vf.pair.seek(index)
    hr = [img for img in vf.read_frame(depth)]
    lr = [img for img in vf.pair.read_frame(depth)]
    hr = [img.convert(self.color_format) for img in hr]
    lr = [img.convert(self.color_format) for img in lr]
    return [(hr, lr, (vf.name, index, vf.frames))]

  @property
  def capacity(self):
    # bytes per pixel
    depth = 1 if self.depth < 1 else self.depth
    bpp = 6 * depth
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                   for vf in self.files])
