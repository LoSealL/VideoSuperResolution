#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/2 下午7:24

import copy

import numpy as np

from . import _logger, parse_index
from ..VirtualFile import ImageFile


class Parser(object):
  def __init__(self, dataset, config):
    urls = dataset.get(config.method, [])
    files = [ImageFile(fp) for fp in urls]
    self.files = self._parse_urls(files)
    self.depth = config.depth
    self.method = config.method
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
    n_frames = []
    for key in self.files:
      l = self.files[key][0].frames
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
        entry = list(self.files.keys())[key]
        qps = self.files[entry]
        k = np.random.randint(25, len(qps))
        hq = qps[0]
        lq = qps[k]
        ret += self.gen_frames(copy.deepcopy(hq), copy.deepcopy(lq), seq)
      return ret
    else:
      key, seq = self.index[index]
      entry = list(self.files.keys())[key]
      qps = self.files[entry]
      k = np.random.randint(25, len(qps))
      hq = qps[0]
      lq = qps[k]
      return self.gen_frames(copy.deepcopy(hq), copy.deepcopy(lq), seq)

  def __len__(self):
    return len(self.index)

  @property
  def capacity(self):
    # bytes per pixel
    bpp = 6 * self.depth
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf[0].shape, vf[0].frames, bpp), dtype=np.uint64)
                   for vf in self.files.values()])

  def gen_frames(self, hq, lq, seq):
    assert isinstance(hq, ImageFile) and isinstance(lq, ImageFile)

    _logger.debug(f'Prefetching {hq.name} @{seq}')
    depth = self.depth
    depth = min(depth, hq.frames)
    assert hq.frames == lq.frames
    hq.seek(seq)
    lq.seek(seq)
    hq_frames = [img.convert(self.color_format) for img in hq.read_frame(depth)]
    lq_frames = [img.convert(self.color_format) for img in lq.read_frame(depth)]
    return [(hq_frames, lq_frames, (lq.name, seq, lq.frames))]

  def _parse_urls(self, files):
    ret = {}
    for i in files:
      name, size, qp = i.name.split('_')
      if not name in ret:
        ret[name] = [None] * 52
      ret[name][int(qp)] = i
    return ret
