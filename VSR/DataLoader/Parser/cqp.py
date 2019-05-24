#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/24 下午2:47

import copy

import numpy as np

from . import _logger, parse_index
from ..VirtualFile import ImageFile

_target_qp = 40


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
    n_idr = []
    for key in self.files:
      l = self.files[key][0].frames
      num_idr = l // self.depth
      if l % self.depth and config.method not in ('train', 'val'):
        num_idr += 1
      n_idr.append(num_idr)
    index = np.arange(int(np.sum(n_idr)))
    self.index = [parse_index(i, n_idr) for i in index]

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for key, seq in self.index[index]:
        entry = list(self.files.keys())[key]
        qps = self.files[entry]
        seq = seq * self.depth
        hq = qps[0]
        lq = qps[1]
        ret += self.gen_frames(copy.deepcopy(hq), copy.deepcopy(lq), seq)
      return ret
    else:
      key, seq = self.index[index]
      entry = list(self.files.keys())[key]
      qps = self.files[entry]
      # get the [seq]-th IDR
      seq = seq * self.depth
      hq = qps[0]
      lq = qps[1]
      return self.gen_frames(copy.deepcopy(hq), copy.deepcopy(lq), seq)

  def __len__(self):
    return len(self.index)

  @property
  def capacity(self):
    # bytes per pixel
    bpp = 6
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
      name_components = i.name.split('_')
      qp = name_components[-1]
      name = ''
      for j in name_components[:-2]: name += j
      if not name in ret:
        ret[name] = [None] * 2
      if int(qp) == _target_qp:
        ret[name][1] = i
      else:
        ret[name][0] = i
    return ret
