#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import copy

import numpy as np

from . import _logger
from ..VirtualFile import ImageFile


class Parser(object):
  def __init__(self, dataset, config):
    urls = dataset.get(config.method, [])
    pair = getattr(dataset, '{}_pair'.format(config.method))
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

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for vf in self.files[index]:
        ret += self.gen_frames(copy.deepcopy(vf), vf.frames)
      return ret
    else:
      vf = self.files[index]
      return self.gen_frames(copy.deepcopy(vf), vf.frames)

  def __len__(self):
    return len(self.files)

  def gen_frames(self, vf, clips):
    assert isinstance(vf, ImageFile)

    _logger.debug('Prefetching ' + vf.name)
    depth = self.depth
    if self.method in ('test', 'infer') and depth > 1:
      # padding head and tail
      vf.pad([depth // 2, depth // 2])
    # read all frames if depth is set to -1
    if depth == -1:
      depth = vf.frames
    index = np.arange(0, vf.frames - depth + 1)
    if self.method == 'train':
      np.random.shuffle(index)
    frames = []
    for i in index[:clips]:
      vf.seek(i)
      hr = [img for img in vf.read_frame(depth)]
      lr = [img for img in vf.pair.read_frame(depth)]
      hr = [img.convert(self.color_format) for img in hr]
      lr = [img.convert(self.color_format) for img in lr]
      frames.append((hr, lr, (vf.name, i, vf.frames)))
    vf.reopen()  # necessary, rewind the read pointer
    return frames

  @property
  def capacity(self):
    bpp = 6  # bytes per pixel
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                   for vf in self.files])
