#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import numpy as np

from . import _logger
from ..VirtualFile import ImageFile


class Parser(object):
  def __init__(self, dataset, config):
    urls = sorted(dataset.get(config.method, []))
    flow = sorted(dataset['flow'])
    self.files = [ImageFile(fp).attach_flow(f) for fp, f in zip(urls, flow)]
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

  def __getitem__(self, index):
    if isinstance(index, slice):
      ret = []
      for vf in self.files[index]:
        ret += self.gen_frames(vf)
      return ret
    else:
      vf = self.files[index]
      return self.gen_frames(vf)

  def __len__(self):
    return len(self.files)

  def gen_frames(self, vf):
    assert isinstance(vf, ImageFile)

    _logger.debug('Prefetching ' + vf.name)
    vf.reopen()
    img = [x for x in vf.read_frame(2)]
    img = [x.convert(self.color_format) for x in img]
    frames = [(img, [vf.flow], (vf.name, 0, vf.frames))]
    return frames

  @property
  def capacity(self):
    bpp = 14  # bytes per pixel
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                   for vf in self.files])
