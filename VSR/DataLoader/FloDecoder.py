#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7
import logging
from importlib import import_module

import numpy as np


def open_flo(fn):
  """ Read .flo file in Middlebury format
  # Code adapted from:
  # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

  # WARNING: this will work on little-endian architectures (eg Intel x86) only!
  # print 'fn = %s'%(fn)
  """
  with open(fn, 'rb') as f:
    magic = np.fromfile(f, np.float32, count=1)
    if 202021.25 != magic:
      logging.error('Magic number incorrect. Invalid .flo file')
      return None
    else:
      w = np.fromfile(f, np.int32, count=1)[0]
      h = np.fromfile(f, np.int32, count=1)[0]
      # print 'Reading %d x %d flo file\n' % (w, h)
      data = np.fromfile(f, np.float32, count=2 * w * h)
      # Reshape data into 3D array (columns, rows, bands)
      # The reshape here is for visualization, the original code is (w,h,2)
      return np.resize(data, (int(h), int(w), 2))


def write_flo(filename, uv, v=None):
  """ Write optical flow to file.

  Original code by Deqing Sun, adapted from Daniel Scharstein.
  """
  n_bands = 2
  _TAG_CHAR = np.array([202021.25], np.float32)

  if v is None:
    u = uv[..., 0]
    v = uv[..., 1]
  else:
    u = uv
  height, width = u.shape
  with open(filename, 'wb') as f:
    # write the header
    f.write(_TAG_CHAR.tobytes())
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * n_bands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)


class KITTI:
  @staticmethod
  def open_png16(fn):
    """Read 16bit png file"""

    png = import_module('png')
    reader = png.Reader(fn)
    data = reader.asDirect()
    pixels = []
    for row in data[2]:
      row = np.reshape(np.asarray(row), [-1, 3])
      pixels += [row]
    return np.stack(pixels, 0)

  @staticmethod
  def open_flow(fn):
    flow = KITTI.open_png16(fn)
    valid = flow[..., -1]
    u = flow[..., 0].astype('float32')
    v = flow[..., 1].astype('float32')
    u = (u - 2 ** 15) / 64 * valid
    v = (v - 2 ** 15) / 64 * valid
    return np.stack([u, v], -1)
