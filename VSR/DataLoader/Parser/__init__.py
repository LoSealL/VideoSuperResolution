#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import logging

import numpy as np

_logger = logging.getLogger('VSR.Loader.Parser')

# each file is selected equally
_EQUAL_FILE = 0
# each pixel is selected equally, that is,
# a larger image has a higher probability to
# be selected, and vice versa
_EQUAL_PIXEL = 1


def file_weights(vfs, method=_EQUAL_PIXEL):
  """Get probability for selecting each file object.

  Args:
    vfs: a list of `VirtualFile`
    method: We offer two method, see `Select` for details.
  """
  weights = []
  for f in vfs:
    if method == _EQUAL_PIXEL:
      weights += [np.prod(f.shape) * f.frames]
    elif method == _EQUAL_FILE:
      weights += [1]
    else:
      raise ValueError('unknown select method ' + str(method))
  prob = np.array(weights, 'float32') / np.sum(weights, dtype='float32')
  prob = np.cumsum(prob)
  return prob


def random_select(vfs, probability, n_frames):
  """Randomly select `size` file objects

  Args:
    vfs: a list of `VirtualFile`
    probability: probability weights of each file
    n_frames: the number of files to select
  Return:
    Dict: map file objects to its select quantity.
  """
  # Q: Is `s` relevant to poisson dist.?
  s = {f: 0 for f in vfs}
  for x in np.random.rand(n_frames).tolist():
    x *= np.ones_like(probability)
    diff = np.array(probability >= x)
    index = diff.nonzero()[0].tolist()
    if index:
      index = index[0]
    else:
      index = 0
    s[vfs[index]] += 1
  return s


def parse_index(index, frames):
  """generate (video-id, frame-id), where video-id is the order of videos, and
    frame-id is the order of frames of each video.

  Args:
    index: an integer, representing an order from **ALL** frames;
    frames: a list of integers, representing total frame numbers for each video

  I.E:
  >>> parse_index(1, [3, 4, 5]) -> (0, 1)  # the 2nd frame of video 0
  >>> parse_index(7, [3, 4, 5]) -> (2, 0)  # the 1st frame of video 2
  """
  for key, i in enumerate(np.cumsum(frames)):
    if index < i:
      seq = (index - np.sum(frames[:key])) % frames[key]
      break
  return key, int(seq)
