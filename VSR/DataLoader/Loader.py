"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: April 2nd 2019

Load frames with specified filter in given directories,
and provide inheritable API for specific loaders.

changelog 2019-4-2
- Introduce `parser` to deal with more & more complex data distribution

changelog 2018-8-29
- Added BasicLoader and QuickLoader (multiprocessor loader)
- Deprecated BatchLoader (and Loader)
"""

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/4 下午2:42

import importlib
import threading as th

import numpy as np
from psutil import virtual_memory

from . import _logger
from ..Util import Utility
from ..Util.Config import Config
from ..Util.ImageProcess import crop


def _augment(image, op):
  """Image augmentation"""
  if op[0]:
    image = np.rot90(image, 1)
  if op[1]:
    image = np.fliplr(image)
  if op[2]:
    image = np.flipud(image)
  return image


class EpochIterator:
  """An iterator for generating batch data in one epoch

  Args:
      loader: A `BasicLoader` or `QuickLoader` to provide properties.
      grids: A list of tuple, commonly returned from
        `BasicLoader._generate_crop_grid`.
  """

  def __init__(self, loader, grids):
    self.batch = loader.batch
    self.scale = loader.scale
    self.aug = loader.aug
    self.loader = loader
    self.grids = grids

  def __len__(self):
    t = len(self.grids)
    b = self.batch
    return t // b + int(np.ceil((t % b) / b))

  def __iter__(self):
    return self

  def __next__(self):
    batch_hr, batch_lr, batch_name = [], [], []
    if not self.grids:
      raise StopIteration

    while self.grids and len(batch_hr) < self.batch:
      hr, lr, box, name = self.grids.pop(0)
      box = np.array(box, 'int32')
      box_lr = box // [*self.scale, *self.scale]
      # if self.loader.method == 'train':
      #   assert (np.mod(box, [*self.scale, *self.scale]) == 0).all()
      crop_hr = [crop(img, box) for img in hr]
      crop_lr = [crop(img, box_lr) for img in lr]
      ops = np.random.randint(0, 2, [3]) if self.aug else [0, 0, 0]
      clip_hr = [_augment(img, ops) for img in crop_hr]
      clip_lr = [_augment(img, ops) for img in crop_lr]
      batch_hr.append(np.stack(clip_hr))
      batch_lr.append(np.stack(clip_lr))
      batch_name.append(name)

    if batch_hr and batch_lr and batch_name:
      try:
        batch_hr = np.squeeze(np.stack(batch_hr), 1)
        batch_lr = np.squeeze(np.stack(batch_lr), 1)
      except ValueError:
        # squeeze error
        batch_hr = np.stack(batch_hr)
        batch_lr = np.stack(batch_lr)
      batch_name = np.stack(batch_name)

    if np.ndim(batch_hr) == 3:
      batch_hr = np.expand_dims(batch_hr, -1)
    if np.ndim(batch_lr) == 3:
      batch_lr = np.expand_dims(batch_lr, -1)

    return batch_hr, batch_lr, batch_name, batch_lr


class BasicLoader:
  """Basic loader in single thread

  Args:
      dataset: A `Dataset` to load by this loader.
      method: A string in ('train', 'val', 'test') specifies which subset to
        use in the dataset. Also 'train' set will shuffle buffers each epoch.
      config: A `Config` class, including 'batch', 'depth', 'patch_size',
        'scale', 'steps_per_epoch' and 'convert_to' arguments.
      augmentation: A boolean to specify whether call `_augment` to batches.
        `_augment` will randomly flip or rotate images.
      kwargs: override config key-values.
  """

  def __init__(self, dataset, method, config, augmentation=False, **kwargs):
    config = self._parse_config(config, **kwargs)
    config.method = method.lower()
    parser = dataset.get('parser', 'default_parser')
    _logger.debug(f"Parser: [{parser}]")
    try:
      _m = importlib.import_module(parser)
    except ImportError:
      _m = importlib.import_module(f'.{parser}', 'VSR.DataLoader.Parser')
    self.parser = _m.Parser(dataset, config)
    self.aug = augmentation
    if hasattr(self.parser, 'color_format'):
      self.color_format = self.parser.color_format
    else:
      self.color_format = 'RGB'
    # self.pair = getattr(dataset, '{}_pair'.format(method))
    self.loaded = 0
    self.frames = []  # a list of tuple represents (HR, LR, name) of a clip

  def _parse_config(self, config: Config, **kwargs):
    _config = Config(config)
    _config.update(kwargs)
    _needed_args = ('batch', 'depth', 'scale',
                    'steps_per_epoch', 'convert_to', 'modcrop')
    for _arg in _needed_args:
      # Set default and check values
      if _arg not in _config:
        if _arg in ('batch', 'scale'):
          raise ValueError(_arg + ' is required in config.')
        elif _arg == 'depth':
          _config.depth = 1
        elif _arg == 'steps_per_epoch':
          _config.steps_per_epoch = -1
        elif _arg == 'convert_to':
          _config.convert_to = 'RGB'
        elif _arg == 'modcrop':
          _config.modcrop = True
    self.depth = _config.depth
    self.patch_size = _config.patch_size
    self.scale = Utility.to_list(_config.scale, 2)
    self.patches_per_epoch = _config.steps_per_epoch * _config.batch
    self.batch = _config.batch
    self.crop = _config.crop
    self.modcrop = _config.modcrop
    self.resample = _config.resample
    return _config

  def _generate_crop_grid(self, frames, size, shuffle=False):
    """generate randomly cropped box of `frames`

    Args:
        frames: a list of tuple, commonly returned from `_process_at_file`.
        size: an int scalar to specify number of generated crops.
        shuffle: a boolean, whether to shuffle the outputs.

    Return:
        list of tuple: containing (HR, LR, box, name) respectively,
          where HR and LR are reference frames, box is a list of 4
          int of crop coordinates.
    """
    if not frames:
      _logger.warning('frames is empty. [size={}]'.format(size))
      return []
    patch_size = Utility.to_list(self.patch_size, 2)
    patch_size = Utility.shrink_mod_scale(patch_size, self.scale)
    if size < 0:
      index = np.arange(len(frames)).tolist()
      size = len(frames)
    else:
      if self.crop == 'random':
        index = np.random.randint(len(frames), size=size).tolist()
      else:
        index = np.arange(size).tolist()
    grids = []
    for i, (hr, lr, name) in enumerate(frames):
      _w, _h = hr[0].width, hr[0].height
      if self.crop in ('not', 'none') or self.crop is None:
        _pw, _ph = _w, _h
      else:
        _pw, _ph = patch_size
      amount = index.count(i)
      if self.crop == 'random':
        x = np.random.randint(0, _w - _pw + 1, size=amount)
        y = np.random.randint(0, _h - _ph + 1, size=amount)
      elif self.crop == 'center':
        x = np.array([(_w - _pw) // 2] * amount)
        y = np.array([(_h - _ph) // 2] * amount)
      elif self.crop == 'stride':
        _x = np.arange(0, _w - _pw + 1, _pw)
        _y = np.arange(0, _h - _ph + 1, _ph)
        x, y = np.meshgrid(_x, _y)
        x = x.flatten()
        y = y.flatten()
      else:
        x = np.zeros([amount])
        y = np.zeros([amount])
      x -= x % self.scale[0]
      y -= y % self.scale[1]
      grids += [(hr, lr, [_x, _y, _x + _pw, _y + _ph], name)
                for _x, _y in zip(x, y)]
    if shuffle:
      np.random.shuffle(grids)
    return grids[:size]

  def _prefetch(self, memory_usage=None, shard=1, index=0):
    """Prefetch `size` files and load into memory. Specify `shard` will
    divide loading files into `shard` shards in order to execute in
    parallel.

    NOTE: parallelism is implemented via `QuickLoader`

    Args:
      memory_usage: desired virtual memory to use, could be int (bytes) or
        a readable string ('3GB', '1TB'). Default to use all available
        memories.
      shard: an int scalar to specify the number of shards operating in
        parallel.
      index: an int scalar, representing shard index
    """

    if self.loaded & (1 << index):
      return
    # check memory usage
    if isinstance(memory_usage, str):
      memory_usage = Utility.str_to_bytes(memory_usage)
    free_memory = virtual_memory().available
    if not memory_usage:
      memory_usage = free_memory
    memory_usage = np.min(
      [np.uint64(memory_usage), free_memory])
    if hasattr(self.parser, 'capacity'):
      cap = self.parser.capacity
    else:
      cap = -1
    if cap <= memory_usage:
      _logger.debug("Load all data into memory.")
      # load all clips
      interval = int(np.ceil(len(self.parser) / shard))
      if index == shard - 1:
        frames = self.parser[index * interval:]
      else:
        frames = self.parser[index * interval:(index + 1) * interval]
      self.frames += frames
      self.loaded |= (1 << index)
    else:
      scale_factor = 0.9
      prop = memory_usage / cap / shard * scale_factor
      _logger.debug(f"Load {prop * 100:.1f}% data into memory.")
      # How many frames can be read into memory each thread each epoch
      # Note: we assume each "frame" has a close size.
      n = max(1, int(np.round(len(self.parser) * prop)))  # at least 1 sample
      frames = []
      for i in np.random.permutation(len(self.parser))[:n]:
        frames += self.parser[i]
      self.frames += frames

  def make_one_shot_iterator(self, memory_usage=None, shuffle=False):
    """make an `EpochIterator` to enumerate batches of the dataset

    Args:
        memory_usage: desired virtual memory to use, could be int (bytes) or
          a readable string ('3GB', '1TB'). Default to use all available
          memories.
        shuffle: A boolean whether to shuffle the patch grids.

    Returns:
        An EpochIterator
    """
    self._prefetch(memory_usage, 1, 0)
    grids = self._generate_crop_grid(self.frames, self.patches_per_epoch,
                                     shuffle=shuffle)
    if not (self.loaded == 1):
      self.frames.clear()
    return EpochIterator(self, grids)


class QuickLoader(BasicLoader):
  """Async data loader with high efficiency.

  `QuickLoader` concurrently pre-fetches clips into memory every n iterations,
  and provides several methods to select clips. `QuickLoader` won't loads all
  files in the dataset into memory if your memory isn't enough.

  NOTE: A clip is a bunch of consecutive frames, which can represent either a
  dynamic video or single image.

  Args:
      dataset: A `Dataset` to load by this loader.
      method: A string in ('train', 'val', 'test') specifies which subset to
        use in the dataset. Also 'train' set will shuffle buffers each epoch.
      config: A `Config` class, including 'batch', 'depth', 'patch_size',
        'scale', 'steps_per_epoch' and 'convert_to' arguments.
      augmentation: A boolean to specify whether call `_augment` to batches.
        `_augment` will randomly flip or rotate images.
      n_threads: number of threads to load dataset
      kwargs: override config key-values.
  """

  def __init__(self, dataset, method, config, augmentation=False, n_threads=1,
               **kwargs):

    self.shard = n_threads
    self.threads = []
    super(QuickLoader, self).__init__(dataset, method, config,
                                      augmentation, **kwargs)

  def prefetch(self, memory_usage=None):
    """Prefetch data.

    This call will spawn threads of `_prefetch` and returns immediately.
    The next call of `make_one_shot_iterator` will join all the threads.
    If this is not called in advance, data will be fetched at
    `make_one_shot_iterator`.

    Args:
        memory_usage: desired virtual memory to use, could be int (bytes) or
          a readable string ('3GB', '1TB'). Default to use all available
          memories.

    Note: call `prefetch` twice w/o `make_one_shot_iterator` is
      undefined behaviour.
    """

    for i in range(self.shard):
      t = th.Thread(target=self._prefetch,
                    args=(memory_usage, self.shard, i),
                    name='fetch_thread_{}'.format(i))
      t.start()
      self.threads.append(t)

  def make_one_shot_iterator(self, memory_usage=None, shuffle=False):
    """make an `EpochIterator` to enumerate batches of the dataset. Specify
    `shard` will divide loading files into `shard` shards in order to
    execute in parallel.

    Will create TFIterator if use TFRecordDataset.

    Args:
        memory_usage: desired virtual memory to use, could be int (bytes) or
          a readable string ('3GB', '1TB'). Default to use all available
          memories.
        shuffle: A boolean whether to shuffle the patch grids.

    Return:
        An EpochIterator or TFIterator

    Known issues:
        If data of either shard is too large (i.e. use 1 shard and total
        frames is around 6GB in my machine), windows Pipe may broke and
        `get()` never returns.
    """

    if not self.threads:
      self.prefetch(memory_usage)
    for t in self.threads:
      t.join()
    self.threads.clear()
    # reduce
    grids = self._generate_crop_grid(self.frames,
                                     self.patches_per_epoch,
                                     shuffle=shuffle)
    if not (self.loaded & 0xFFFF):
      self.frames.clear()
    return EpochIterator(self, grids)
