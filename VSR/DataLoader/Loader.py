"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: Jan 9th 2019

Load frames with specified filter in given directories,
and provide inheritable API for specific loaders.

changelog 2019-1-9
- Add TFRecordDataset support. (Under test)

changelog 2018-8-29
- Added BasicLoader and QuickLoader (multiprocessor loader)
- Deprecated BatchLoader (and Loader)
"""

import copy
import threading as th

import numpy as np
import tensorflow as tf
from psutil import virtual_memory

from .VirtualFile import ImageFile, RawFile, _ALLOWED_RAW_FORMAT
from ..Util import Utility
from ..Util.Config import Config
from ..Util.ImageProcess import (array_to_img, crop, imresize,
                                 shrink_to_multiple_scale)

_TFRECORD_SUFFIX = '.tfrecords'


def _augment(image, op):
  """Image augmentation"""
  if op[0]:
    image = np.rot90(image, 1)
  if op[1]:
    image = np.fliplr(image)
  if op[2]:
    image = np.flipud(image)
  return image


class Select:
  # each file is selected equally
  EQUAL_FILE = 0
  # each pixel is selected equally, that is,
  # a larger image has a higher probability to
  # be selected, and vice versa
  EQUAL_PIXEL = 1


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
    return len(self.grids) // self.batch

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
      if self.loader.method == 'train':
        assert (np.mod(box, [*self.scale, *self.scale]) == 0).all()
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


class TFIterator:
  def __init__(self, next_element, max_step):
    self.max = max_step
    self.nx = next_element
    self.count = 0

  def __len__(self):
    return self.max

  def __iter__(self):
    return self

  def __next__(self):
    if 0 <= self.max <= self.count:
      raise StopIteration
    self.count += 1
    sess = tf.get_default_session()
    try:
      hr, lr, name, post = sess.run(self.nx)
    except tf.errors.OutOfRangeError:
      raise StopIteration
    name = np.stack([n.decode() for n in name])
    return hr, lr, name, post


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
    self.file_names = dataset.__getattr__(method.lower()) or []
    self.method = method
    self.flow = dataset.flow
    self.pair = dataset.pair
    self.aug = augmentation
    if config.convert_to.lower() in ('gray', 'l'):
      self.color_format = 'L'
    elif config.convert_to.lower() in ('yuv', 'ycbcr'):
      self.color_format = 'YCbCr'
    elif config.convert_to.lower() in ('rgb',):
      self.color_format = 'RGB'
    else:
      tf.logging.warning('Use grayscale by default. '
                         'Unknown format {}'.format(config.convert_to))
      self.color_format = 'L'
    self.loaded = 0
    self.free_memory_on_start = virtual_memory().free
    self.frames = []  # a list of tuple represents (HR, LR, name) of a clip
    self.prob = self._read_file(dataset)._calc_select_prob()

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

  def _read_file(self, dataset):
    """Initialize all `File` objects"""
    if dataset.mode.lower() == 'pil-image1':
      if self.flow:
        # map flow
        flow = sorted(self.flow)
        files = sorted(self.file_names)
        self.file_objects = [ImageFile(fp).attach_flow(f) for fp, f in
                             zip(files, flow)]
      elif self.pair:
        pair = sorted(self.pair)
        files = sorted(self.file_names)
        self.file_objects = [ImageFile(fp).attach_flow(p) for fp, p in
                             zip(files, pair)]
      else:
        self.file_objects = [ImageFile(fp) for fp in self.file_names]
    elif dataset.mode.upper() in _ALLOWED_RAW_FORMAT:
      self.file_objects = [
        RawFile(fp, dataset.mode, (dataset.width, dataset.height))
        for fp in self.file_names]
    elif dataset.mode.lower() == 'numpy':
      """already loaded numpy array, in case anyone want to use 
      external loaders, data can be a 4-D or 5-D ndarray"""
      tf.logging.debug('reading numpy array')
      param = dataset.numpy
      if param.exec is not None:
        exec(param.exec)
      src = param.get(self.method)
      if src:
        data = eval(param.get(self.method))
      else:
        data = None
      if isinstance(data, np.ndarray):
        for i, hr in enumerate(data):
          if hr.ndim == 3:
            frames_hr = [array_to_img(hr, 'RGB')]
          else:
            frames_hr = [array_to_img(x, 'RGB') for x in hr]
          frames_lr = [imresize(img,
                                np.reciprocal(self.scale,
                                              dtype='float32'),
                                resample=self.resample)
                       for img in frames_hr]
          frames_hr = [img.convert(self.color_format) for img in
                       frames_hr]
          frames_lr = [img.convert(self.color_format) for img in
                       frames_lr]
          name = (dataset.name, i, len(frames_hr))
          self.frames.append((frames_hr, frames_lr, name))
      self.loaded = 1
      self.file_objects = []
    return self

  def _calc_select_prob(self, method=Select.EQUAL_PIXEL):
    """Get probability for selecting each file object.

    Args:
        method: We offer two method, see `Select` for details.
    """
    weights = []
    for f in self.file_objects:
      if method == Select.EQUAL_PIXEL:
        weights += [np.prod(f.shape) * f.frames]
      elif method == Select.EQUAL_FILE:
        weights += [1]
      else:
        raise ValueError('unknown select method ' + str(method))
    prob = np.array(weights, 'float32') / np.sum(weights, dtype='float32')
    prob = np.cumsum(prob)
    return prob

  def _random_select(self, size, seed=None):
    """Randomly select `size` file objects

    Args:
        size: the number of files to select
        seed: set the random seed (of `numpy.random`)

    Return:
        Dict: map file objects to its select quantity.
    """
    if seed:
      np.random.seed(seed)
    x = np.random.rand(size)
    # Q: Is `s` relevant to poisson dist.?
    s = {f: 0 for f in self.file_objects}
    for _x in x.tolist():
      _x *= np.ones_like(self.prob)
      diff = self.prob >= _x
      index = diff.nonzero()[0].tolist()
      if index:
        index = index[0]
      else:
        index = 0
      s[self.file_objects[index]] += 1
    return s

  def _vf_gen_lr_hr_pair(self, vf, depth, index):
    vf.seek(index)
    frames_hr = [shrink_to_multiple_scale(img, self.scale)
                 if self.modcrop else img for img in vf.read_frame(depth)]
    frames_lr = [imresize(img,
                          np.reciprocal(self.scale, dtype='float32'),
                          resample=self.resample)
                 for img in frames_hr]
    frames_hr = [img.convert(self.color_format) for img in frames_hr]
    frames_lr = [img.convert(self.color_format) for img in frames_lr]
    return frames_hr, frames_lr, (vf.name, index, vf.frames)

  def _vf_gen_flow_img_pair(self, vf, depth, index):
    assert depth == 2 and index == 0
    img = [img for img in vf.read_frame(depth)]
    img = [i.convert(self.color_format) for i in img]
    return img, [vf.flow], (vf.name, index, vf.frames)

  def _vf_gen_custom_pair(self, vf, depth, index):
    vf.seek(index)
    frames_hr = [img for img in vf.read_frame(depth)]
    frames_lr = [img for img in vf.pair.read_frame(depth)]
    frames_hr = [img.convert(self.color_format) for img in frames_hr]
    frames_lr = [img.convert(self.color_format) for img in frames_lr]
    return frames_hr, frames_lr, (vf.name, index, vf.frames)

  def _process_at_file(self, vf, clips=1):
    """load frames of `File` into memory, crop and generate corresponded
     LR frames.

    Args:
        vf: A `File` object.
        clips: an integer to specify how many clips to generate from `vf`.

    Return:
        List of Tuple: containing (HR, LR, name) respectively
    """
    assert isinstance(vf, (RawFile, ImageFile))

    tf.logging.debug('Prefetching ' + vf.name)
    depth = self.depth
    # read all frames if depth is set to -1
    if depth == -1:
      depth = vf.frames
    index = np.arange(0, vf.frames - depth + 1)
    np.random.shuffle(index)
    frames = []
    for i in index[:clips]:
      if self.flow:
        frames.append(self._vf_gen_flow_img_pair(vf, depth, i))
      elif self.pair:
        frames.append(self._vf_gen_custom_pair(vf, depth, i))
      else:
        frames.append(self._vf_gen_lr_hr_pair(vf, depth, i))
    vf.reopen()  # necessary, rewind the read pointer
    return frames

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
      tf.logging.warning('frames is empty. [size={}]'.format(size))
      return []
    patch_size = Utility.to_list(self.patch_size, 2)
    patch_size = Utility.shrink_mod_scale(patch_size, self.scale)
    if size < 0:
      index = np.arange(len(frames)).tolist()
    else:
      if self.crop == 'random':
        index = np.random.randint(len(frames), size=size).tolist()
      else:
        index = np.arange(size).tolist()
    grids = []
    for i in range(len(frames)):
      hr, lr, name = frames[i]
      _w, _h = hr[0].width, hr[0].height
      if self.crop == 'not' or self.crop is None:
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
      else:
        x = np.zeros([amount])
        y = np.zeros([amount])
      x -= x % self.scale[0]
      y -= y % self.scale[1]
      grids += [(hr, lr, [_x, _y, _x + _pw, _y + _ph], name)
                for _x, _y in zip(x, y)]
    if shuffle:
      np.random.shuffle(grids)
    return grids

  @property
  def size(self):
    """expected total memory usage of the loader"""
    bpp = 3  # bytes per pixel
    if self.flow:
      bpp += 8  # two more float channel
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                   for vf in self.file_objects])

  def __len__(self):
    """length of a BasicLoader is defined as the total frames in Dataset"""
    return np.sum([vf.frames for vf in self.file_objects])

  def change_select_method(self, method):
    """change to different select method, see `Select`"""
    self.prob = self._calc_select_prob(method)
    return self

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
    if not memory_usage:
      memory_usage = self.free_memory_on_start
    memory_usage = np.min(
      [np.uint64(memory_usage), self.free_memory_on_start])
    capacity = self.size
    frames = []
    tf.logging.debug('memory limit: ' + str(memory_usage))
    if capacity <= memory_usage:
      # load all clips
      interval = int(np.ceil(len(self.file_objects) / shard))
      if index == shard - 1:
        for file in self.file_objects[index * interval:]:
          frames += self._process_at_file(file, file.frames)
      else:
        for file in self.file_objects[
                    index * interval:(index + 1) * interval]:
          frames += self._process_at_file(file, file.frames)
      self.frames += frames
      self.loaded |= (1 << index)
    else:
      scale_factor = 0.8
      prop = memory_usage / capacity / shard * scale_factor
      size = int(np.round(len(self) * prop))
      for file, amount in self._random_select(size).items():
        frames += self._process_at_file(copy.deepcopy(file), amount)
      self.frames = frames

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
    return EpochIterator(self, grids)


def _unpack_image_bytes(image_bytes):
  value = tf.parse_single_example(
    image_bytes,
    features={
      "image/hr": tf.FixedLenFeature([], tf.string),
      "image/lr": tf.FixedLenFeature([], tf.string),
      "name": tf.FixedLenFeature([], tf.string),
      "image/post": tf.FixedLenFeature([], tf.string),
    })
  hr = tf.image.decode_image(value["image/hr"])
  lr = tf.image.decode_image(value["image/lr"])
  post = tf.image.decode_image(value["image/post"])
  return hr, lr, value["name"], post


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
    self.is_tfrecord = self._check_tfrecord(dataset, method)
    if self.is_tfrecord:
      # Read TFRecord files
      file_names = dataset.__getattr__(method.lower()) or []
      data = tf.data.TFRecordDataset([str(f) for f in file_names],
                                     num_parallel_reads=n_threads)
      data = data.map(_unpack_image_bytes, n_threads)
      if method == 'train':
        data = data.shuffle(config.batch * config.steps_per_epoch)
        data = data.batch(config.batch, True).repeat(-1)
        self.max_steps = config.steps_per_epoch
      elif method == 'val':
        data = data.batch(config.batch)
        self.max_steps = 1
      else:
        data = data.batch(config.batch)
        self.max_steps = -1
      if config.convert_to.lower() in ('gray', 'l'):
        self.color_format = 'L'
      elif config.convert_to.lower() in ('yuv', 'ycbcr'):
        self.color_format = 'YCbCr'
      elif config.convert_to.lower() in ('rgb',):
        self.color_format = 'RGB'
      else:
        tf.logging.warning('Use grayscale by default. '
                           'Unknown format {}'.format(config.convert_to))
        self.color_format = 'L'
      self._data = data
      self._iter = None
      self._next = None
    else:
      self.shard = n_threads
      self.threads = []
      super(QuickLoader, self).__init__(dataset, method, config,
                                        augmentation, **kwargs)

  def _check_tfrecord(self, dataset, method) -> bool:
    """check files in file list, returns True is all ends with
      `_TFRECORD_SUFFIX`."""
    method = method.lower()
    file_list = getattr(dataset, method, [])
    if not file_list:
      return False
    return np.all([str(f).endswith(_TFRECORD_SUFFIX) for f in file_list])

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

    if self.is_tfrecord:
      return
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

    if self.is_tfrecord:
      if not self._iter and not self._next:
        self._iter = self._data.make_one_shot_iterator()
        self._next = self._iter.get_next()
      return TFIterator(self._next, self.max_steps)
    else:
      if not self.threads:
        self.prefetch(memory_usage)
      for t in self.threads:
        t.join()
      self.threads.clear()
      # reduce
      grids = self._generate_crop_grid(self.frames,
                                       self.patches_per_epoch,
                                       shuffle=shuffle)
      return EpochIterator(self, grids)
