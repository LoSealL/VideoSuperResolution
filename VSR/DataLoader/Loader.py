#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import logging
from concurrent import futures

import numpy as np
from PIL import Image
from psutil import virtual_memory

from .Crop import RandomCrop
from .Dataset import Container, Dataset
from .Transform import Bicubic, Tidy
from ..Backend import DATA_FORMAT
from ..Util import Utility
from ..Util.ImageProcess import img_to_array

FREE_MEMORY = virtual_memory().available * 0.5
LOG = logging.getLogger('VSR.Loader')


def _augment(image, op):
  """Image augmentation"""
  assert image.ndim == 4
  if op[0]:
    if DATA_FORMAT == 'channels_last':
      image = np.rot90(image, 1, axes=(1, 2))
    else:
      image = np.rot90(image, 1, axes=(2, 3))
  if op[1]:
    if DATA_FORMAT == 'channels_last':
      image = image[:, :, ::-1]
    else:
      image = image[..., ::-1]
  if op[2]:
    if DATA_FORMAT == 'channels_last':
      image = image[:, ::-1]
    else:
      image = image[:, :, ::-1]
  return image


class EpochIterator:
  """An iterator for generating batch data in one epoch

  Args:
      loader: A `Loader` object to provide properties.
      shape: The shape of the generated batch, 5-D requested [N, T, C, H, W].
      steps: The number of batches to generate in one epoch.
      shuffle: A boolean representing whether to shuffle the dataset.

  Note:
      The rules for -1 shape:
      - If shape[1] is -1, which represents the temporal length,
        will generate the entire video clips;
      - if shape[2] is -1, will auto choose the channel number according to
        color format;
      - if shape[3] or shape[4] is -1, will deduce the frame height or width
        to replace the value.

      The rules for steps:
      - If the `steps` is -1, will generate batches in sequential order;
  """

  def __init__(self, loader, shape, steps, shuffle=None):
    self.loader = loader
    self.shape = shape
    if steps <= 0:
      t = len(self.loader.data['hr'])
      b = self.shape[0]
      self.steps = t // b + int(np.ceil((t % b) / b))
    else:
      self.steps = steps
    self.count = 0
    if shuffle:
      mmax = len(self.loader.data['hr'])
      if mmax > 0:
        self.index = np.random.randint(mmax, size=self.steps * shape[0])
      else:
        self.steps = 0
    else:
      self.index = np.arange(self.steps * shape[0])

  def __len__(self):
    return self.steps

  def __iter__(self):
    return self

  def __next__(self):
    pack = {'hr': [], 'lr': [], 'name': []}
    if self.count >= self.steps:
      raise StopIteration("All batch data generated.")

    slc = slice(self.count * self.shape[0], (self.count + 1) * self.shape[0])
    crop = self.loader.crop
    cb_hr = (self.loader.hr['transform1'], self.loader.hr['transform2'])
    cb_lr = (self.loader.lr['transform1'], self.loader.lr['transform2'])
    for i in self.index[slc]:
      if i >= len(self.loader.data['hr']):
        continue
      hr = self.loader.data['hr'][i]
      lr = self.loader.data['lr'][i]
      name = self.loader.data['names'][i]
      hr2 = hr
      for fn in cb_hr[0]:
        hr2 = [fn(img) for img in hr2]
      hr2 = [img.convert(self.loader.hr['color']) for img in hr2]
      lr2 = lr
      for fn in cb_lr[0]:
        lr2 = [fn(img) for img in lr2]
      lr2 = [img.convert(self.loader.lr['color']) for img in lr2]
      hr3 = np.stack([img_to_array(img, DATA_FORMAT) for img in hr2])
      lr3 = np.stack([img_to_array(img, DATA_FORMAT) for img in lr2])
      del hr2, lr2
      if hr3.shape[0] == 1 and lr3.shape[0] == 1:
        hr3 = hr3.squeeze(0)
        lr3 = lr3.squeeze(0)
        hr4, lr4 = crop((hr3, lr3), shape=self.shape[2:]) if crop else (
          hr3, lr3)
      else:
        hr4, lr4 = crop((hr3, lr3), shape=self.shape[1:]) if crop else (
          hr3, lr3)
      del hr3, lr3
      hr4 = np.expand_dims(hr4, 0)  # 4-D or 5-D
      lr4 = np.expand_dims(lr4, 0)  # [1, (T,) C, H, W]
      for fn in cb_hr[1]:
        hr4 = fn(hr4)
      for fn in cb_lr[1]:
        lr4 = fn(lr4)

      if self.loader.aux['augmentation']:
        ops = np.random.randint(0, 2, [3])
      else:
        ops = [0, 0, 0]
      _shape0 = hr4.shape
      _shape1 = lr4.shape
      hr5 = _augment(hr4.reshape([-1, *_shape0[-3:]]), ops)
      lr5 = _augment(lr4.reshape([-1, *_shape1[-3:]]), ops)
      del hr4, lr4
      pack['hr'].append(hr5.reshape(_shape0))
      pack['lr'].append(lr5.reshape(_shape1))
      pack['name'].append(name)

    if pack['hr']:
      pack['hr'] = np.concatenate(pack['hr'])
    if pack['lr']:
      pack['lr'] = np.concatenate(pack['lr'])
    self.count += 1
    return pack


class Loader(object):
  """A parallel data loader that generates label and data batches each epoch.

  Args:
      hr_data: this is a data container from `Dataset` object, represents the
               label (high-resolution) data.
      lr_data: this is a data container from `Dataset` object, represents the
               training (low-resolution) data.
      scale: specify the scale factor for this model. If scale is not specified,
             you must set "lr_data", and "cropper" explicitly.
      extra_data: a dict object contains extra information. You won't use this.
      threads: num of threads used to load the data.

  Note:
      A `Loader` object has several attributes to enhance the loader's ability.
      See `Loader.add_data_transform`, `Loader.image_augmentation`,
      `Loader.cropper`, `Loader.set_color_space` for details.
  """

  def __init__(self, hr_data, lr_data=None, scale=None, extra_data: dict = None,
               threads=1):
    # check type
    if isinstance(hr_data, Dataset):
      hr_data = hr_data.compile()
      assert isinstance(hr_data, Container)
    if isinstance(lr_data, Dataset):
      lr_data = lr_data.compile()
      assert isinstance(lr_data, Container)
    if lr_data is not None and hr_data is not None:
      assert len(hr_data) == len(lr_data)
    else:
      hr_data = hr_data or lr_data
      lr_data = lr_data or hr_data
    if hr_data is None and lr_data is None:
      hr_data = lr_data = Container([], False)
    scale = scale or 1  # deduce to 1
    if extra_data is not None:
      assert isinstance(extra_data, dict)

    # default params
    self.hr = {
      'data': hr_data,
      'transform1': [],
      'transform2': [],
      'color': 'RGB'
    }
    self.lr = {
      'data': lr_data,
      'transform1': [],
      'transform2': [],
      'color': 'RGB'
    }
    self.aux = {
      'augmentation': False,
      'scale': scale,
      'fetchList': list(np.arange(len(hr_data)))
    }
    self.data = {
      'hr': [],
      'lr': [],
      'names': [],
      'extra': []
    }
    self.cache = {
      'hr': [],
      'lr': [],
      'names': [],
      'extra': []
    }
    self.extra = extra_data or {}
    self.crop = None
    self.threads = threads
    self.thp = futures.ThreadPoolExecutor(max_workers=threads)
    self.fs = []
    self.loaded = 0
    if self.hr['data'] is self.lr['data']:
      cap = self.hr['data'].capacity
    else:
      cap = self.hr['data'].capacity + self.lr['data'].capacity
    if self.extra and isinstance(self.extra['data'], Container):
      cap += self.extra['data'].capacity
    self.aux['cap'] = cap  # estimated memory usage in bytes
    if hr_data is lr_data and scale > 1:
      self.add_data_transform('hr', Tidy(scale))
      self.add_data_transform('lr', Tidy(scale), Bicubic(1 / scale))

  def add_data_transform(self, target: str, *fn, dtype='pillow'):
    """Add data transform functions. Each function will be called before
    generating batch data.

    Args:
        target: either "hr" or "lr", specify which data to apply.
        *fn: functions with only one argument, the type of the argument will
             be specified through `dtype`.
        dtype: specify the type of the function's argument.

    Note:
        `dtype` supports `numpy.ndarray` and `PIL.Image.Image`
    """
    assert target.lower() in ('hr', 'lr')
    if isinstance(dtype, Image.Image):
      dtype = 'pillow'
    elif isinstance(dtype, np.ndarray):
      dtype = 'numpy'
    assert dtype.lower() in ('pillow', 'numpy', 'pil', 'np')
    fn = filter(callable, fn)
    if dtype.lower() in ('pillow', 'pil'):
      getattr(self, target.lower())['transform1'] += list(fn)
    else:
      getattr(self, target.lower())['transform2'] += list(fn)

  def image_augmentation(self):
    """Enable data augmentation

    The data augmentation for single image will randomly rotate and flip the
    image.
    The data augmentation for video will randomly rotate, flip and revert the
    video.

    TODO: revert is not implemented.
    """
    self.aux['augmentation'] = True

  def cropper(self, fn):
    assert callable(fn)
    self.crop = fn

  def set_color_space(self, target: str, mode: str):
    if not mode.upper() in ('RGB', 'L', 'YCbCr', 'Gray'):
      raise ValueError(f"Invalid mode: {mode}, must be RGB | L | YCbCr | Gray")
    assert target.lower() in ('hr', 'lr')
    getattr(self, target.lower()).update(color=mode)

  def make_one_shot_iterator(self, batch_shape, steps, shuffle=None,
                             memory_limit=None):
    """Make an iterator object to generate batch data for models.

    Args:
        batch_shape: The shape of batch to generate.
        steps: The number of batches to generate in one epoch.
        shuffle: A boolean representing whether to shuffle the dataset.
        memory_limit: the maximum system memory to use. (Not GPU memory!!)

    Note:
        The rules for -1 shape:
        - If shape[1] is -1, which represents the temporal length,
          will generate the entire video clips;
        - if shape[2] is -1, will auto choose the channel number according to
          color format;
        - if shape[3] or shape[4] is -1, will deduce the frame height or width
          to replace the value.

        The rules for steps:
        - If the `steps` is -1, will generate batches in sequential order;
        - If the `steps` is an positive integer, the generated batches are
          randomly shuffled.
    """
    shape = list(batch_shape)
    if len(shape) == 4:
      shape.insert(1, 1)
    assert len(shape) is 5
    if shape[-2] != -1 and self.crop is None:
      self.cropper(RandomCrop(self.aux['scale']))
    if isinstance(memory_limit, str):
      memory_limit = Utility.str_to_bytes(memory_limit)
    self.prefetch(shuffle, memory_limit)
    futures.as_completed(self.fs)
    for fs in self.fs:
      if fs.exception():
        raise fs.exception()
      assert fs.done()
    self.fs.clear()
    if not (self.loaded & int(2 ** self.threads - 1)):
      self.data, self.cache = self.cache, self.data
      [self.cache[k].clear() for k in self.cache]
      loaded = self.loaded >> (self.threads * 2)
      if not shuffle:
        loaded += 1  # move to next chunk
        if loaded >= self.aux['cap'] / memory_limit:
          loaded = 0
      self.loaded = loaded << (self.threads * 2)
    return EpochIterator(self, shape, steps, shuffle)

  def prefetch(self, shuffle=None, memory_usage=None):
    # check memory usage
    if isinstance(memory_usage, str):
      memory_usage = Utility.str_to_bytes(memory_usage)
    if not memory_usage:
      memory_usage = FREE_MEMORY
    available_mem = min([np.uint64(memory_usage), np.uint64(FREE_MEMORY)])
    if not self.fs:
      if shuffle:
        self.aux['fetchList'] = list(
            np.random.permutation(self.aux['fetchList']))
      if available_mem > self.aux['cap']:
        LOG.debug("Loading all data into memory.")
        for i in range(self.threads):
          if self.loaded & (1 << i):
            continue
          self.fs.append(self.thp.submit(self._prefetch_all, i))
      else:
        prop = memory_usage / self.aux['cap'] / self.threads
        # How many frames can be read into memory each thread each epoch
        # Note: we assume each "frame" has a close size.
        n = max(1, int(np.round(len(self.hr['data']) * prop)))
        LOG.debug(f"Loading {prop * self.threads * 100:.4f}% data.")
        [self.fs.append(self.thp.submit(self._prefecth_chunk, n, i)) for i in
         range(self.threads)]

  def _prefetch_all(self, index):
    length = len(self.hr['data'])
    # load all clips
    interval = int(np.ceil(length / self.threads))
    frames = []
    names = []
    for img in self.hr['data'][index * interval:(index + 1) * interval]:
      frames.append(img.read_frame(img.frames))
      names.append(img.name)
    self.data['hr'] += frames
    self.data['names'] += names
    if self.hr['data'] is self.lr['data']:
      self.data['lr'] += frames
    else:
      frames = []
      for img in self.lr['data'][index * interval:(index + 1) * interval]:
        frames.append(img.read_frame(img.frames))
      self.data['lr'] += frames
    if self.extra and isinstance(self.extra['data'], Container):
      frames = []
      for img in self.extra['data'][index * interval:(index + 1) * interval]:
        frames.append(img.read_frame(img.frames))
      self.data['extra'] += frames
    self.loaded |= (1 << index)

  def _prefecth_chunk(self, chunk_size, index):
    loaded = self.loaded >> (self.threads * 2)
    n = chunk_size
    st = n * self.threads * loaded  # start chunk
    for i in self.aux['fetchList'][st + n * index:st + n * (index + 1)]:
      img = self.hr['data'][i]
      self.cache['hr'].append(img.read_frame(img.frames))
      img.reopen()
      self.cache['names'].append(img.name)
      if self.hr['data'] is self.lr['data']:
        self.cache['lr'].append(self.cache['hr'][-1])
      else:
        img = self.lr['data'][i]
        self.cache['lr'].append(img.read_frame(img.frames))
        img.reopen()
      if self.extra and isinstance(self.extra['data'], Container):
        img = self.extra['data'][i]
        self.cache['extra'].append(img.read_frame(img.frames))
        img.reopen()
    loaded <<= self.threads
    loaded |= (1 << index)
    self.loaded = loaded << self.threads
