"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 24th 2018

Load frames with specified filter in given directories,
and provide inheritable API for specific loaders.
"""
import numpy as np
import tensorflow as tf
import asyncio

from .Dataset import Dataset
from .VirtualFile import RawFile, ImageFile, _ALLOWED_RAW_FORMAT
from ..Util import ImageProcess, Utility


class Loader(object):
    """Build am iterative loader

      Args:
          dataset: `Dataset` object, see Dataset.py
          method: 'train', 'val', or 'test'
          loop: if True, read data infinitely
    """

    def __init__(self, dataset, method, loop=False):
        if not isinstance(dataset, Dataset):
            raise TypeError('dataset must be Dataset object')

        self.method = method  # train/val/test
        self.mode = dataset.mode  # file format
        self.patch_size = dataset.patch_size  # crop patch size
        self.scale = dataset.scale  # scale factor
        self.strides = dataset.strides  # crop strides when cropping in grid
        self.depth = dataset.depth  # the length of a video sequence
        self.modcrop = dataset.modcrop  # crop boarder pixels can't be divided by scale
        self.loop = loop  # infinite iterate
        self.random = dataset.random and not (method == 'test')  # random crop, or gridded crop
        self.max_patches = dataset.max_patches  # max random crop patches
        self.grid = []  # crop coordinates
        self.frames = []  # loaded HR/LR frames
        self.batch_iterator = None
        self.built = False

        dataset_file = dataset.__getattr__(method.lower())
        if self.mode.lower() == 'pil-image1':
            self.dataset = [ImageFile(fp, loop) for fp in dataset_file]
        elif self.mode.upper() in _ALLOWED_RAW_FORMAT:
            self.dataset = [RawFile(
                fp, dataset.mode, (dataset.width, dataset.height), loop) for fp in dataset_file]

    def __next__(self):
        if not self.built:
            raise RuntimeError(
                'This loader has not been built! Call **build_loader** first.')
        next(self.batch_iterator)

    def __iter__(self):
        return self.batch_iterator

    def __len__(self):
        if self.random:
            return self.max_patches
        else:
            n_patches = 0
            for vf in self.dataset:
                w, h = vf.shape
                sr = self.strides or [w, h]
                sz = self.patch_size or [w, h]
                n_patches += ((w - sz[0]) // sr[0] + 1) * ((h - sz[1]) // sr[1] + 1)
            return n_patches

    def _build_iter(self):
        while True:
            if self.method == 'train':
                np.random.shuffle(self.grid)
            for frames_hr, frames_lr, x, y, name in self.grid:
                assert x % self.scale[0] == 0 and y % self.scale[1] == 0
                patch_size = self.patch_size or frames_hr[0].size
                box = np.array([x, y, x + patch_size[0], y + patch_size[1]])
                crop_hr = [img.crop(box) for img in frames_hr]
                crop_lr = [img.crop(box // [*self.scale, *self.scale]) for img in frames_lr]
                yield crop_hr, crop_lr, name
            if not self.loop:
                break

    def reset(self, *args, **kwargs):
        if not self.built:
            self.build_loader(**kwargs)
        else:
            if self.random:
                self.grid = []
                rand_index = np.random.randint(len(self.frames), size=self.max_patches)
                for i in rand_index:
                    hr, lr, name = self.frames[i]
                    _w, _h = hr[0].width, hr[0].height
                    _pw, _ph = self.patch_size or [_w, _h]
                    x = np.random.randint(0, _w - _pw + 1)
                    x -= x % self.scale[0]
                    y = np.random.randint(0, _h - _ph + 1)
                    y -= y % self.scale[1]
                    self.grid.append((hr, lr, x, y, name))
            self.batch_iterator = self._build_iter()
        self.built = True

    def build_loader(self, crop=True, **kwargs):
        """Build image1(s) pair loader, make self iterable

         Args:
             crop: if True, crop the images into patches
             kwargs: you can override attribute in the dataset
        """
        _crop_args = [
            'scale',
            'patch_size',
            'strides',
            'depth'
        ]
        for _arg in _crop_args:
            if _arg in kwargs and kwargs[_arg]:
                self.__setattr__(_arg, kwargs[_arg])

        self.scale = Utility.to_list(self.scale, 2)
        self.patch_size = Utility.to_list(self.patch_size, 2)
        self.strides = Utility.to_list(self.strides, 2)
        self.patch_size = Utility.shrink_mod_scale(self.patch_size, self.scale) if crop else None
        self.strides = Utility.shrink_mod_scale(self.strides, self.scale) if crop else None

        for vf in self.dataset:
            tf.logging.debug('loading ' + vf.name)
            depth = self.depth
            # read all frames if depth is set to -1
            if depth == -1: depth = vf.frames
            for _ in range(vf.frames // depth):
                frames_hr = [ImageProcess.shrink_to_multiple_scale(img, self.scale) if self.modcrop else img for img in
                             vf.read_frame(depth)]
                frames_lr = [ImageProcess.imresize(img, np.ones(2) / self.scale) for img in frames_hr]
                self.frames.append((frames_hr, frames_lr, vf.name))
            vf.reopen()
        tf.logging.debug('all files load finished, generating cropping meshes...')
        self.random = self.random and crop
        if self.random:
            rand_index = np.random.randint(len(self.frames), size=self.max_patches)
            for i in rand_index:
                hr, lr, name = self.frames[i]
                _w, _h = hr[0].width, hr[0].height
                _pw, _ph = self.patch_size or [_w, _h]
                x = np.random.randint(0, _w - _pw + 1)
                x -= x % self.scale[0]
                y = np.random.randint(0, _h - _ph + 1)
                y -= y % self.scale[1]
                self.grid.append((hr, lr, x, y, name))
        else:
            for hr, lr, name in self.frames:
                _w, _h = hr[0].width, hr[0].height
                _sw, _sh = self.strides or [_w, _h]
                _pw, _ph = self.patch_size or [_w, _h]
                x, y = np.mgrid[0:_w - _pw - (_w - _pw) % _sw + _sw:_sw,
                       0:_h - _ph - (_h - _ph) % _sh + _sh:_sh]
                self.grid += [(hr, lr, _x, _y, name) for _x, _y in zip(x.flatten(), y.flatten())]
        tf.logging.info('data loader is ready!')
        self.batch_iterator = self._build_iter()
        self.built = True


class BatchLoader:
    """Build an iterator to load datasets in batch mode

      Args:
          batch_size: an integer, the size of a batch
          dataset: an instance of Dataset, see DataLoader.Dataset
          method: 'train', 'val', or 'test', each for different frames in datasets
          scale: scale factor
          loop: if True, iterates infinitely
          convert_to: can be either 'gray', 'rgb' or 'ycbcr', case insensitive
          augmentation: augment dataset by randomly rotate and flip
          kwargs: you can override attribute in the dataset
    """

    def __init__(self,
                 batch_size,
                 dataset,
                 method,
                 scale,
                 loop=False,
                 convert_to='gray',
                 augmentation=True,
                 **kwargs):
        tf.logging.debug(f'Loading {method} data files...')
        self.loader = Loader(dataset, method, loop)
        self.loader.build_loader(scale=scale, **kwargs)
        self.aug = augmentation
        self.batch = batch_size
        if convert_to.lower() in ('gray', 'l'):
            self.color_format = 'L'
        elif convert_to.lower() in ('yuv', 'ycbcr'):
            self.color_format = 'YCbCr'
        elif convert_to.lower() == 'rgb':
            self.color_format = 'RGB'
        else:
            tf.logging.warning(f'Unknown format {convert_to}, use grayscale by default')
            self.color_format = 'L'

    def __iter__(self):
        return self

    def __next__(self):
        hr, lr, name = self._load_batch()
        if isinstance(hr, np.ndarray) and isinstance(lr, np.ndarray):
            try:
                return np.squeeze(hr, 1), np.squeeze(lr, 1), np.squeeze(name)
            except ValueError:
                return hr, lr, np.squeeze(name)
        raise StopIteration('End BatchLoader!')

    def __len__(self):
        """Total iteration steps"""
        steps = np.ceil(len(self.loader) / self.batch)
        return int(steps)

    def _augment(self, image, op):
        if op[0]:
            image = np.rot90(image, 1)
        if op[1]:
            image = np.fliplr(image)
        if op[2]:
            image = np.flipud(image)
        return image

    def _load_batch(self):
        batch_hr, batch_lr = [], []
        batch_name = []
        for hr, lr, name in self.loader:
            hr = [img.convert(self.color_format) for img in hr]
            lr = [img.convert(self.color_format) for img in lr]
            clip_hr, clip_lr = [], []
            ops = np.random.randint(0, 2, [3]) if self.aug else [0, 0, 0]
            for _hr, _lr in zip(hr, lr):
                clip_hr.append(self._augment(ImageProcess.img_to_array(_hr), ops))
                clip_lr.append(self._augment(ImageProcess.img_to_array(_lr), ops))
            batch_hr.append(np.stack(clip_hr))
            batch_lr.append(np.stack(clip_lr))
            batch_name.append(name)
            if len(batch_hr) == self.batch:
                return np.stack(batch_hr), np.stack(batch_lr), np.stack(batch_name)
        if batch_hr and batch_lr:
            return np.stack(batch_hr), np.stack(batch_lr), np.stack(batch_name)
        return [], [], []

    def reset(self, *args, **kwargs):
        """reset the iterator"""
        self.loader.reset(*args, **kwargs)


class Select:
    # each file is selected equally
    EQUAL_FILE = 0
    # each pixel is selected equally, that is,
    # a larger image has a higher probability to
    # be selected, and vice versa
    EQUAL_PIXEL = 1


class QuickLoader:
    """Async data loader with high efficiency.

    QuickLoader uses asyncio to prefetch data into memory every n iterations,
    and provides several methods to select files. Unlike BatchLoader, QuickLoader
    never loads all files in the dataset into memory.

    """

    def __init__(self, batch_size, dataset, method, scale, loop=False, modcrop=True,
                 convert_to='gray', augmentation=False, **kwargs):
        self.file_names = dataset.__getattr__(method.lower())
        self.dataset = dataset
        self.scale = Utility.to_list(scale, 2)
        self.loop = loop
        self.modcrop = modcrop
        self.aug = augmentation
        self.batch = batch_size
        self.color_mode = convert_to.upper()
        self.prob = self._read_file()._calc_select_prob()
        self.frames = []  # pre-fetched frame buffer

    def _read_file(self):
        if self.dataset.mode.lower() == 'pil-image1':
            self.file_objects = [ImageFile(fp, self.loop) for fp in self.file_names]
        elif self.dataset.mode.upper() in _ALLOWED_RAW_FORMAT:
            self.file_objects = [
                RawFile(fp, self.dataset.mode, (self.dataset.width, self.dataset.height), self.loop) \
                for fp in self.file_names]
        return self

    def _calc_select_prob(self, method=Select.EQUAL_PIXEL):
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

    def _random_select(self, seed=None):
        if seed:
            np.random.seed(seed)
        x = np.random.rand()
        x *= np.ones_like(self.prob)
        diff = self.prob >= x
        index = diff.nonzero()[0].tolist()
        if index:
            return index[0]
        return 0

    def _process_at_file(self, vf):
        assert isinstance(vf, (RawFile, ImageFile))

        tf.logging.info('Prefetching ' + vf.name)
        depth = self.dataset.depth
        # read all frames if depth is set to -1
        if depth == -1: depth = vf.frames
        for i in range(vf.frames // depth):
            print(i, flush=True)
            frames_hr = [ImageProcess.shrink_to_multiple_scale(img, self.scale) \
                             if self.modcrop else img for img in vf.read_frame(depth)]
            frames_lr = [ImageProcess.imresize(img, np.ones(2) / self.scale) for img in frames_hr]
            self.frames.append({"hr": frames_hr, "lr": frames_lr, "name": f'{vf.name}_{i:04d}'})
        vf.reopen()  # necessary, rewind the file pointer

    def change_select_method(self, method):
        self.prob = self._calc_select_prob(method)

    def prefetch(self, size):
        """Prefetch `size` files and load into memory

        Args:
            size: if 0 < size < 1, size is the percentage of the fetched files.
                  if size >= 1, size is the absolute amount of the fetched files.
        """
        if size <= 0:
            raise ValueError('wrong prefetch size!')
        elif size < 1:
            size = len(self.file_objects) * size
        elif size > len(self.file_objects):
            size = len(self.file_objects)
        size = int(np.round(size))
        patch_size = Utility.to_list(self.dataset.patch_size, 2)
        patch_size = Utility.shrink_mod_scale(patch_size, self.scale)

        for _ in range(size):
            index = self._random_select()
            self._process_at_file(self.file_objects[index])
        return self

    def dump_info(self):
        pass
