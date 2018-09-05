"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: Aug 29th 2018

Load frames with specified filter in given directories,
and provide inheritable API for specific loaders.
- Added QuickLoader and MpLoader (multiprocessor loader)
- Will BatchLoader (and Loader)
"""

import numpy as np
import tensorflow as tf
import multiprocessing as mp
from psutil import virtual_memory

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
        loader: A `QuickLoader` or `BatchLoader` to provide properties.
        grids: A list of tuple, commonly returned from `QuickLoader._generate_crop_grid`.
    """

    def __init__(self, loader, grids):
        self.batch = loader.batch
        self.scale = loader.scale
        self.aug = loader.aug
        self.loader = loader
        self.grids = grids
        self.synced = True
        if hasattr(grids, 'get'):
            # this is a pool.ApplyResult, need to get its values
            self.synced = False

    def __len__(self):
        if not self.synced:
            # get results from process pool
            grids = []
            for p in self.grids.get():
                grids += p
            self.grids = grids
            self.synced = True
        return len(self.grids) // self.batch

    def __iter__(self):
        return self

    def __next__(self):
        batch_hr, batch_lr, batch_name = [], [], []
        if not self.synced:
            # get results from process pool
            grids = []
            for p in self.grids.get():
                grids += p
            self.grids = grids
            self.synced = True
        if not self.grids:
            raise StopIteration

        while self.grids and len(batch_hr) < self.batch:
            hr, lr, box, name = self.grids.pop(0)
            box = np.array(box, 'int32')
            try:
                assert (np.mod(box, [*self.scale, *self.scale]) == 0).all()
            except AssertionError:
                tf.logging.error(f'Scale x{self.scale[0]}, but crop box is ' + str(box))
            crop_hr = [ImageProcess.crop(img, box) for img in hr]
            crop_lr = [ImageProcess.crop(img, box // [*self.scale, *self.scale]) for img in lr]
            ops = np.random.randint(0, 2, [3]) if self.aug else [0, 0, 0]
            clip_hr = [_augment(ImageProcess.img_to_array(img), ops) for img in crop_hr]
            clip_lr = [_augment(ImageProcess.img_to_array(img), ops) for img in crop_lr]
            batch_hr.append(np.stack(clip_hr))
            batch_lr.append(np.stack(clip_lr))
            batch_name.append(name)

        if batch_hr and batch_lr and batch_name:
            batch_hr = np.stack(batch_hr)
            batch_lr = np.stack(batch_lr)
            batch_name = np.stack(batch_name)
        if isinstance(batch_hr, np.ndarray) and isinstance(batch_lr, np.ndarray):
            try:
                return np.squeeze(batch_hr, 1), np.squeeze(batch_lr, 1), np.squeeze(batch_name)
            except ValueError:
                return batch_hr, batch_lr, np.squeeze(batch_name)

        return batch_hr, batch_lr, batch_name


class QuickLoader:
    """Async data loader with high efficiency.

    QuickLoader concurrently pre-fetches clips into memory every n iterations,
    and provides several methods to select clips. Unlike BatchLoader, QuickLoader
    never loads all files in the dataset into memory.

    NOTE: A clip is a bunch of consecutive frames, which can represent either a dynamic
    video or single image.

    Args:
        batch_size: An int scalar, representing the number of consecutive clips
          to combine in a single batch.
        dataset: A `Dataset` to load by this loader.
        method: A string in ('train', 'val', 'test') specifies which subset to use
          in the dataset. Also 'train' set will shuffle buffers each epoch.
        scale: A list of int to specify the scale factor of each dimension of images
          to generate corresponded LR patches. If `scale` is a scalar, the factor is
          equal for each dimension.
        batches_per_epoch: An int scalar, representing the number of batches generated
          for each epoch, note that `total patches := batches_per_epoch * batch_size`.
        no_patch: A boolean, set to True disable cropping patches, used for test.
        convert_to: A string in ('gray', 'yuv', 'rgb'), see codes below for details. This
          will convert the color format of the image to desired one. *Case insensitive*.
        augmentation: A boolean to specify whether call `_augment` to batches. `_augment`
          will randomly flip or rotate images.
    """

    def __init__(self, batch_size, dataset, method, scale, batches_per_epoch=-1,
                 no_patch=False, convert_to='gray', augmentation=False,
                 **kwargs):
        self.file_names = dataset.__getattr__(method.lower())
        self.depth = dataset.depth
        self.patch_size = dataset.patch_size
        self.flow = dataset.flow
        self.scale = Utility.to_list(scale, 2)
        self.patches_per_epoch = batches_per_epoch * batch_size
        self.no_patch = no_patch
        self.modcrop = dataset.modcrop
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
        self.prob = self._read_file(dataset)._calc_select_prob()
        self.all_loaded = False
        self.frames = []  # a list of tuple represents (HR, LR, name) of a clip

    def _read_file(self, dataset):
        """Initialize all `File` objects"""
        if dataset.mode.lower() == 'pil-image1':
            if self.flow:
                # map flow
                flow = {f.stem: f for f in self.flow}
                self.file_objects = [ImageFile(fp).attach_flow(flow[fp.stem]) for fp in self.file_names]
            else:
                self.file_objects = [ImageFile(fp) for fp in self.file_names]
        elif dataset.mode.upper() in _ALLOWED_RAW_FORMAT:
            self.file_objects = [
                RawFile(fp, dataset.mode, (dataset.width, dataset.height))
                for fp in self.file_names]
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
        # TODO is `s` relevant to poisson dist.?
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
        frames_hr = [ImageProcess.shrink_to_multiple_scale(img, self.scale)
                     if self.modcrop else img for img in vf.read_frame(depth)]
        frames_lr = [ImageProcess.imresize(img, np.ones(2) / self.scale) for img in frames_hr]
        frames_hr = [img.convert(self.color_format) for img in frames_hr]
        frames_lr = [img.convert(self.color_format) for img in frames_lr]
        return frames_hr, frames_lr, f'{vf.name}_{index:04d}'

    def _vf_gen_flow_img_pair(self, vf, depth, index):
        assert depth == 2 and index == 0
        img = [img for img in vf.read_frame(depth)]
        img = [i.convert(self.color_format) for i in img]
        return img, [vf.flow], f'{vf.name}_{index:04d}'

    def _process_at_file(self, vf, clips=1):
        """load frames of `File` into memory, crop and generate corresponded LR frames.

        Args:
            vf: A `File` object.
            clips: an int scalar to specify how many clips to generate from `vf`.

        Return:
            List of Tuple: containing (HR, LR, name) respectively
        """
        assert isinstance(vf, (RawFile, ImageFile))

        tf.logging.debug('Prefetching ' + vf.name)
        depth = self.depth
        # read all frames if depth is set to -1
        if depth == -1: depth = vf.frames
        index = np.arange(0, vf.frames - depth + 1)
        np.random.shuffle(index)
        if clips > len(index):
            tf.logging.log_every_n(
                tf.logging.WARN, 'clips are greater than actual frames in the file', 100)
        frames = []
        for i in index[:clips]:
            if self.flow:
                frames.append(self._vf_gen_flow_img_pair(vf, depth, i))
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
            list of tuple: containing (HR, LR, box, name) respectively, where HR and LR
              are reference frames, box is a list of 4 int of crop coordinates.
        """
        patch_size = Utility.to_list(self.patch_size, 2)
        patch_size = Utility.shrink_mod_scale(patch_size, self.scale)
        if size < 0:
            index = np.arange(len(frames)).tolist()
        else:
            index = np.random.randint(len(frames), size=size).tolist()
        grids = []
        for i in range(len(frames)):
            hr, lr, name = frames[i]
            _w, _h = hr[0].width, hr[0].height
            if not self.no_patch:
                _pw, _ph = patch_size
            else:
                _pw, _ph = _w, _h
            amount = index.count(i)
            x = np.random.randint(0, _w - _pw + 1, size=amount)
            x -= x % self.scale[0]
            y = np.random.randint(0, _h - _ph + 1, size=amount)
            y -= y % self.scale[1]
            grids += [(ImageProcess.img_to_array(hr),
                       ImageProcess.img_to_array(lr),
                       [_x, _y, _x + _pw, _y + _ph], name) for _x, _y in zip(x, y)]
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
        return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64) for vf in self.file_objects])

    def __len__(self):
        """length of a QuickLoader is defined as the total frames in Dataset"""
        return np.sum([vf.frames for vf in self.file_objects])

    def change_select_method(self, method):
        """change to different select method, see `Select`"""
        self.prob = self._calc_select_prob(method)
        return self

    def _prefetch(self, memory_usage=None, shard=1, index=0):
        """Prefetch `size` files and load into memory. Specify `shard` will divide
        loading files into `shard` shards in order to execute in parallel.

        NOTE: parallelism is implemented via `MpLoader`

        Args:
            memory_usage: desired virtual memory to use, could be int (in bytes) or
              a readable string (i.e. '3GB', '1TB'). Default to use all available
              memories.
            shard: an int scalar to specify the number of shards operating in parallel.
            index: an int scalar, representing shard index
        """

        if self.all_loaded:
            interval = len(self.frames) // shard
            return self.frames[index * interval:(index + 1) * interval], True
        # check memory usage
        if isinstance(memory_usage, str):
            memory_usage = Utility.str_to_bytes(memory_usage)
        if not memory_usage:
            memory_usage = virtual_memory().total
        memory_usage = np.min([np.uint64(memory_usage), virtual_memory().free])
        capacity = self.size
        frames = []
        if capacity <= memory_usage:
            # load all clips
            self.all_loaded = True
            interval = len(self.file_objects) // shard
            for file in self.file_objects[index * interval:(index + 1) * interval]:
                frames += self._process_at_file(file, file.frames)
        else:
            prop = memory_usage / capacity / shard
            size = int(np.round(len(self) * prop))
            for file, amount in self._random_select(size).items():
                frames += self._process_at_file(file, amount)
        self.frames = frames
        return frames, self.all_loaded

    def _prefetch_handler(self, memory_usage=None, shard=1, index=0, error=None):
        try:
            return self._prefetch(memory_usage, shard, index)
        except Exception as e:
            # TODO have no idea what to do with this stuff
            return [], False

    def make_one_shot_iterator(self, memory_usage=None, shard=1, shuffle=False):
        """make an `EpochIterator` to enumerate batches of the dataset

        Args:
            memory_usage: desired virtual memory to use, could be int (in bytes) or
              a readable string (i.e. '3GB', '1TB'). Default to use all available
              memories.
            shard: no use
            shuffle: A boolean whether to shuffle the patch grids.

        Return:
            An EpochIterator
        """
        fr, _ = self._prefetch(memory_usage, 1)
        grids = self._generate_crop_grid(fr, self.patches_per_epoch, shuffle=shuffle)
        return EpochIterator(self, grids)


class MpLoader:
    """Multiprocessor loader"""

    def __init__(self, *args, **kwargs):
        self.pool = mp.Pool(mp.cpu_count() // 2)
        self.loader = QuickLoader(*args, **kwargs)

    def __del__(self):
        self.pool.close()
        self.pool.terminate()

    def make_one_shot_iterator(self, memory_usage=None, shard=1, shuffle=False):
        """make an `EpochIterator` to enumerate batches of the dataset. Specify
        `shard` will divide loading files into `shard` shards in order to execute
        in parallel.

        Args:
            memory_usage: desired virtual memory to use, could be int (in bytes) or
              a readable string (i.e. '3GB', '1TB'). Default to use all available
              memories.
            shard: an int scalar to specify the number of shards operating in parallel.
            shuffle: A boolean whether to shuffle the patch grids.

        Return:
            An EpochIterator

        Known issues:
            If data of either shard is too large (i.e. use 1 shard and total frames
            is around 6GB in my machine), windows Pipe may broke and `get()` never
            returns.
        """

        def map_cb(values):
            # update loader object in current process
            self.loader.all_loaded = values[0][1]
            self.loader.frames = []
            # reduce
            for i in range(len(values)):
                self.loader.frames += values[i][0]
                values[i] = self.loader._generate_crop_grid(
                    values[i][0], self.loader.patches_per_epoch // shard, shuffle=shuffle)

        # map
        if self.loader.all_loaded:
            # TODO a work around 'cause AssertionError when map functions again
            grids = self.loader._generate_crop_grid(
                self.loader.frames, self.loader.patches_per_epoch, shuffle=shuffle)
        else:
            grids = self.pool.starmap_async(self.loader._prefetch_handler,
                                            [(memory_usage, shard, i) for i in range(shard)],
                                            callback=map_cb)
        return EpochIterator(self.loader, grids)
