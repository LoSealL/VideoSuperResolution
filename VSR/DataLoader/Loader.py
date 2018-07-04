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

from .Dataset import Dataset
from .VirtualFile import RawFile, ImageFile, _ALLOWED_RAW_FORMAT
from ..Util import ImageProcess, Utility


class Loader(object):

    def __init__(self, dataset, method, loop=False):
        """Initiate loader for given path `path`

        Args:
            dataset: dataset object, see Dataset.py
            method: 'train', 'val', or 'test'
            loop: if True, read data infinitely
        """
        if not isinstance(dataset, Dataset):
            raise TypeError('dataset must be Dataset object')

        self.method = method  # train/val/test
        self.mode = dataset.mode  # file format
        self.patch_size = dataset.patch_size  # crop patch size
        self.scale = dataset.scale  # scale factor
        self.strides = dataset.strides  # crop strides when cropping in grid
        self.depth = dataset.depth  # the length of a video sequence
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
            np.random.shuffle(self.grid)
            for frames_hr, frames_lr, x, y, name in self.grid:
                patch_size = self.patch_size or frames_hr[0].size
                box = np.array([x, y, x + patch_size[0], y + patch_size[1]])
                crop_hr = [img.crop(box) for img in frames_hr]
                crop_lr = [img.crop(box // [*self.scale, *self.scale]) for img in frames_lr]
                yield crop_hr, crop_lr, name
            if not self.loop:
                break

    def reset(self, **kwargs):
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
                    y = np.random.randint(0, _h - _ph + 1)
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
            for _ in range(vf.frames // self.depth):
                frames_hr = [ImageProcess.shrink_to_multiple_scale(img, self.scale) for img in
                             vf.read_frame(self.depth)]
                frames_lr = [ImageProcess.bicubic_rescale(img, np.ones(2) / self.scale) for img in frames_hr]
                self.frames.append((frames_hr, frames_lr, vf.name))
            vf.reopen()
        print('files load finished, generating cropping meshes...')
        self.random = self.random and crop
        if self.random:
            rand_index = np.random.randint(len(self.frames), size=self.max_patches)
            for i in rand_index:
                hr, lr, name = self.frames[i]
                _w, _h = hr[0].width, hr[0].height
                _pw, _ph = self.patch_size or [_w, _h]
                x = np.random.randint(0, _w - _pw + 1)
                y = np.random.randint(0, _h - _ph + 1)
                self.grid.append((hr, lr, x, y, name))
        else:
            for hr, lr, name in self.frames:
                _w, _h = hr[0].width, hr[0].height
                _sw, _sh = self.strides or [_w, _h]
                _pw, _ph = self.patch_size or [_w, _h]
                x, y = np.mgrid[0:_w - _pw - (_w - _pw) % _sw + _sw:_sw,
                       0:_h - _ph - (_h - _ph) % _sh + _sh:_sh]
                self.grid += [(hr, lr, _x, _y, name) for _x, _y in zip(x.flatten(), y.flatten())]
        print('data loader ready!')
        self.batch_iterator = self._build_iter()
        self.built = True


class BatchLoader:

    def __init__(self,
                 batch_size,
                 dataset,
                 method,
                 scale,
                 loop=False,
                 convert_to='gray',
                 **kwargs):
        """Build an iterable to load datasets in batch size

        Args:
            batch_size: an integer, the size of a batch
            dataset: an instance of Dataset, see DataLoader.Dataset
            method: 'train', 'val', or 'test', each for different frames in datasets
            scale: scale factor
            loop: if True, iterates infinitely
            convert_to: can be either 'gray', 'rgb' or 'yuv', case insensitive
            kwargs: you can override attribute in the dataset
        """

        print(f'Loading {method} data files...')
        self.loader = Loader(dataset, method, loop)
        self.loader.build_loader(scale=scale, **kwargs)
        self.batch = batch_size
        if convert_to.lower() in ('gray', 'l'):
            self.color_format = 'L'
        elif convert_to.lower() in ('yuv', 'ycbcr'):
            self.color_format = 'YCbCr'
        elif convert_to.lower() == 'rgb':
            self.color_format = 'RGB'
        else:
            print(f'Unknown format {convert_to}, use grayscale by default')
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

    def _load_batch(self):
        batch_hr, batch_lr = [], []
        batch_name = []
        for hr, lr, name in self.loader:
            hr = [img.convert(self.color_format) for img in hr]
            lr = [img.convert(self.color_format) for img in lr]
            batch_hr.append(np.stack([ImageProcess.img_to_array(img) for img in hr]))
            batch_lr.append(np.stack([ImageProcess.img_to_array(img) for img in lr]))
            batch_name.append(name)
            if len(batch_hr) == self.batch:
                return np.stack(batch_hr), np.stack(batch_lr), np.stack(batch_name)
        if batch_hr and batch_lr:
            return np.stack(batch_hr), np.stack(batch_lr), np.stack(batch_name)
        return [], [], []

    def reset(self, *args, **kwargs):
        """reset the iterator"""
        self.loader.reset(*args, **kwargs)
