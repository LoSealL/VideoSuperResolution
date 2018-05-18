"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 10th 2018

Load files with specified filter in given directories,
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
        dataset_file = dataset.__getattr__(method.lower())
        self.mode = dataset.mode
        if self.mode.lower() == 'pil-image':
            self.dataset = (ImageFile(fp, loop) for fp in dataset_file)
        elif self.mode.upper() in _ALLOWED_RAW_FORMAT:
            self.dataset = (RawFile(
                fp, dataset.mode, (dataset.width, dataset.height), loop) for fp in dataset_file)
        self.patch_size = dataset.patch_size
        self.scale = dataset.scale
        self.strides = dataset.strides
        self.depth = dataset.depth
        self.batch_iterator = None
        self.loop = loop
        self.built = False

    def __next__(self):
        if not self.built:
            raise RuntimeError(
                'This loader has not been built! Call **build_loader** first.')
        next(self.batch_iterator)

    def __iter__(self):
        return self.batch_iterator

    def _build_iter(self):
        while True:
            for vf in self.dataset:
                for _ in range(vf.frames // self.depth):
                    frames_hr = [ImageProcess.shrink_to_multiple_scale(img, self.scale) for img in
                                 vf.read_frame(self.depth)]
                    frames_lr = [ImageProcess.bicubic_rescale(
                        img, np.ones(2) / self.scale) for img in frames_hr]
                    width, height = frames_hr[0].size
                    strides = self.strides
                    patch_size = self.patch_size
                    if not strides:
                        strides = [width, height]
                    if not patch_size:
                        patch_size = [width, height]
                    for w in range(0, width, strides[0]):
                        for h in range(0, height, strides[1]):
                            if w + patch_size[0] > width or h + patch_size[1] > height:
                                continue
                            box = np.array([w, h, w + patch_size[0], h + patch_size[1]])
                            crop_hr = [img.crop(box) for img in frames_hr]
                            crop_lr = [img.crop(box // [*self.scale, *self.scale]) for img in frames_lr]
                            yield crop_hr, crop_lr
                vf.read_frame(vf.frames)
            if not self.loop:
                # StopIterator
                raise StopIteration('Dataset iterates over')

    def build_loader(self, shuffle=False, crop=True, **kwargs):
        """Build image(s) pair loader, make self iterable

         Args:
             shuffle: if True, shuffle the dataset. Note this will take long time if dataset contains many files
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
        if shuffle:
            self.dataset = list(self.dataset)
            np.random.shuffle(self.dataset)
        self.batch_iterator = self._build_iter()
        self.built = True


class BatchLoader:

    def __init__(self,
                 batch_size,
                 dataset,
                 method,
                 loop=False,
                 convert_to_gray=True,
                 **kwargs):
        """Build an iterable to load datasets in batch size

        Args:
            batch_size: an integer, the size of a batch
            dataset: an instance of Dataset, see DataLoader.Dataset
            method: 'train', 'val', or 'test', each for different files in datasets
            loop: if True, iterates infinitely
            kwargs: you can override attribute in the dataset
        """
        self.loader = Loader(dataset, method, loop)
        self.loader.build_loader(**kwargs)
        self.batch = batch_size
        self.to_gray = convert_to_gray

    def __iter__(self):
        return self

    def __next__(self):
        hr, lr = self._load_batch()
        if isinstance(hr, np.ndarray) and isinstance(lr, np.ndarray):
            try:
                return np.squeeze(hr, 1), np.squeeze(lr, 1)
            except ValueError:
                return hr, lr
        raise StopIteration('End BatchLoader!')

    def _load_batch(self):
        batch_hr, batch_lr = [], []
        for hr, lr in self.loader:
            if self.to_gray:
                hr = [img.convert('L') for img in hr]
                lr = [img.convert('L') for img in lr]
            else:
                hr = [img.convert('YCbCr') for img in hr]
                lr = [img.convert('YCbCr') for img in lr]
            batch_hr.append(np.stack([ImageProcess.img_to_array(img) for img in hr]))
            batch_lr.append(np.stack([ImageProcess.img_to_array(img) for img in lr]))
            if len(batch_hr) == self.batch:
                return np.stack(batch_hr), np.stack(batch_lr)
        return [], []
