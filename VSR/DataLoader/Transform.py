"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-7

Image data transform functions
"""
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from numpy.lib.arraysetops import isin


class Transformer(object):
    """Image transformer.

    Args:
        value: the parameter for each transform function.
        random: if specify 'uniform', generate value sampled from 0 to `+value`;
                if specify 'normal', generate value N~(mean=0, std=value)
    """

    def __init__(self, value=1, random=None):
        self._v = value
        self._r = random

    @property
    def value(self):
        """return random value"""

        if self._r == 'uniform':
            return np.random.uniform(0, self._v)
        elif self._r == 'normal':
            return np.random.normal(0, self._v)
        else:
            return self._v

    def __call__(self, img: Union[Image.Image, np.ndarray]):
        assert isinstance(img, (Image.Image, np.ndarray))
        return self.call(img)

    def call(self, img):
        raise NotImplementedError


class Tidy(Transformer):
    """Crop extra border pixels which not divisable by scaling factor
    """

    def call(self, img: Image.Image):
        scale = self.value
        shape = np.array((img.width, img.height))
        shape -= shape % scale
        return img.crop([0, 0, *shape.tolist()])


class Bicubic(Transformer):
    """Resize image by scaling factor via bicubic method (`PIL.Image.resize`)
    """

    def call(self, img: Image.Image):
        scale = self.value
        shape = np.array((img.width, img.height))
        if scale < 1:
            rscale = int(1 / scale)
            if np.any(shape % rscale):
                raise ValueError(f"Image size is not divisible by {rscale}.")
            return img.resize(shape // rscale, resample=Image.BICUBIC)
        else:
            return img.resize((shape * scale).astype('int32'), resample=Image.BICUBIC)


class Brightness(Transformer):
    """Adjust the brightness of the image
    """

    def call(self, img: Image.Image):
        brightness = max(0, self.value)
        return ImageEnhance.Brightness(img).enhance(brightness)


class Contrast(Transformer):
    """Adjust the contrast of the image
    """

    def call(self, img: Image.Image):
        contrast = self.value
        return ImageEnhance.Contrast(img).enhance(contrast)


class Sharpness(Transformer):
    """Enhance the high frequenct part of the image
    """

    def call(self, img):
        sharp = min(max(0, self.value), 2)
        return ImageEnhance.Sharpness(img).enhance(sharp)


class GaussianBlur(Transformer):
    """Blur image through gaussian kernel
    """

    def call(self, img):
        radius = self.value
        return ImageFilter.GaussianBlur(radius).filter(img)


class GaussianWhiteNoise(Transformer):
    """Add additive gaussian white noise (AWGN)
    """

    def call(self, img):
        shape = img.shape
        noise = np.random.normal(0, self.value, shape)
        noise += img.astype('float32')
        return np.clip(np.round(noise), 0, 255).astype('uint8')


class FixedVideoLengthBatch(Transformer):
    def call(self, img):
        assert img.ndim == 5, f"img is not 5D, which is {img.ndim}"
        depth = int(self.value)
        shape = img.shape
        if shape[1] <= depth:
            return img
        ret = []
        for i in range(shape[1] - depth + 1):
            ret.append(img[:, i * depth: (i + 1) * depth])
        return np.stack(ret, 1).reshape([-1, depth, *shape[-3:]])
