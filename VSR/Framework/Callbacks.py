"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 17th 2018
Updated Date: May 17th 2018

Training environment callbacks preset
"""

from pathlib import Path
from functools import partial
import numpy as np

from ..Util.ImageProcess import array_to_img, img_to_array, bicubic_rescale


def _sub_residual(**kwargs):
    img = kwargs['input'] if 'input' in kwargs else None
    res = kwargs['output'] if 'output' in kwargs else np.zeros_like(img)
    return img - res


def _save_model_predicted_images(**kwargs):
    img = kwargs['output'] if 'output' in kwargs else None
    save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '.'
    step = kwargs['step'] if 'step' in kwargs else 0
    if img is not None:
        img = img[0] if isinstance(img, list) else img
        img = _to_normalized_image(img)
        path = Path(f'{save_dir}/{step:03d}-predict.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        img.convert('RGB').save(str(path))


def _colored_grayscale_image(output, input, **kwargs):
    img = output[0][0] if isinstance(output, list) else output[0]
    assert img.shape[-1] == 1
    uv = array_to_img(input[0], 'YCbCr')
    scale = img.shape[1:3] // uv.shape[1:3]
    uv = bicubic_rescale(uv, scale)
    uv = img_to_array(uv)
    img = np.concatenate([img.uv[..., 1:]], axis=-1)
    img = array_to_img(img, 'YCbCr').convert('RGB')
    return img


def _to_normalized_image(img):
    img = np.asarray(img)
    # squeeze to [H, W, C]
    for i in range(np.ndim(img)):
        try:
            img = np.squeeze(img, i)
        except ValueError:
            pass
    if img.dtype == np.float32 and img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255)
    if img.ndim == 2:
        return array_to_img(img, 'L')
    elif img.ndim == 3:
        return array_to_img(img, 'YCbCr')
    else:
        raise ValueError('Invalid img data, must be an array of 2D image with channel less than 3')


def save_image(save_dir='.'):
    return partial(_save_model_predicted_images, save_dir=save_dir)


def reduce_residual(**kwargs):
    return partial(_sub_residual, **kwargs)


def to_rgb(**kwargs):
    return partial(_colored_grayscale_image, **kwargs)
