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
from PIL.Image import Image

from ..Util.ImageProcess import array_to_img, img_to_array, bicubic_rescale


def _sub_residual(**kwargs):
    img = kwargs['input'] if 'input' in kwargs else None
    res = kwargs['output'] if 'output' in kwargs else np.zeros_like(img)
    res = res[0] if isinstance(res, list) else res
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


def _colored_grayscale_image(output, input, label, **kwargs):
    img = output[0] if isinstance(output, list) else output
    assert img.shape[-1] == 1
    scale = np.array(img.shape[1:3]) // np.array(input.shape[1:3])
    uv = array_to_img(input[0], 'YCbCr')
    uv = bicubic_rescale(uv, scale)
    uv = img_to_array(uv)[..., 1:]
    img = np.concatenate([img[0], uv], axis=-1)
    img = np.clip(img, 0, 255)
    img = array_to_img(img, 'YCbCr')
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
        raise ValueError('Invalid img data, must be an array of 2D image1 with channel less than 3')


def _add_noise(feature, stddev, mean):
    x = feature.astype('int32') + np.random.normal(mean, stddev, feature.shape)
    return np.clip(x, 0, 255).astype('uint8')


def _add_random_noise(feature, low, high, step, mean):
    n = list(range(low, high, step))
    i = np.random.randint(len(n))
    stddev = n[i]
    x = feature.astype('int32') + np.random.normal(mean, stddev, feature.shape)
    return np.clip(x, 0, 255).astype('uint8')


def _exponential_decay(lr, start_lr, epochs, steps, decay_step, decay_rate):
    return start_lr * decay_rate ** (steps / decay_step)


def _poly_decay(lr, start_lr, end_lr, epochs, steps, decay_step, power):
    return (start_lr - end_lr) * (1 - steps / decay_step) ** power + end_lr


def _stair_decay(lr, start_lr, epochs, steps, decay_step, decay_rate):
    return start_lr * decay_rate ** (steps // decay_step)


def _eval_psnr(output, label, **kwargs):
    if isinstance(output, Image):
        output = img_to_array(output.convert('RGB'))
    if isinstance(label, Image):
        label = img_to_array(label.convert('RGB'))
    if label.ndim == 4:
        label = label[0]
    assert output.shape == label.shape

    mse = np.mean(np.square(output - label))
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    print(f'PSNR = {psnr:.2f}dB')


def save_image(save_dir='.'):
    return partial(_save_model_predicted_images, save_dir=save_dir)


def print_psnr():
    return _eval_psnr


def reduce_residual(**kwargs):
    return partial(_sub_residual, **kwargs)


def to_rgb(**kwargs):
    return partial(_colored_grayscale_image, **kwargs)


def to_gray():
    def _gray_colored_image(input):
        return input[..., 0:1]

    return _gray_colored_image


def to_uv():
    def _uv_colored_image(input):
        return input[..., 1:]

    return _uv_colored_image


def add_noise(sigma, mean=0):
    return partial(_add_noise, stddev=sigma, mean=mean)


def add_random_noise(low, high, step=1, mean=0):
    return partial(_add_random_noise, low=low, high=high, step=step, mean=mean)


def lr_decay(method, lr, **kwargs):
    if method == 'exp':
        return partial(_exponential_decay, start_lr=lr, **kwargs)
    elif method == 'poly':
        return partial(_poly_decay, start_lr=lr, **kwargs)
    elif method == 'stair':
        return partial(_stair_decay, start_lr=lr, **kwargs)
    else:
        raise ValueError('invalid decay method!')
