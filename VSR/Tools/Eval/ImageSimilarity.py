"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Nov 26th 2018

Calculate similarity metrics for images:
- PSNR
- SSIM
"""

import tensorflow as tf
import numpy as np

from .Task import Task
from ...Util.ImageProcess import rgb_to_yuv

tf.flags.DEFINE_bool("enable_psnr", False, "evaluate psnr.")
tf.flags.DEFINE_bool("enable_ssim", False, "evaluate ssim.")
tf.flags.DEFINE_bool("l_only", False, "compute luminance only")
tf.flags.DEFINE_string("l_standard", "matlab", "yuv convertion standard, either 'bt601', 'bt709' or 'matlab'")
tf.flags.DEFINE_integer("shave", 0, "shave border pixels")
FLAGS = tf.flags.FLAGS


def shave(img, border):
    """shave away border pixels"""
    return img[..., border:-border, border:-border, :]


def normalize(img):
    assert isinstance(img, np.ndarray)
    if img.dtype in ('uint8', 'int32'):
        img = img.astype('float32')
    if img.dtype not in ('float32', 'float64'):
        raise TypeError(f'img with type {img.dtype} is not allowed.')
    if img.ndim == 3:
        img = [np.expand_dims(img, 0)]
    elif img.ndim == 4:
        img = np.split(img, img.shape[0])
    elif img.ndim == 5:
        img = img.reshape([-1, *img.shape[-3:]])
        img = np.split(img, img.shape[0])
    else:
        raise ValueError('ndim of img is not supported.')
    if FLAGS.l_only:
        img = [rgb_to_yuv(i, 255, FLAGS.l_standard)[..., 0:1] for i in img]
    if FLAGS.shave:
        img = [shave(i, FLAGS.shave) for i in img]
    return np.concatenate(img)


class PsnrTask(Task):
    def __call__(self, label_images, fake_images):
        assert isinstance(label_images, list)
        assert isinstance(fake_images, list)
        results = []
        # Placeholder avoids TF copy images' data into graph
        label_ph = tf.placeholder('float32', [None, None, None, None])
        fake_ph = tf.placeholder('float32', [None, None, None, None])
        psnr_tensor = tf.image.psnr(label_ph, fake_ph, 255)
        for x0, x1 in zip(label_images, fake_images):
            x0 = normalize(x0)
            x1 = normalize(x1)
            results.append(psnr_tensor.eval({label_ph: x0, fake_ph: x1}))
        return np.mean(results)


class SsimTask(Task):
    def __call__(self, label_images, fake_images):
        assert isinstance(label_images, list)
        assert isinstance(fake_images, list)
        results = []
        label_ph = tf.placeholder('float32', [None, None, None, None])
        fake_ph = tf.placeholder('float32', [None, None, None, None])
        ssim_tensor = tf.image.ssim(label_ph, fake_ph, 255)
        for x0, x1 in zip(label_images, fake_images):
            x0 = normalize(x0)
            x1 = normalize(x1)
            results.append(ssim_tensor.eval({label_ph: x0, fake_ph: x1}))
        return np.mean(results)
