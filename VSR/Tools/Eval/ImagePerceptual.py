"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Nov 26th 2018

Calculate perceptual metrics for images:
- FID
- Inception Score
TODO:
- KID
- Perceptual Index
"""

import tensorflow as tf
import numpy as np

from .Task import Task
from ...Framework.GAN import inception_score, fid_score

tf.flags.DEFINE_bool("enable_fid", False, help="evaluate fid.")
tf.flags.DEFINE_bool("enable_inception_score", False, help="evaluate inception score.")
FLAGS = tf.flags.FLAGS


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
    return np.concatenate(img)


class InceptionTask(Task):
    def __call__(self, label_images, fake_images):
        del label_images
        results = []
        for x1 in fake_images:
            x1 = tf.constant(x1.astype('float32'))
            score = inception_score(x1)
            results.append(score.eval())
        return results


class FidTask(Task):
    def __call__(self, label_images, fake_images):
        results = []
        for x0, x1 in zip(label_images, fake_images):
            fid = fid_score(x0, x1)
            results.append(fid.eval())
        return results
