"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 8th 2018

utility functions
"""
from typing import Generator
import tensorflow as tf
import numpy as np


def to_list(x, repeat=1):
    """convert x to list object

    Args:
         x: any object to convert
         repeat: if x is to make as [x], repeat `repeat` elements in the list
    """
    if isinstance(x, (Generator, tuple, set)):
        return list(x)
    elif isinstance(x, list):
        return x
    elif isinstance(x, dict):
        return list(x.values())
    elif x is not None:
        return [x] * repeat
    else:
        return []


def shrink_mod_scale(x, scale):
    """clip each dim of x to multiple of scale

    """
    scale = to_list(scale, 2)
    mod_x = []
    for _x, _s in zip(x, scale):
        mod_x.append(_x - _x % _s)
    return mod_x


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    """
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)


def pixel_shift(t, scale, c):
    with tf.name_scope('SubPixel'):
        r = to_list(scale, 2)
        shape = tf.shape(t)
        H, W = shape[1], shape[2]
        C = c
        t = tf.reshape(t, [-1, H, W, *r, C])
        # Here we are different from Equation 4 from the paper. That equation
        # is equivalent to switching 3 and 4 in `perm`. But I feel my
        # implementation is more natural.
        t = tf.transpose(t, perm=[0, 1, 3, 2, 4, 5])  # S, H, r, H, r, C
        t = tf.reshape(t, [-1, H * r[1], W * r[0], C])
        return t


def xavier_cnn_initializer(shape, uniform=True, **kwargs):
    fan_in = shape[0] * shape[1] * shape[2]
    fan_out = shape[0] * shape[1] * shape[3]
    n = fan_in + fan_out
    if uniform:
        init_range = np.sqrt(6.0 / n)
        return tf.random_uniform(shape, minval=-init_range, maxval=init_range)
    else:
        stddev = np.sqrt(3.0 / n)
        return tf.truncated_normal(shape=shape, stddev=stddev)


def he_initializer(shape, **kwargs):
    n = shape[0] * shape[1] * shape[2]
    stddev = np.sqrt(2.0 / n)
    return tf.truncated_normal(shape=shape, stddev=stddev)


def bias(shape, initial_value=0.0, name=None):
    initial = tf.constant(initial_value, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)
