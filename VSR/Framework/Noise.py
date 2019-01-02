"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 25th 2018

Utility for complicated noise model
Ref:
[1] https://arxiv.org/abs/1807.04686
"""

import tensorflow as tf
import numpy as np


def camera_response_function(inputs, crf_table, max_val=1):
    """Estimated CRF, transform irradiance L to RGB image. If `crf_table` is
      inverted, transform RGB image to irradiance L.

    Args:
        inputs: A 3-D or 4-D tensor, representing irradiance.
        crf_table: CRF lookup table shape (1024,).
        max_val: specify the range of inputs: \in (0, max_val)
    Return:
        RGB images (or L) with the same shape as inputs, in range [0, max_val].
    """

    inputs_norm = np.clip(inputs / max_val, 0, 1)
    quant = crf_table.shape[0] - 1
    inputs_index = (inputs_norm * quant).astype('int32')
    ret = []
    for i in inputs_index.flatten():
        ret.append(crf_table[i])
    return np.reshape(ret, inputs.shape) * max_val


def tf_camera_response_function(inputs, crf_table, max_val=1):
    """TF implementation of CRF."""

    with tf.name_scope('CRF'):
        inputs_norm = tf.clip_by_value(inputs / max_val, 0, 1)
        quant = int(crf_table.shape[0] - 1)
        inputs_index = tf.to_int32(inputs_norm * quant)
        return tf.nn.embedding_lookup(crf_table, inputs_index) * max_val


def poisson_noise(inputs, stddev=None, sigma_max=0.16):
    """Add poisson noise to inputs."""

    if stddev is None:
        stddev = np.random.rand(inputs.shape[-1]) * sigma_max
    stddev = np.reshape(stddev, [1] * (inputs.ndim - 1) + [-1])
    sigma_map = inputs * stddev
    return np.random.randn(*inputs.shape) * sigma_map


def tf_poisson_noise(inputs, stddev=None, sigma_max=0.16):
    with tf.name_scope('PoissonNoise'):
        if stddev is None:
            stddev = tf.random_uniform(inputs.shape[-1:], maxval=sigma_max)
        stddev = tf.reshape(stddev, [1, 1, 1, -1])
        sigma_map = inputs * stddev
        return tf.random_normal(tf.shape(inputs)) * sigma_map


def gaussian_noise(inputs, stddev=None, sigma_max=0.06, channel_wise=True):
    """Add channel wise gaussian noise."""

    channel = inputs.shape[-1] if channel_wise else 1
    if stddev is None:
        stddev = np.random.rand(channel) * sigma_max
    stddev = np.reshape(stddev, [1] * (inputs.ndim - 1) + [-1])
    noise_map = np.random.randn(*inputs.shape) * stddev
    return noise_map


def tf_gaussian_noise(inputs, stddev=None, sigma_max=0.06, channel_wise=True):
    with tf.name_scope('GaussianNoise'):
        channel = inputs.shape[-1:] if channel_wise else [1]
        if stddev is None:
            stddev = tf.random_uniform(channel, maxval=sigma_max)
        stddev = tf.reshape(stddev, [1, 1, 1, -1])
        noise_map = tf.random_normal(tf.shape(inputs)) * stddev
        return noise_map


def gaussian_poisson_noise(inputs, stddev_s=None, stddev_c=None,
                           max_s=0.16, max_c=0.06):
    noise = poisson_noise(inputs, stddev_s, max_s)
    return noise + gaussian_noise(inputs, stddev_c, max_c)


def tf_gaussian_poisson_noise(inputs, stddev_s=None, stddev_c=None,
                              max_s=0.16, max_c=0.06):
    with tf.name_scope('PoissonGaussianNoise'):
        ns = tf_poisson_noise(inputs, stddev_s, max_s)
        nc = tf_gaussian_noise(inputs, stddev_c, max_c)
        return ns + nc
