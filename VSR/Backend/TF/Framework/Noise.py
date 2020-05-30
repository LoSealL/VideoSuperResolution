"""
Copyright: Wenyi Tang 2017-2020
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 25th 2018

Utility for complicated noise model
Ref:
[1] https://arxiv.org/abs/1807.04686
"""

from .. import tf


def tf_camera_response_function(inputs, crf_table, max_val=1):
  """TF implementation of CRF."""

  with tf.name_scope('CRF'):
    inputs_norm = tf.clip_by_value(inputs / max_val, 0, 1)
    quant = int(crf_table.shape[0] - 1)
    inputs_index = tf.to_int32(inputs_norm * quant)
    return tf.nn.embedding_lookup(crf_table, inputs_index)


def tf_poisson_noise(inputs, stddev=None, sigma_max=0.16):
  with tf.name_scope('PoissonNoise'):
    if stddev is None:
      stddev = tf.random_uniform(inputs.shape[-1:], maxval=sigma_max)
    stddev = tf.reshape(stddev, [1, 1, 1, -1])
    sigma_map = (1 - inputs) * stddev
    return tf.random_normal(tf.shape(inputs)) * sigma_map


def tf_gaussian_noise(inputs, stddev=None, sigma_max=0.06, channel_wise=True):
  with tf.name_scope('GaussianNoise'):
    channel = inputs.shape[-1:] if channel_wise else [1]
    if stddev is None:
      stddev = tf.random_uniform(channel, maxval=sigma_max)
    stddev = tf.reshape(stddev, [1, 1, 1, -1])
    noise_map = tf.random_normal(tf.shape(inputs)) * stddev
    return noise_map


def tf_gaussian_poisson_noise(inputs, stddev_s=None, stddev_c=None,
                              max_s=0.16, max_c=0.06):
  with tf.name_scope('PoissonGaussianNoise'):
    ns = tf_poisson_noise(inputs, stddev_s, max_s)
    nc = tf_gaussian_noise(inputs, stddev_c, max_c)
    return ns + nc
