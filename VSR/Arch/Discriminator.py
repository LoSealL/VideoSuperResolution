"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 4th 2018

Architectures of common discriminator
"""
import tensorflow as tf
import numpy as np

from ..Framework.LayersHelper import Layers


def _view(inputs, input_shape):
    """view inputs' shape as `input_shape`

    Args:
        inputs: a tensor of any shape
        input_shape: a list of 3 or 4 integers, specify the shape of the inputs.
    Return:
        identified tensor `x` and a boolean `has_shape` whether H and W is None.
    """
    input_shape = list(input_shape)
    if len(input_shape) == 3:
        input_shape.insert(0, -1)
    if len(input_shape) != 4:
        raise ValueError('input_shape must be a vector of 3 or 4 integers,'
                         ' representing [H W C] or [B H W C].'
                         'input_shape: %s' % str(input_shape))
    if input_shape[1] and input_shape[2]:
        if input_shape[0] is None:
            input_shape[0] = -1
        x = tf.reshape(inputs, input_shape)
        has_shape = True
    else:
        has_shape = False
        x = tf.identity(inputs)
    return x, has_shape


def dcgan_d(layers: Layers, input_shape, filters_in=64, times_stride=3,
            bias=True, leaky_alpha=0.2, norm=None, name_or_scope=None):
    """Discriminator similar as DCGAN.
    Ref: [1]
         [2]
         [3]

    Args:
        layers: a parent object (usually a derived SuperResolution class)
        input_shape: a list of 3 or 4 integers, specify the shape of the inputs.
        filters_in: number of filters of the 1st layer.
        times_stride: number of stride (a.k.a downsample) times.
        bias: to add bias before activation or not.
        leaky_alpha: alpha value of `leaky_relu`.
        norm: a string either 'sn' or 'bn' or None, representing spectral norm,
          batch norm respectively.
        name_or_scope: a string or tf.VariableScope.
    """
    if norm:
        bn = np.any([word in norm for word in ('bn', 'batch')])
        sn = np.any([word in norm for word in ('sn', 'spectral')])
    else:
        bn = sn = False

    def critic(inputs, conditions=None):
        with tf.variable_scope(name_or_scope, 'D', reuse=tf.AUTO_REUSE):
            x, has_shape = _view(inputs, input_shape)
            kwargs = dict(use_sn=sn,
                          use_batchnorm=bn,
                          use_bias=bias,
                          kernel_initializer='truncated_normal_0.02')
            ch = filters_in
            for i in range(times_stride):
                x = layers.conv2d(x, ch * (2 ** i), 3, **kwargs)
                x = tf.nn.leaky_relu(x, leaky_alpha)
                x = layers.conv2d(x, ch * (2 ** (i + 1)), 4, 2, **kwargs)
                x = tf.nn.leaky_relu(x, leaky_alpha)
            x = layers.conv2d(x, ch * 8, 3, **kwargs)
            x = tf.nn.leaky_relu(x, leaky_alpha)
            if has_shape:
                x = tf.layers.flatten(x)
            else:
                x = tf.reduce_mean(x, [1, 2])
            x = layers.dense(x, 1, use_sn=sn,
                             kernel_initializer='random_normal_0.02')
            return x

    return critic


def resnet_d(layers: Layers, input_shape, filters_in=64, times_pooling=5,
             norm=None, bias=True, name_or_scope=None):
    """Discriminator similar as SN paper table 4.
    Ref: https://arxiv.org/pdf/1802.05957

    Args:
        layers: a parent object (usually a derived SuperResolution class)
        input_shape: a list of 3 or 4 integers, specify the shape of the inputs.
        filters_in: number of filters of the 1st layer.
        times_pooling: number of pooling (a.k.a downsample) times.
        bias: to add bias before activation or not.
        norm: a string either 'sn' or 'bn' or None, representing spectral norm,
          batch norm respectively.
        name_or_scope: a string or tf.VariableScope.
    """
    if norm:
        bn = np.any([word in norm for word in ('bn', 'batch')])
        sn = np.any([word in norm for word in ('sn', 'spectral')])
    else:
        bn = sn = False

    def critic(inputs, conditions=None):
        with tf.variable_scope(name_or_scope, 'D', reuse=tf.AUTO_REUSE):
            x, has_shape = _view(inputs, input_shape)
            kwargs = dict(use_sn=sn,
                          use_batchnorm=bn,
                          use_bias=bias,
                          activation='relu',
                          placement='front',
                          kernel_initializer='truncated_normal_0.02')
            ch = filters_in
            magic = (1, 2, 4, 4, 8, 8,) + (16,) * times_pooling
            for i in range(times_pooling):
                x = layers.resblock(x, ch * magic[i], 3, **kwargs)
                x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.nn.relu(x)
            if has_shape:
                x = tf.layers.flatten(x)
            else:
                x = tf.reduce_mean(x, [1, 2])
            x = layers.dense(x, 1, use_sn=sn,
                             kernel_initializer='random_normal_0.02')
            return x

    return critic


def projection_d(layers: Layers, input_shape, filters_in=64, times_pooling=5,
                 norm=None, bias=True, name_or_scope=None):
    """Discriminator similar as projection GAN paper.
    Ref: https://arxiv.org/abs/1802.05637

    Args:
        layers: a parent object (usually a derived SuperResolution class)
        input_shape: a list of 3 or 4 integers, specify the shape of the inputs.
        filters_in: number of filters of the 1st layer.
        times_pooling: number of pooling (a.k.a downsample) times.
        bias: to add bias before activation or not.
        norm: a string either 'sn' or 'bn' or None, representing spectral norm,
          batch norm respectively.
        name_or_scope: a string or tf.VariableScope.
    """
    if norm:
        bn = np.any([word in norm for word in ('bn', 'batch')])
        sn = np.any([word in norm for word in ('sn', 'spectral')])
    else:
        bn = sn = False

    def critic(inputs, conditions=None):
        with tf.variable_scope(name_or_scope, 'D', reuse=tf.AUTO_REUSE):
            x, has_shape = _view(inputs, input_shape)
            kwargs = dict(use_sn=sn,
                          use_batchnorm=bn,
                          use_bias=bias,
                          activation='relu',
                          placement='front',
                          kernel_initializer='truncated_normal_0.02')
            ch = filters_in
            magic = (1, 2, 2, 4, 8,) + (16, ) * times_pooling
            scale = layers.scale[0]  # determine position to embed condition
            n_pooling = int(np.log2(scale)) - 1
            for i in range(times_pooling):
                x = layers.resblock(x, ch * magic[i], 3, **kwargs)
                x = tf.layers.average_pooling2d(x, 2, 2)
                if i == n_pooling and conditions is not None:
                    cond = layers.conv2d(x, layers.channel, 3, use_bias=bias,
                                         use_batchnorm=bn, use_sn=sn)
                    cond = tf.matmul(tf.layers.flatten(cond),
                                     tf.layers.flatten(conditions),
                                     transpose_b=True)
                else:
                    cond = None
            x = layers.resblock(x, ch * magic[-1], 3, **kwargs)
            x = tf.nn.relu(x)
            if has_shape:
                x = tf.layers.flatten(x)
            else:
                x = tf.reduce_mean(x, [1, 2])
            x = layers.dense(x, 1, use_sn=sn,
                             kernel_initializer='random_normal_0.02')
            if cond is None:
                cond = tf.zeros_like(x)
            return x + cond

    return critic
