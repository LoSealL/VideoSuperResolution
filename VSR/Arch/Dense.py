"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 17th 2018

Architectures of common dense blocks used in SR researches
"""

from ..Framework.LayersHelper import Layers
import tensorflow as tf


def dense_block(layers: Layers, inputs, depth=8, rate=16, out_dims=128,
                scope=None, reuse=None):
    filters = out_dims - rate * depth
    feat = [inputs]
    with tf.variable_scope(scope, 'DenseBlock', reuse=reuse):
        for _ in range(depth):
            filters += rate
            x = layers.relu_conv2d(feat[-1], filters, 3)
            feat.append(x)
            feat[-1] = tf.concat(feat[1:], axis=-1)
        return x
