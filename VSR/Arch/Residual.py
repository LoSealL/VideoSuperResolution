"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 17th 2018

Architectures of common residual blocks used in SR researches
"""
from ..Framework.LayersHelper import Layers

import tensorflow as tf


def rcab(layers: Layers, inputs,
         filters=64, ratio=16, scope=None, reuse=None):
    """Residual channel attention block (ECCV18)

    """
    with tf.variable_scope(scope, 'RCAB', reuse=reuse):
        pre_input = inputs
        x = layers.relu_conv2d(inputs, filters, 3)
        y = layers.conv2d(x, filters, 3)
        x = tf.reduce_mean(y, axis=[1, 2], keepdims=True)
        x = layers.relu_conv2d(x, filters // ratio, 1)
        x = layers.conv2d(x, filters, 1, activation=tf.nn.sigmoid)
        y *= x
        y += pre_input
    return y


def msrb(layers: Layers, inputs, filters=64, scope=None, reuse=None):
    """Multi-scale residual block (ECCV18)

    """
    with tf.variable_scope(scope, 'MSRB', reuse=reuse):
        pre_input = inputs
        s1 = layers.relu_conv2d(inputs, filters, 3)
        p1 = layers.relu_conv2d(inputs, filters, 5)

        s2 = layers.relu_conv2d(tf.concat([s1, p1], -1), filters * 2, 3)
        p2 = layers.relu_conv2d(tf.concat([p1, s1], -1), filters * 2, 5)

        s = layers.conv2d(tf.concat([s2, p2], -1), filters, 1)
        s += pre_input
    return s


def cascade_block(layers: Layers, inputs,
                  filters=64, depth=4, scope=None, reuse=None):
    """Cascading residual block (ECCV18)

    """
    with tf.variable_scope(scope, 'CARB', reuse=reuse):
        feat = [inputs]
        for i in range(depth):
            x = layers.resblock(inputs, filters, 3, activation='relu')
            feat.append(x)
            inputs = layers.conv2d(
                tf.concat(feat, axis=-1), filters, 1,
                kernel_initializer='he_uniform')
        inputs = layers.conv2d(inputs, filters, 3)
        return inputs
