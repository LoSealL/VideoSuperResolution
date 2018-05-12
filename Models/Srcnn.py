"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 8th 2018

SRCNN mainly for framework tests
Ref https://arxiv.org/abs/1501.00092
"""
from VSR.Framework.SuperResolution import SuperResolution

import tensorflow as tf
import numpy as np


class SRCNN(SuperResolution):

    def __init__(self, scale, name='srcnn', **kwargs):
        self.name = name
        super(SRCNN, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        self.inputs.append(tf.placeholder(
            tf.uint8, shape=[None, None, None, 1], name='input_lr_gray'))
        x = tf.cast(self.inputs[-1], tf.float32, name='cast/input_lr')
        x = x / 255.0
        shape = tf.shape(x)
        shape_enlarge = shape * [1, *self.scale, 1]
        x = tf.image.resize_bicubic(x, shape_enlarge[1:3], name='bicubic')
        x = self.build_conv2d(64, 9, 1, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2 / 81)))(x)
        x = self.build_conv2d(32, 1, 1, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2 / 1)))(x)
        x = self.build_conv2d(1, 5, 1, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.truncated_normal(stddev=np.sqrt(2 / 25)))(x)
        self.outputs.append(x * 255.0)

    def build_loss(self):
        self.label.append(tf.placeholder(
            tf.uint8, shape=[None, None, None, 1], name='input_label_gray'))
        y_true = tf.cast(self.label[-1], tf.float32, name='cast/input_label')
        y_true /= 255.0
        diff = y_true - self.outputs[-1]
        mse = tf.reduce_mean(tf.square(diff), name='loss/mse')
        optimizer = tf.train.AdamOptimizer()
        self.loss.append(optimizer.minimize(mse))
        self.metrics['mse'] = mse

    def build_summary(self):
        tf.summary.scalar('always_zero', tf.constant(0))
        tf.summary.scalar('loss/mse', self.metrics['mse'])
