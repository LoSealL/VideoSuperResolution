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
from VSR.Util.Utility import *

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
        shape = tf.shape(x)
        shape_enlarge = shape * [1, *self.scale, 1]
        x = tf.image.resize_bicubic(x, shape_enlarge[1:3], name='bicubic')
        nn = list()
        nn.append(tf.layers.Conv2D(64, 9, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer))
        nn.append(tf.layers.Conv2D(32, 5, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer))
        nn.append(tf.layers.Conv2D(1, 5, 1, padding='same',
                                   kernel_initializer=he_initializer))
        for _n in nn:
            x = _n(x)
            self.trainable_weights += [_n.kernel]
        self.outputs.append(x)

    def build_loss(self):
        self.label.append(tf.placeholder(
            tf.uint8, shape=[None, None, None, 1], name='input_label_gray'))
        y_true = tf.cast(self.label[-1], tf.float32, name='cast/input_label')
        diff = y_true - self.outputs[-1]
        mse = tf.reduce_mean(tf.square(diff), name='loss/mse')
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_weights])
        loss = mse + 0.0001 * l2_loss
        optimizer = tf.train.AdamOptimizer(1e-5)
        self.loss.append(optimizer.minimize(loss))
        self.metrics['mse'] = mse
        self.metrics['l2'] = l2_loss
        self.metrics['psnr'] = 20 * tf.log(255.0 / tf.sqrt(mse)) / tf.log(10.0)
        self.diff = diff

    def build_summary(self):
        tf.summary.scalar('loss/psnr', self.metrics['psnr'])
        tf.summary.scalar('loss/l2', self.metrics['l2'])
        tf.summary.image('diff', self.diff, 5)
