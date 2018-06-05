"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 25th 2018

SRCNN mainly for framework tests
Ref https://arxiv.org/abs/1501.00092
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util.Utility import *

import tensorflow as tf
import numpy as np


class SRCNN(SuperResolution):

    def __init__(self, scale, layers=3, name='srcnn', **kwargs):
        self.name = name
        self.layers = layers
        super(SRCNN, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(SRCNN, self).build_graph()
            shape = tf.shape(self.inputs_preproc[-1])
            shape_enlarge = shape * [1, *self.scale, 1]
            l2_decay = 1e-4
            x = tf.image.resize_bicubic(self.inputs_preproc[-1], shape_enlarge[1:3], name='bicubic')
            x = tf.layers.conv2d(x, 64, 9, padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            for _ in range(1, self.layers - 1):
                x = tf.layers.conv2d(x, 32, 5, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            x = tf.layers.conv2d(x, 1, 5, padding='same',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            self.outputs.append(x)

    def build_loss(self):
        with tf.variable_scope('loss'):
            self.label.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1]))
            y_true = tf.cast(self.label[-1], tf.float32)
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            tv_decay = 1e-4
            tv_loss = tv_decay * tf.reduce_mean(tf.image.total_variation(y_pred))
            regular_loss = tf.add_n(tf.losses.get_regularization_losses()) + tv_loss
            loss = mse + regular_loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.loss.append(optimizer.minimize(loss, self.global_steps))
            self.metrics['mse'] = mse
            self.metrics['regularization'] = regular_loss
            self.metrics['psnr'] = tf.image.psnr(y_true, y_pred, max_val=255)
            self.metrics['ssim'] = tf.image.ssim(y_true, y_pred, max_val=255)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', tf.reduce_mean(self.metrics['psnr']))
        tf.summary.scalar('ssim', tf.reduce_mean(self.metrics['ssim']))
