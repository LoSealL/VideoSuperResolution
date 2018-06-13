"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 12th 2018
Updated Date: May 25th 2018

Spatial transformer motion compensation model
Ref https://arxiv.org/abs/1609.05158
"""
from ..Framework.SuperResolution import SuperResolution
from ..Util import Utility

import tensorflow as tf
import numpy as np


class ESPCN(SuperResolution):

    def __init__(self, scale, layers=3, name='espcn', **kwargs):
        self.layers = layers
        self.name = name
        super(ESPCN, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            self.inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input/lr/gray'))
            self.inputs_preproc = self.inputs
            l2_decay = 1e-4
            x = tf.layers.conv2d(self.inputs_preproc[-1], 64, 5, padding='same', activation=tf.nn.tanh,
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            for _ in range(1, self.layers - 1):
                x = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.tanh,
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            x = tf.layers.conv2d(x, self.scale[0] * self.scale[1], 3, padding='same',
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            x = Utility.pixel_shift(x, self.scale, 1)
            self.outputs.append(x)

    def build_loss(self):
        with tf.variable_scope('loss'):
            self.label.append(tf.placeholder(tf.float32, shape=[None, None, None, 1]))
            y_true = self.label[-1]
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            tv_decay = 1e-4
            tv_loss = tv_decay * tf.reduce_mean(tf.image.total_variation(y_pred))
            regular_loss = tf.add_n(tf.losses.get_regularization_losses()) + tv_loss
            loss = mse + regular_loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.loss.append(optimizer.minimize(loss, self.global_steps))
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['regularization'] = regular_loss
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
