"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 24th 2018
Updated Date: May 25th 2018

Architecture of Residual Dense Network (CVPR 2018)
See https://arxiv.org/abs/1802.08797
"""

from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util import *

import tensorflow as tf
import numpy as np


class ResidualDenseNetwork(SuperResolution):

    def __init__(self, scale, rgb_input=False, name='rdn',
                 global_filters=64, rdb_blocks=10, rdb_conv=6, rdb_filters=64,
                 **kwargs):
        """

        Args:
            conv: number of convolutional networks
            filters: number of feature maps
        """
        self.name = name
        self.gfilter = global_filters
        self.block = rdb_blocks
        self.conv = rdb_conv
        self.growth_rate = rdb_filters
        super(ResidualDenseNetwork, self).__init__(scale=scale, rgb_input=rgb_input, **kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(ResidualDenseNetwork, self).build_graph()
            x = self.inputs_preproc[-1]
            # shallow feature extraction
            # NOTE: no activation
            with tf.variable_scope('sfe'):
                sf0 = tf.layers.conv2d(self.inputs_preproc[-1], self.gfilter, 3, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       kernel_initializer=tf.keras.initializers.he_normal())
                sf1 = tf.layers.conv2d(sf0, self.gfilter, 3, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       kernel_initializer=tf.keras.initializers.he_normal())
            with tf.variable_scope('blocks'):
                F = [sf1]
                for i in range(self.block):
                    F += [self._make_rdb(F[-1])]
            with tf.variable_scope('gf'):
                gf0 = tf.layers.conv2d(tf.concat(F[1:], axis=-1), self.gfilter, 1, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       kernel_initializer=tf.keras.initializers.he_normal())
                gf1 = tf.layers.conv2d(gf0, self.gfilter, 3, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       kernel_initializer=tf.keras.initializers.he_normal())
            dense_feature = sf0 + gf1
            # use pixel shift in ESPCN to upscale
            upscaled = tf.layers.conv2d(dense_feature, self.scale[0] * self.scale[1], 3, padding='same',
                                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                        kernel_initializer=tf.keras.initializers.he_normal())
            upscaled = Utility.pixel_shift(upscaled, self.scale, 1)
            hr = tf.layers.conv2d(upscaled, 1, 3, padding='same', kernel_initializer=tf.keras.initializers.he_normal())
            self.outputs.append(hr)

    def build_loss(self):
        """In paper, authors use L1 loss instead of MSE error. Claimed a better perf."""
        with tf.variable_scope('loss'):
            self.label.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1]))
            y_true = tf.cast(self.label[-1], tf.float32)
            y_pred = self.outputs[-1]
            mae = tf.losses.absolute_difference(y_true, self.outputs[-1])
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            regularization = tf.add_n(tf.losses.get_regularization_losses())
            loss = mae + regularization
            self.loss.append(opt.minimize(loss, self.global_steps))
            self.metrics['mse'] = mse
            self.metrics['mae'] = mae
            self.metrics['psnr'] = tf.image.psnr(y_true, self.outputs[-1], 255)
            self.metrics['ssim'] = tf.image.ssim(y_true, self.outputs[-1], 255)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/mae', self.metrics['mae'])
        tf.summary.scalar('metric/psnr', tf.reduce_mean(self.metrics['psnr']))
        tf.summary.scalar('metric/ssim', tf.reduce_mean(self.metrics['ssim']))

    def _make_rdb(self, inputs):
        """Make Residual Dense Block

        Args:
            inputs: input features
        """

        filters, conv = self.growth_rate, self.conv
        x = [inputs]
        with tf.variable_scope('rdb'):
            x += [tf.layers.conv2d(x[-1], filters, 3, padding='same', activation=tf.nn.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                   kernel_initializer=tf.keras.initializers.he_normal())]
            for i in range(1, conv):
                x += [tf.layers.conv2d(tf.concat(x, axis=-1), filters, 3, padding='same', activation=tf.nn.relu,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       kernel_initializer=tf.keras.initializers.he_normal())]
            # 1x1 conv
            local_fusion = tf.layers.conv2d(tf.concat(x, axis=-1), filters, 1, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                            kernel_initializer=tf.keras.initializers.he_normal())
            # local residual learning
            outputs = inputs + local_fusion
            return outputs
