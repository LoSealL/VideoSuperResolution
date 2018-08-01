"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 24th 2018
Updated Date: May 25th 2018

Architecture of Residual Dense Network (CVPR 2018)
See https://arxiv.org/abs/1802.08797
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import pixel_shift

import tensorflow as tf


class ResidualDenseNetwork(SuperResolution):

    def __init__(self, global_filters=64, rdb_blocks=10, rdb_conv=6, rdb_filters=64,
                 name='rdn', **kwargs):
        self.name = name
        self.gfilter = global_filters
        self.block = rdb_blocks
        self.conv = rdb_conv
        self.growth_rate = rdb_filters
        super(ResidualDenseNetwork, self).__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(ResidualDenseNetwork, self).build_graph()
            x = self.inputs_preproc[-1]
            # shallow feature extraction
            # NOTE: no activation
            with tf.variable_scope('ShallowFeature'):
                sf0 = self.conv2d(x, self.gfilter, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
                sf1 = self.conv2d(sf0, self.gfilter, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
            with tf.variable_scope('ResBlocks'):
                F = [sf1]
                for i in range(self.block):
                    F += [self._make_rdb(F[-1])]
            with tf.variable_scope('GlobalFusion'):
                gf0 = self.conv2d(tf.concat(F[1:], axis=-1), self.gfilter, 1, kernel_initializer='he_normal', kernel_regularizer='l2')
                gf1 = self.conv2d(gf0, self.gfilter, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
            dense_feature = sf0 + gf1
            # use pixel shift in ESPCN to upscale
            upscaled = self.upscale(dense_feature, 'espcn')
            hr = self.conv2d(upscaled, self.channel, 3, kernel_initializer=tf.keras.initializers.he_normal())
            self.outputs.append(hr)

    def build_loss(self):
        """In paper, authors use L1 loss instead of MSE error. Claimed a better perf."""
        with tf.name_scope('loss'):
            y_true = self.label[-1]
            y_pred = self.outputs[-1]
            mae = tf.losses.absolute_difference(y_true, y_pred)
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            loss = tf.add_n([mae] + tf.losses.get_regularization_losses())
            self.loss.append(opt.minimize(loss, self.global_steps))
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['mae'] = mae
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(y_true, self.outputs[-1], 255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(y_true, self.outputs[-1], 255))

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/mae', self.metrics['mae'])
        tf.summary.scalar('metric/psnr', self.metrics['psnr'])
        tf.summary.scalar('metric/ssim', self.metrics['ssim'])

    def _make_rdb(self, inputs):
        """Make Residual Dense Block

        Args:
            inputs: input features
        """

        filters, conv = self.growth_rate, self.conv
        x = [inputs]
        x += [self.conv2d(x[-1], filters, 3, activation='relu', kernel_regularizer='l2', kernel_initializer='he_normal')]
        for i in range(1, conv):
            x += [self.conv2d(tf.concat(x, axis=-1), filters, 3, activation='relu', kernel_regularizer='l2', kernel_initializer='he_normal')]
        # 1x1 conv
        local_fusion = self.conv2d(tf.concat(x, axis=-1), filters, 1, kernel_regularizer='l2', kernel_initializer='he_normal')
        # local residual learning
        outputs = inputs + local_fusion
        return outputs
