"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 24th 2018
Updated Date: June 5th 2018

Architecture of Information Distillation Network (CVPR 2018)
See https://arxiv.org/abs/1803.09454
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util import *

import tensorflow as tf


class InformationDistillationNetwork(SuperResolution):

    def __init__(self, scale,
                 blocks=4,
                 filters=64,
                 delta=16,
                 slice_factor=4,
                 leaky_slope=0.05,
                 fine_tune=100000,
                 name='idn',
                 **kwargs):
        self.blocks = blocks
        self.D = filters
        self.d = delta
        self.s = slice_factor
        self.leaky_slope = leaky_slope
        self.fine_tune = fine_tune
        self.name = name
        super(InformationDistillationNetwork, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(InformationDistillationNetwork, self).build_graph()
            x = self.inputs_preproc[-1]
            with tf.variable_scope('feature_blocks'):
                x = self.conv2d(x, self.D, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
                x = tf.nn.leaky_relu(x, self.leaky_slope)
                x = self.conv2d(x, self.D, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
                x = tf.nn.leaky_relu(x, self.leaky_slope)
            with tf.variable_scope('distillation_blocks'):
                for i in range(self.blocks):
                    x = self._make_idn(i, x, self.D, self.d, self.s)
            with tf.variable_scope('reconstruction'):
                x = self.deconv2d(x, 1, 17, strides=self.scale, kernel_regularizer='l2', kernel_initializer='he_normal')
            self.outputs.append(x)

    def build_loss(self):
        """The paper first use MSE to train network, then use MAE to fine-tune it

        """
        w = tf.placeholder(tf.float32, name='mse_weight')
        with tf.name_scope('loss'):
            y_true = self.label[-1]
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred, weights=w)
            mae = tf.losses.absolute_difference(y_true, y_pred, weights=(1 - w))
            loss = tf.add_n([mse, mae] + tf.losses.get_regularization_losses())
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.loss.append(optimizer.minimize(loss, self.global_steps))
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['mae'] = mae
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        self.feed_dict.update({'mse_weight:0': self._mse_weight_decay_fn()})
        return super(InformationDistillationNetwork, self).train_batch(feature, label, learning_rate, **kwargs)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/mae', self.metrics['mae'])
        tf.summary.scalar('metric/psnr', self.metrics['psnr'])
        tf.summary.scalar('metric/ssim', self.metrics['ssim'])

    def _make_idn(self, index, inputs, D3=64, d=16, s=4):
        """ the information distillation block contains:
                - enhancement unit
                - compression unit

            Args:
                inputs: input feature maps
                D3: filters of the 3rd conv2d
                d: according to paper, D = D3 - D1 = D1 - D2 = D6 - D4 = D4 - D5,
                   where D3=D4, D_{i} is the filters of i-th conv2d
                s: s is the number of channels sliced out from the 3rd conv2d
        """
        D1 = D3 - d
        D2 = D1 - d
        D4 = D3
        D5 = D4 - d
        D6 = D4 + d
        D = [D1, D2, D3, D4, D5, D6]
        with tf.variable_scope(f'enhancement_{index}'):
            x = inputs
            for _D in D[:3]:
                x = self.conv2d(x, _D, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
                x = tf.nn.leaky_relu(x, self.leaky_slope)
            R, P2 = x[..., :D3 // s], x[..., D3 // s:]
            x = P2
            for _D in D[3:]:
                x = self.conv2d(x, _D, 3, kernel_regularizer='l2', kernel_initializer='he_normal')
                x = tf.nn.leaky_relu(x, self.leaky_slope)
            x += tf.concat([inputs, R], axis=-1)
        with tf.variable_scope(f'compression_{index}'):
            outputs = self.conv2d(x, D3, 1, kernel_regularizer='l2', kernel_initializer='he_normal')
        return outputs

    def _mse_weight_decay_fn(self):
        if self.global_steps.eval() < self.fine_tune:
            return 1.0
        else:
            return 0
