"""
Copyright: Wenyi Tang 2017-2018
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
    """Fast and Accurate Single Image Super-Resolution via Information Distillation Network

    Args:
        blocks: number of distillation blocks
        filters: number of filters in distillation blocks
        delta: according to paper, D = D3 - D1 = D1 - D2 = D6 - D4 = D4 - D5,
          where D3=D4, D_{i} is the filters of i-th conv2d
        slice_factor: the number of channels sliced out from the 3rd conv2d
        leaky_slope: slope used in leaky relu activators
        fine_tune_epoch: epoch number beyond which use L1 loss to replace L2 loss
    """

    def __init__(self, blocks=4, filters=64, delta=16, slice_factor=4, leaky_slope=0.05,
                 fine_tune_epoch=200, name='idn', **kwargs):
        super(InformationDistillationNetwork, self).__init__(**kwargs)
        self.blocks = blocks
        self.F = filters
        self.d = delta
        self.s = slice_factor
        self.leaky_slope = leaky_slope
        self.fine_tune = fine_tune_epoch
        self.name = name

    def _idn(self, inputs, D3=64, d=16, s=4, **kwargs):
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
        with tf.variable_scope(kwargs.get('name'), 'Enhancement'):
            x = inputs
            for _d in D[:3]:
                x = self.conv2d(x, _d, 3)
                x = tf.nn.leaky_relu(x, self.leaky_slope)
            R, P2 = x[..., :D3 // s], x[..., D3 // s:]
            x = P2
            for _d in D[3:]:
                x = self.conv2d(x, _d, 3)
                x = tf.nn.leaky_relu(x, self.leaky_slope)
            x += tf.concat([inputs, R], axis=-1)
        with tf.variable_scope(kwargs.get('name'), 'Compression'):
            outputs = self.conv2d(x, D3, 1)
        return outputs

    def _mse_weight_decay_fn(self, step):
        if step < self.fine_tune:
            return 1.0
        else:
            return 0

    def build_graph(self):
        super(InformationDistillationNetwork, self).build_graph()
        with tf.variable_scope(self.name):
            x = self.inputs_preproc[-1] / 255
            with tf.variable_scope('Features'):
                x = self.conv2d(x, self.F, 3)
                x = tf.nn.leaky_relu(x, self.leaky_slope)
                x = self.conv2d(x, self.F, 3)
                x = tf.nn.leaky_relu(x, self.leaky_slope)
            with tf.variable_scope('Distillation'):
                for _ in range(self.blocks):
                    x = self._idn(x, self.F, self.d, self.s)
            with tf.variable_scope('Reconstruction'):
                x = self.deconv2d(x, self.channel, 17, strides=self.scale)
            self.outputs.append(x * 255)

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
        epoch = kwargs.get('epochs')
        self.feed_dict.update({'mse_weight:0': self._mse_weight_decay_fn(epoch)})
        return super(InformationDistillationNetwork, self).train_batch(feature, label, learning_rate, **kwargs)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/mae', self.metrics['mae'])
        tf.summary.scalar('metric/psnr', self.metrics['psnr'])
        tf.summary.scalar('metric/ssim', self.metrics['ssim'])
