"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 11th 2018

Residual Channel Attention Networks (ECCV 2018)
See https://arxiv.org/abs/1807.02758
"""

from VSR.Framework.SuperResolution import SuperResolution

import tensorflow as tf


class RCAN(SuperResolution):
    """Image Super-Resolution Using Very Deep Residual Channel Attention Networks

    Args:
        channel_downscaling: channel downscaling ratio as `r` in the paper
        n_rcab: number of RCABs in each RG(Residual Group)
        n_rg: number of RGs in RIR(Residual in Residual)
        filters: number of filters in primary conv2d(s)
    """

    def __init__(self, name='rcan',
                 channel_downscaling=16,
                 n_rcab=20,
                 n_rg=10,
                 filters=64,
                 **kwargs):
        super(RCAN, self).__init__(**kwargs)
        self.name = name
        self.R = channel_downscaling
        self.n_rcab = n_rcab
        self.n_rg = n_rg
        self.F = filters

    def _rir(self, inputs, **kwargs):
        """Residual in residual block"""
        with tf.variable_scope(kwargs.get('name'), 'RIR'):
            x = inputs
            for _ in range(self.n_rg):
                x = self._rg(x)
            x = self.conv2d(x, self.F, 3)
            # LCC
            return inputs + x

    def _rg(self, inputs, **kwargs):
        """Residual group"""
        with tf.variable_scope(kwargs.get('name'), 'RG'):
            x = inputs
            for _ in range(self.n_rcab):
                x = self._rcab(x)
            x = self.conv2d(x, self.F, 3)
            # SCC
            return inputs + x

    def _rcab(self, inputs, **kwargs):
        """Residual channel attention block"""
        with tf.variable_scope(kwargs.get('name'), 'RCAB'):
            x = self.relu_conv2d(inputs, self.F, 3)
            y = self.conv2d(x, self.F, 3)
            x = tf.reduce_mean(y, axis=[1, 2], keepdims=True)
            x = self.relu_conv2d(x, self.F // self.R, 1)
            x = self.conv2d(x, self.F, 1, activation=tf.nn.sigmoid)
            y *= x
            return inputs + y

    def build_graph(self):
        super(RCAN, self).build_graph()
        with tf.variable_scope(self.name):
            x = self.inputs_preproc[-1] / 255
            sf = self.conv2d(x, self.F, 3)
            df = self._rir(sf)
            sr = self.upscale(df, direct_output=False)
            sr = self.conv2d(sr, self.channel, 3)
            self.outputs.append(sr * 255)

    def build_loss(self):
        with tf.name_scope('Loss'):
            l1_loss = tf.losses.absolute_difference(self.label[0], self.outputs[0])
            re_loss = tf.losses.get_regularization_losses()
            mse = tf.losses.mean_squared_error(self.label[0], self.outputs[0])
            loss = tf.add_n(re_loss + [l1_loss], name='Loss')

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(opt)

        # tensorboard
        self.train_metric['loss'] = loss
        self.train_metric['l1'] = l1_loss
        self.metrics['mse'] = mse
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[0], self.outputs[0], 255))
        self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[0], self.outputs[0], 255))

    def build_summary(self):
        tf.summary.scalar('loss', self.train_metric['loss'])
        tf.summary.scalar('mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
        tf.summary.image('SR', self.outputs[0], 1)
