"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 15th 2018
Updated Date: Sep 11th 2018

Deep Back-Projection Networks For Super-Resolution (CVPR 2018)
See https://arxiv.org/abs/1803.02735
"""

from VSR.Framework.SuperResolution import SuperResolution

import tensorflow as tf
import numpy as np


class DBPN(SuperResolution):
    """Deep Back-Projection Networks For Super-Resolution

    Args:
        bp_layers: number of back-projection stages
        use_dense: use dense unit or not
        filters: number of filters in primary conv2d(s)
    """

    def __init__(self, bp_layers=7, use_dense=True, filters=64, name='dbpn', **kwargs):
        super(DBPN, self).__init__(**kwargs)
        self.bp = bp_layers
        self.dense = use_dense
        self.filter = filters
        self.name = name
        s0, s1 = self.scale
        assert s0 == s1
        if s0 == 3:
            self.kernel_size = 7
        elif s0 == 2 or s0 == 4 or s0 == 8:
            self.kernel_size = int(4 + 2 * np.log2(s0))

    def _up_projection(self, inputs, dense=True, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'UpProjection'):
            if dense:
                L_pre = self.conv2d(inputs, self.filter, 1, activation='prelu')
            else:
                L_pre = inputs
            H0 = self.deconv2d(L_pre, self.filter, self.kernel_size, self.scale, activation='prelu')
            L0 = self.conv2d(H0, self.filter, self.kernel_size, self.scale, activation='prelu')
            res_cur = L0 - L_pre
            H1 = self.deconv2d(res_cur, self.filter, self.kernel_size, self.scale, activation='prelu')
            H_cur = H0 + H1
            return H_cur

    def _down_projection(self, inputs, dense=True, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'DownProjection'):
            if dense:
                H_pre = self.conv2d(inputs, self.filter, 1, activation='prelu')
            else:
                H_pre = inputs
            L0 = self.conv2d(H_pre, self.filter, self.kernel_size, self.scale, activation='prelu')
            H0 = self.deconv2d(L0, self.filter, self.kernel_size, strides=self.scale, activation='prelu')
            res_cur = H0 - H_pre
            L1 = self.conv2d(res_cur, self.filter, self.kernel_size, self.scale, activation='prelu')
            L_cur = L0 + L1
            return L_cur

    def build_graph(self):
        super(DBPN, self).build_graph()
        with tf.variable_scope(self.name):
            with tf.variable_scope('FE-Net'):
                x = self.conv2d(self.inputs_preproc[-1], 256, 3, activation='prelu')
                x = self.conv2d(x, self.filter, 1, activation='prelu')
            with tf.variable_scope('BP-Net'):
                L, H = [x], []
                for _ in range(1, self.bp):
                    t = tf.concat(L, axis=-1) if self.dense else L[-1]
                    H.append(self._up_projection(t, self.dense))
                    t = tf.concat(H, axis=-1) if self.dense else H[-1]
                    L.append(self._down_projection(t, self.dense))
                t = tf.concat(L, axis=-1) if self.dense else L[-1]
                H.append(self._up_projection(t, self.dense))
            x = tf.concat(H, axis=-1)
            with tf.variable_scope('ReconNet'):
                x = self.conv2d(x, self.channel, 3)
            self.outputs.append(x)

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
