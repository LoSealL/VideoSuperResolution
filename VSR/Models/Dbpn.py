"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 15th 2018
Updated Date: June 15th 2018

Deep Back-Projection Networks For Super-Resolution
See https://arxiv.org/abs/1803.02735
"""

from VSR.Framework.SuperResolution import SuperResolution

import tensorflow as tf
import numpy as np


class DBPN(SuperResolution):

    def __init__(self, bp_layers=7, use_dense=True, name='dbpn', **kwargs):
        self.bp = bp_layers
        self.dense = use_dense
        self.filter = 64
        self.name = name
        super(DBPN, self).__init__(**kwargs)
        s0, s1 = self.scale
        assert s0 == s1
        if s0 == 3:
            self.kernel_size = 7
        elif s0 == 2 or s0 == 4 or s0 == 8:
            self.kernel_size = int(4 + 2 * np.log2(s0))

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(DBPN, self).build_graph()
            with tf.variable_scope('FE-Net'):
                x = self.conv2d(self.inputs_preproc[-1], 256, 3, activation='prelu', kernel_initializer='he_normal',
                                kernel_regularizer='l2')
                x = self.conv2d(x, self.filter, 1, activation='prelu', kernel_initializer='he_normal',
                                kernel_regularizer='l2')
            with tf.variable_scope('BP-Net'):
                L, H = [x], []
                for i in range(1, self.bp):
                    t = tf.concat(L, axis=-1) if self.dense else L[-1]
                    H.append(self._up_projection(i, t, self.dense))
                    t = tf.concat(H, axis=-1) if self.dense else H[-1]
                    L.append(self._down_projection(i, t, self.dense))
                t = tf.concat(L, axis=-1) if self.dense else L[-1]
                H.append(self._up_projection(self.bp, t, self.dense))
            x = tf.concat(H, axis=-1)
            with tf.variable_scope('ReconNet'):
                x = self.conv2d(x, 1, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            self.outputs.append(x)

    def build_loss(self):
        with tf.variable_scope('loss'):
            mse, loss = super(DBPN, self).build_loss()
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])

    def _up_projection(self, index, inputs, dense=True):
        with tf.variable_scope(f'UpProj_{index}'):
            if dense:
                L_pre = self.conv2d(inputs, self.filter, 1, activation='prelu', kernel_initializer='he_normal',
                                    kernel_regularizer='l2')
            else:
                L_pre = inputs
            H0 = self.deconv2d(L_pre, self.filter, self.kernel_size, self.scale, activation='prelu',
                               kernel_initializer='he_normal', kernel_regularizer='l2')
            L0 = self.conv2d(H0, self.filter, self.kernel_size, self.scale, activation='prelu',
                             kernel_initializer='he_normal', kernel_regularizer='l2')
            res_cur = L0 - L_pre
            H1 = self.deconv2d(res_cur, self.filter, self.kernel_size, self.scale, activation='prelu',
                               kernel_initializer='he_normal', kernel_regularizer='l2')
            H_cur = H0 + H1
            return H_cur

    def _down_projection(self, index, inputs, dense=True):
        with tf.variable_scope(f'DownProj_{index}'):
            if dense:
                H_pre = self.conv2d(inputs, self.filter, 1, activation='prelu', kernel_initializer='he_normal',
                                    kernel_regularizer='l2')
            else:
                H_pre = inputs
            L0 = self.conv2d(H_pre, self.filter, self.kernel_size, self.scale, activation='prelu',
                             kernel_initializer='he_normal', kernel_regularizer='l2')
            H0 = self.deconv2d(L0, self.filter, self.kernel_size, strides=self.scale, activation='prelu',
                               kernel_initializer='he_normal', kernel_regularizer='l2')
            res_cur = H0 - H_pre
            L1 = self.conv2d(res_cur, self.filter, self.kernel_size, self.scale, activation='prelu',
                             kernel_initializer='he_normal', kernel_regularizer='l2')
            L_cur = L0 + L1
            return L_cur
