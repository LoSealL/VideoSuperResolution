"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 5th 2018
Updated Date: June 15th 2018

Accurate Image Super-Resolution Using Very Deep Convolutional Networks
See https://arxiv.org/abs/1511.04587
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import bicubic_rescale
import tensorflow as tf


class VDSR(SuperResolution):

    def __init__(self, layers=20, filters=64, name='vdsr', **kwargs):
        self.layers = layers
        self.filters = filters
        self.name = name
        super(VDSR, self).__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(VDSR, self).build_graph()
            # bicubic upscale
            bic = bicubic_rescale(self.inputs_preproc[-1], self.scale)
            x = bic
            for _ in range(self.layers - 1):
                x = self.conv2d(x, self.filters, 3, activation='relu', kernel_initializer='he_normal',
                                kernel_regularizer='l2')
            x = self.conv2d(x, 1, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            self.outputs.append(x + bic)

    def build_loss(self):
        with tf.variable_scope('loss'):
            mse, loss = super(VDSR, self).build_loss()
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
