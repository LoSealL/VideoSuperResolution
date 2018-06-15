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
            tf.summary.image('input', self.inputs[-1], 1)
            x = self.inputs_preproc[-1] / 255.0
            y_near = tf.concat([x] * self.scale[0] * self.scale[1], -1)
            y_near = Utility.pixel_shift(y_near, self.scale, 1)
            x = self.conv2d(x, 64, 5, activation=tf.nn.tanh,
                            kernel_initializer='he_normal',
                            kernel_regularizer='l2')
            for i in range(64):
                tf.summary.image('layer/0', x[..., i:i + 1], 1)
            for _ in range(1, self.layers - 1):
                x = self.conv2d(x, 32, 3, activation=tf.nn.tanh, kernel_initializer='he_normal',
                                kernel_regularizer='l2')
                for i in range(32):
                    tf.summary.image(f'layer/{_}', x[..., i:i + 1], 1)

            x = self.conv2d(x, self.scale[0] * self.scale[1], 3, kernel_initializer='he_normal',
                            kernel_regularizer='l2')
            for i in range(self.scale[0] * self.scale[1]):
                tf.summary.image('layer/99', x[..., i:i + 1], 1)
            x = Utility.pixel_shift(x, self.scale, 1)
            # x = tf.nn.tanh(x)
            x += y_near
            tf.summary.image('feature/output', x, 1)
            self.outputs.append(x * 255.0)

    def build_loss(self):
        with tf.variable_scope('loss'):
            mse, loss = super(ESPCN, self).build_loss()
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('training_loss', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
