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
        super(ESPCN, self).build_graph()
        with tf.variable_scope(self.name):
            x = self.inputs_preproc[-1] / 255.0
            x = self.tanh_conv2d(x, 64, 5)
            for _ in range(1, self.layers - 1):
                x = self.tanh_conv2d(x, 32, 3)
            x = self.upscale(x)
            self.outputs.append(x * 255.0)

    def build_loss(self):
        with tf.name_scope('loss'):
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
