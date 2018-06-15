"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 25th 2018

SRCNN mainly for framework tests
Ref https://arxiv.org/abs/1501.00092
"""
from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import bicubic_rescale
import tensorflow as tf


class SRCNN(SuperResolution):

    def __init__(self, scale, layers=3, filters=64, kernel=(9, 5, 5), name='srcnn', **kwargs):
        self.name = name
        self.layers = layers
        self.filters = filters
        self.kernel_size = kernel
        super(SRCNN, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            self.inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input/lr/gray'))
            self.inputs_preproc = self.inputs
            x = bicubic_rescale(self.inputs_preproc[-1], self.scale)
            f = self.filters
            ks = self.kernel_size
            x = self.conv2d(x, f, ks[0], activation='relu', kernel_regularizer='l2', kernel_initializer='he_normal')
            for i in range(1, self.layers - 1):
                x = self.conv2d(x, f, ks[i], activation='relu', kernel_regularizer='l2',
                                kernel_initializer='he_normal')
            x = self.conv2d(x, 1, ks[-1], kernel_regularizer='l2', kernel_initializer='he_normal')
            self.outputs.append(x)

    def build_loss(self):
        with tf.variable_scope('loss'):
            mse, loss = super(SRCNN, self).build_loss()
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
