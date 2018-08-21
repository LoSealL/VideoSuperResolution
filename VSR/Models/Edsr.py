"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 15th 2018
Updated Date: June 15th 2018

Enhanced Deep Residual Networks for Single Image Super-Resolution
See https://arxiv.org/abs/1707.02921
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import pixel_shift
import tensorflow as tf


class EDSR(SuperResolution):

    def __init__(self, layers=32, filters=256, clip=0.1, name='edsr', **kwargs):
        self.layers = layers
        self.filters = filters
        self.clip = clip
        self.name = name
        super(EDSR, self).__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(EDSR, self).build_graph()
            fe = self.conv2d(self.inputs_preproc[-1], self.filters, 3,
                             kernel_initializer='he_normal', kernel_regularizer='l2')
            x = fe
            for i in range(self.layers):
                x_old = x
                x = self.conv2d(x, self.filters, 3, activation='relu', kernel_initializer='he_normal',
                                kernel_regularizer='l2')
                x = self.conv2d(x, self.filters, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
                x *= self.clip
                x += x_old
            x = self.conv2d(x, self.filters, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            x += fe
            x = self.upscale(x)
            x = self.conv2d(x, self.channel, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            self.outputs.append(x)

    def build_loss(self):
        with tf.name_scope('loss'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            mse = tf.losses.mean_squared_error(self.label[-1], self.outputs[-1])
            mae = tf.losses.absolute_difference(self.label[-1], self.outputs[-1])
            loss = tf.add_n([mae] + tf.losses.get_regularization_losses(), name='total_loss')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.loss.append(opt.minimize(loss, self.global_steps))

            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['mae'] = mae
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/mae', self.metrics['mae'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
