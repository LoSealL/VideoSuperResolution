"""
Copyright: Wenyi Tang 2017-2018
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
    """Accurate Image Super-Resolution Using Very Deep Convolutional Networks

    Args:
        layers: number of conv2d layers
        filters: number of filters in conv2d(s)
    """

    def __init__(self, layers=20, filters=64, name='vdsr', **kwargs):
        self.layers = layers
        self.filters = filters
        self.name = name
        super(VDSR, self).__init__(**kwargs)

    def build_graph(self):
        super(VDSR, self).build_graph()
        with tf.variable_scope(self.name):
            # bicubic upscale
            # bic = bicubic_rescale(self.inputs_preproc[-1], self.scale)
            x = self.inputs_preproc[-1]
            for _ in range(self.layers - 1):
                x = self.relu_conv2d(x, self.filters, 3)
            x = self.conv2d(x, self.channel, 3)
            self.outputs.append(x + self.inputs_preproc[-1])

    def build_loss(self):
        with tf.name_scope('loss'):
            mae = tf.losses.absolute_difference(self.label[-1], self.outputs[-1])
            loss = tf.losses.get_total_loss()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(self.learning_rate)
                self.loss.append(opt.minimize(loss, self.global_steps))

            self.train_metric['loss'] = loss
            self.metrics['mae'] = mae
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mae', self.metrics['mae'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
