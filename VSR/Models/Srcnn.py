"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 25th 2018

SRCNN mainly for framework tests (ECCV 2014)
Ref https://arxiv.org/abs/1501.00092
"""
from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import bicubic_rescale, to_list
import tensorflow as tf


class SRCNN(SuperResolution):
    """Image Super-Resolution Using Deep Convolutional Networks

    Args:
        layers: number layers to use
        filters: number of filters of conv2d(s)
        kernel: a tuple of integer, representing kernel size of each layer,
          can also be one integer to specify the same size
    """

    def __init__(self, layers=3, filters=64, kernel=(9, 5, 5), name='srcnn', **kwargs):
        super(SRCNN, self).__init__(**kwargs)
        self.name = name
        self.layers = layers
        self.filters = filters
        self.kernel_size = to_list(kernel)
        if len(self.kernel_size) < self.layers:
            self.kernel_size += to_list(kernel[-1], self.layers - len(self.kernel_size))

    def build_graph(self):
        super(SRCNN, self).build_graph()
        with tf.variable_scope(self.name):
            x = self.inputs_preproc[-1]
            # x = bicubic_rescale(x, self.scale)
            f = self.filters
            ks = self.kernel_size
            x = self.relu_conv2d(x, f, ks[0])
            for i in range(1, self.layers - 1):
                x = self.relu_conv2d(x, f, ks[i])
            x = self.conv2d(x, self.channel, ks[-1])
            self.outputs.append(x)

    def build_loss(self):
        with tf.name_scope('loss'):
            y_pred = self.outputs[-1]
            y_true = self.label[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            loss = tf.losses.get_total_loss()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(self.learning_rate)
                self.loss.append(opt.minimize(loss, self.global_steps))

            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
        tf.summary.image('SR', self.outputs[-1], 1)
