"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 12th 2018
Updated Date: May 25th 2018

Efficient Sub-Pixel Convolutional Neural Network
Ref https://arxiv.org/abs/1609.05158
"""
from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import to_list
import tensorflow as tf


class ESPCN(SuperResolution):
    """Efficient Sub-Pixel Convolutional Neural Network.

    Args:
        layers: layer number of the network
        filters: a tuple of integer, representing each layer's filters
        kernel: a tuple of integer, representing each layer's kernel size
    """

    def __init__(self, layers=3, filters=(64, 32, 32), kernel=(5, 3, 3),
                 name='espcn', **kwargs):
        super(ESPCN, self).__init__(**kwargs)
        self.name = name
        self.layers = layers
        self.filters = to_list(filters, layers)
        self.kernel_size = to_list(kernel, layers)
        if len(self.kernel_size) < self.layers:
            self.kernel_size += to_list(
                kernel[-1], self.layers - len(self.kernel_size))

    def build_graph(self):
        super(ESPCN, self).build_graph()
        with tf.variable_scope(self.name):
            x = self.inputs_preproc[-1] / 127.5 - 1
            for f, k in zip(self.filters, self.kernel_size):
                x = self.tanh_conv2d(x, f, k)
            x = self.upscale(x, 'espcn', direct_output=True)
            self.outputs.append((x + 1) * 127.5)

    def build_loss(self):
        with tf.name_scope('loss'):
            mse, loss = super(ESPCN, self).build_loss()
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(
                tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(
                tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))
