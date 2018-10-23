"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 23rd 2018
Updated Date: May 23rd 2018

Implementing Feed-forward Denoising Convolutional Neural Network
See http://ieeexplore.ieee.org/document/7839189/
"""
from ..Framework.SuperResolution import SuperResolution

import tensorflow as tf


class DnCNN(SuperResolution):
    """Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising

    Args:
        layers: number of layers used
    """

    def __init__(self, layers=20, name='dncnn', **kwargs):
        self.name = name
        self.layers = layers
        if 'scale' in kwargs: kwargs.pop('scale')
        super(DnCNN, self).__init__(scale=1, **kwargs)

    def build_graph(self):
        super(DnCNN, self).build_graph()  # build inputs placeholder
        with tf.variable_scope(self.name):
            # build layers
            x = self.inputs_preproc[-1] / 255  # use channel Y only
            x = self.relu_conv2d(x, 64, 3)
            for i in range(1, self.layers - 1):
                x = self.bn_relu_conv2d(x, 64, 3, use_bias=False)
            # the last layer w/o BN and ReLU
            x = self.conv2d(x, 1, 3)
            # residual training
            outputs = self.inputs_preproc[-1] / 255 - x
            self.outputs.append(outputs * 255)

    def build_loss(self):
        with tf.name_scope('loss'):
            mse, loss = super(DnCNN, self).build_loss()
            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])

    def export_model_pb(self, export_dir='.', export_name='model.pb', **kwargs):
        y_pred = self.outputs[-1]
        if self.rgba:
            y_pred = tf.concat([y_pred / 255, self.inputs_preproc[-2]], axis=-1)
            y_pred = tf.image.yuv_to_rgb(y_pred) * 255
        else:
            y_pred = tf.image.grayscale_to_rgb(y_pred)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 0, 255), tf.uint8)
        y_pred = tf.concat([y_pred, tf.zeros_like(y_pred)[..., 0:1]], axis=-1, name='output/hr/rgba')
        self.outputs[-1] = y_pred
        # tf.get_default_graph().prevent_feeding(self.training_phase)
        super(DnCNN, self).export_model_pb(export_dir, f'{self.name}.pb', **kwargs)
