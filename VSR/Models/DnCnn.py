"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 23rd 2018
Updated Date: May 23rd 2018

Implementing Feed-forward Denoising Convolutional Neural Network
See http://ieeexplore.ieee.org/document/7839189/
**Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising**
"""
from VSR.Framework.SuperResolution import SuperResolution

import tensorflow as tf


class DnCNN(SuperResolution):

    def __init__(self, layers=20, name='dncnn', **kwargs):
        self.name = name
        self.layers = layers
        if 'scale' in kwargs:
            kwargs.pop('scale')
        super(DnCNN, self).__init__(scale=1, **kwargs)

    def build_graph(self):
        with tf.name_scope(self.name):
            super(DnCNN, self).build_graph()  # build inputs placeholder
            # build layers
            x = self.inputs_preproc[-1]  # use channel Y only
            x = self.conv2d(x, 64, 3, activation='relu', kernel_initializer='he_normal', kernel_regularizer='l2')
            for i in range(1, self.layers - 1):
                x = self.conv2d(x, 64, 3, activation='relu', use_batchnorm=True, use_bias=False,
                                kernel_initializer='he_normal', kernel_regularizer='l2')
            # the last layer w/o BN and ReLU
            x = self.conv2d(x, 1, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            # residual training
            outputs = self.inputs_preproc[-1] - x
            self.outputs.append(outputs)

    def build_loss(self):
        with tf.name_scope('loss'):
            self.label.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1]))
            y_true = tf.cast(self.label[-1], tf.float32)
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            regular_loss = tf.add_n(tf.losses.get_regularization_losses())
            loss = mse + regular_loss
            # update BN
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.loss.append(optimizer.minimize(loss, self.global_steps))
            self.metrics['mse'] = mse
            self.metrics['regularization'] = regular_loss
            self.metrics['psnr'] = tf.image.psnr(y_true, y_pred, max_val=255)
            self.metrics['ssim'] = tf.image.ssim(y_true, y_pred, max_val=255)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', tf.reduce_mean(self.metrics['psnr']))
        tf.summary.scalar('ssim', tf.reduce_mean(self.metrics['ssim']))

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
