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
from VSR.Util.Utility import *

import tensorflow as tf
import numpy as np


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
            l2_decay = 1e-5
            x = self.inputs_preproc[-1]  # use channel Y only
            x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
            for i in range(1, self.layers - 1):
                x = tf.layers.conv2d(x, 64, 3, 1, padding='same',
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
                x = tf.layers.batch_normalization(x, training=self.training_phase)
                x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, 1, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.he_normal())
            self.outputs.append(x)

    def build_loss(self):
        with tf.name_scope('loss'):
            self.label.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1]))
            y_true = tf.cast(self.label[-1], tf.float32)
            y_pred = self.inputs_preproc[-1] - self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            regular_loss = tf.add_n(tf.losses.get_regularization_losses())
            loss = mse + regular_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.loss.append(optimizer.minimize(loss))
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
        y_pred = self.inputs_preproc[-1] - self.outputs[-1]
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
