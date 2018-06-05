"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 23rd 2018
Updated Date: May 23rd 2018

Implementing Fast and Accurate Image Super Resolution by
Deep CNN with Skip Connection and Network in Network
See https://arxiv.org/abs/1707.05425
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util.Utility import *

import tensorflow as tf


class DCSCN(SuperResolution):

    def __init__(self,
                 layers,
                 reconstruction_layers,
                 filters,
                 min_filters,
                 nin_filter,
                 reconst_filter,
                 filters_decay_gamma,
                 drop_out,
                 name='dcscn',
                 **kwargs):
        self.layers = layers
        self.reconstruction_layers = reconstruction_layers
        self.filters = filters
        self.min_filters = min_filters
        self.nin_filter = nin_filter
        self.reconst_filter = reconst_filter
        self.filters_decay_gamma = filters_decay_gamma
        self.drop_out = drop_out
        self.name = name
        super(DCSCN, self).__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(DCSCN, self).build_graph()
            shape_enlarged = tf.shape(self.inputs_preproc[-1])[1:3]
            shape_enlarged = shape_enlarged * self.scale
            bic = tf.image.resize_bicubic(self.inputs_preproc[-1], shape_enlarged)
            x = [self.inputs_preproc[-1]]
            drop_out = tf.cond(self.training_phase, lambda: self.drop_out, lambda: 1.0)
            for i in range(self.layers):
                if self.min_filters != 0 and i > 0:
                    x1 = i / float(self.layers - 1)
                    y1 = pow(x1, 1.0 / self.filters_decay_gamma)
                    output_feature_num = int((self.filters - self.min_filters) * (1 - y1) + self.min_filters)
                    nn = self.conv2d(x[-1], output_feature_num, 3, activation='relu', use_batchnorm=True,
                                     kernel_initializer='he_normal', kernel_regularizer='l2')
                    x.append(tf.nn.dropout(nn, drop_out))
            concat_x = tf.concat(x, axis=-1)
            with tf.variable_scope('NIN'):
                a1 = self.conv2d(concat_x, self.nin_filter[0], 1, activation='relu', use_batchnorm=True,
                                 kernel_initializer='he_normal', kernel_regularizer='l2')
                b1 = self.conv2d(concat_x, self.nin_filter[1], 1, activation='relu', use_batchnorm=True,
                                 kernel_initializer='he_normal', kernel_regularizer='l2')
                b2 = self.conv2d(b1, self.nin_filter[1], 3, activation='relu', use_batchnorm=True,
                                 kernel_initializer='he_normal', kernel_regularizer='l2')
            concat_nin = tf.concat([a1, b2], axis=-1)
            ps = self.conv2d(concat_nin, self.scale[0] * self.scale[1], 3, kernel_initializer='he_normal',
                             kernel_regularizer='l2')
            ps = pixel_shift(ps, self.scale, 1)
            with tf.variable_scope('Reconstruction'):
                for i in range(self.reconstruction_layers - 1):
                    ps = self.conv2d(ps, self.reconst_filter, 3, activation='relu', kernel_initializer='he_normal',
                                     kernel_regularizer='l2')
                    ps = tf.nn.dropout(ps, drop_out)
                outputs = self.conv2d(ps, 1, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            self.outputs.append(outputs + bic)

    def build_loss(self):
        with tf.variable_scope('loss'):
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
        tf.summary.scalar('mse', self.metrics['mse'])
        tf.summary.scalar('regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', tf.reduce_mean(self.metrics['psnr']))
        tf.summary.scalar('ssim', tf.reduce_mean(self.metrics['ssim']))
