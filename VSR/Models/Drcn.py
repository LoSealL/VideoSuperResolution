"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 5th 2018
Updated Date: June 5th 2018

Deeply-Recursive Convolutional Network for Image Super-Resolution
See https://arxiv.org/abs/1511.04491
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import bicubic_rescale
import tensorflow as tf
import numpy as np


class DRCN(SuperResolution):

    def __init__(self, recur=16, filters=256, name='drcn', **kwargs):
        self.recur = recur
        self.filters = filters
        self.name = name
        super(DRCN, self).__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(DRCN, self).build_graph()
            # bicubic upscale
            bic = bicubic_rescale(self.inputs_preproc[-1], self.scale)
            x = self._build_embedding(bic)
            y = [bic]
            for _ in range(self.recur):
                x = self._build_inference(x)
                y += [self._build_reconstruction(x)]
            self.outputs = y
            layer_weights = tf.Variable(np.ones_like(y, 'float') / len(y), name="LayerWeights", dtype=tf.float32)
            output = 0
            for i in range(len(y)):
                output += y[i] * layer_weights[i]
            self.outputs.insert(0, output / tf.reduce_sum(layer_weights))

    def build_loss(self):
        with tf.name_scope('loss'):
            y_true = self.label[-1]
            mse_n = []
            for y_pred in self.outputs[1:]:
                mse_n.append(tf.losses.mean_squared_error(y_true, y_pred))
            loss1 = tf.reduce_mean(mse_n)
            loss2 = tf.losses.mean_squared_error(y_true, self.outputs[0])
            regularization = tf.add_n(tf.losses.get_regularization_losses())
            alpha = tf.placeholder(tf.float32, name='alpha')
            loss = alpha * loss1 + (1.0 - alpha) * loss2 + regularization
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.loss.append(optimizer.minimize(loss, self.global_steps))

            self.train_metric['loss'] = loss
            self.metrics['local_mse'] = loss1
            self.metrics['final_mse'] = loss2
            self.metrics['regularization'] = regularization
            self.metrics['psnr'] = tf.image.psnr(y_true, self.outputs[-1], max_val=255)
            self.metrics['ssim'] = tf.image.ssim(y_true, self.outputs[-1], max_val=255)

    def build_summary(self):
        tf.summary.scalar('loss/training', self.train_metric['loss'])
        tf.summary.scalar('loss/local_mse', self.metrics['local_mse'])
        tf.summary.scalar('loss/final_mse', self.metrics['final_mse'])
        tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', tf.reduce_mean(self.metrics['psnr']))
        tf.summary.scalar('ssim', tf.reduce_mean(self.metrics['ssim']))

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        epoch = kwargs.get('epochs')
        if epoch < 50:
            self.feed_dict.update({'alpha:0': 1.0})
        elif epoch < 100:
            self.feed_dict.update({'alpha:0': 1 - (epoch - 50) / 50})
        else:
            self.feed_dict.update({'alpha:0': 0})
        super(DRCN, self).train_batch(feature, label, learning_rate, **kwargs)

    def _build_embedding(self, inputs):
        with tf.variable_scope('embedding'):
            x = self.conv2d(inputs, self.filters, 3, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer='l2')
            x = self.conv2d(x, self.filters, 3, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer='l2')
            return x

    def _build_inference(self, inputs):
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            x = self.conv2d(inputs, self.filters, 3, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer='l2')
            return x

    def _build_reconstruction(self, inputs):
        with tf.variable_scope('reconstruct', reuse=tf.AUTO_REUSE):
            x = self.conv2d(inputs, self.filters, 3, activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer='l2')
            x = self.conv2d(x, self.channel, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            return x
