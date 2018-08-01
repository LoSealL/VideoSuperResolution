"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 8th 2018
Updated Date: June 8th 2018

Image Super-Resolution via Deep Recursive Residual Network
See http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import *

import tensorflow as tf


class DRRN(SuperResolution):

    def __init__(self, residual_unit=3, recursive_block=3, grad_clip=0.01, name='drrn', **kwargs):
        self.ru = residual_unit
        self.rb = recursive_block
        self.grad_clip = grad_clip
        self.name = name
        super(DRRN, self).__init__(**kwargs)

    def summary(self):
        super(DRRN, self).summary()
        print('Recursive Blocks: %d' % self.rb)
        print('Residual Units: %d' % self.ru)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(DRRN, self).build_graph()
            bic = bicubic_rescale(self.inputs_preproc[-1], self.scale)
            x = bic
            for i in range(self.rb):
                x = self._make_resblock(i, x)
            x = self.conv2d(x, self.channel, 3, kernel_initializer='he_normal', kernel_regularizer='l2')
            self.outputs.append(x + bic)

    def build_loss(self):
        with tf.name_scope('loss'):
            y_true = self.label[-1]
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            reg = tf.add_n(tf.losses.get_regularization_losses())
            loss = mse + reg
            opt = tf.train.AdamOptimizer(self.learning_rate)
            if self.grad_clip > 0:
                grads_and_vars = []
                for grad, var in opt.compute_gradients(loss):
                    grads_and_vars.append((
                        tf.clip_by_value(grad, -self.grad_clip / self.learning_rate,
                                         self.grad_clip / self.learning_rate),
                        var))
                op = opt.apply_gradients(grads_and_vars, self.global_steps)
            else:
                op = opt.minimize(loss, self.global_steps)
            self.loss.append(op)

            self.train_metric['loss'] = loss
            self.metrics['mse'] = mse
            self.metrics['regularization'] = reg
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(y_true, y_pred, 255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255))

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])

    def _make_resblock(self, index, inputs):
        H = [self.conv2d(inputs, 128, 3, activation='relu',
                         kernel_initializer='he_normal', kernel_regularizer='l2')]
        for _ in range(self.ru):
            res = self.conv2d(H[-1], 128, 3, activation='relu', kernel_initializer='he_normal',
                              kernel_regularizer='l2', name='red', reuse=tf.AUTO_REUSE)
            res = self.conv2d(res, 128, 3, activation='relu', kernel_initializer='he_normal',
                              kernel_regularizer='l2', name='green', reuse=tf.AUTO_REUSE)
            H.append(H[0] + res)
        return H[-1]
