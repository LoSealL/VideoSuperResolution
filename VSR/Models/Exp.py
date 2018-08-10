"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: July 31st 2018

== DO NOT USE ==
Experimental Models
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Framework import GAN
from VSR.Util.Utility import *
from VSR.Util.ImageProcess import imread, random_crop_batch_image

import tensorflow as tf
import numpy as np
from functools import partial

EXP_MODEL_NAME = 'minigan'


class EXP(SuperResolution):

    def __init__(self, fixed_train_hr_size=None, name=EXP_MODEL_NAME, **kwargs):
        super(EXP, self).__init__(**kwargs)
        self.name = name
        self.F = 64  # filters
        self.K = 3  # kernel size
        # discriminator net
        self.Disc = GAN.Discriminator(
            self,
            input_shape=[-1, fixed_train_hr_size, fixed_train_hr_size, self.channel])

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(EXP, self).build_graph()
            # normalized to [-1, 1]
            x = self.inputs_preproc[-1] / 127.5 - 1
            N = [self.conv2d(x, self.F, self.K, activation='relu',
                             kernel_initializer='he_normal')]
            for _ in range(1, 7):
                N.append(self.conv2d(N[-1], self.F, self.K, activation='relu',
                                     use_batchnorm=False,
                                     kernel_initializer='he_normal'))
            act = partial(self.conv2d, filters=self.F, kernel_size=self.K,
                          activation='relu', kernel_initializer='he_normal')
            hr = self.upscale(N[0] + N[-1], method='nearest',
                              activator=act)
            hr = self.conv2d(hr, self.channel, self.K, activation='tanh')
            self.outputs.append(hr)
            self.outputs.append((hr + 1) * 127.5)

    def build_loss(self):
        label_norm = self.label[-1] / 127.5 - 1
        y_fake = self.Disc(self.outputs[0])
        y_real = self.Disc(label_norm)
        with tf.variable_scope('Loss'):
            mae = tf.losses.absolute_difference(self.label[-1], self.outputs[-1])
            mse = tf.losses.mean_squared_error(self.label[-1], self.outputs[-1])

            g_loss, d_loss = GAN.loss_relative_lsgan(y_real, y_fake, average=True)
            t_loss = mse + g_loss

            var_g = tf.trainable_variables(self.name)
            var_d = tf.trainable_variables('Critic')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.loss.append(tf.train.AdamOptimizer(self.learning_rate).minimize(
                    mse, self.global_steps, var_list=var_g))
                self.loss.append(tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(
                    t_loss, self.global_steps, var_list=var_g))
                self.loss.append(tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9).minimize(
                    d_loss, var_list=var_d))

            self.train_metric['Loss'] = t_loss
            self.train_metric['MSE'] = mse
            self.train_metric['DLoss'] = d_loss
            self.metrics['PSNR'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], 255))
            self.metrics['SSIM'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], 255))

    def build_summary(self):
        tf.summary.scalar('Loss', self.train_metric['Loss'])
        tf.summary.scalar('MSE', self.train_metric['MSE'])
        tf.summary.scalar('PSNR', self.metrics['PSNR'])
        tf.summary.image('HR', self.outputs[-1])
        tf.summary.image('LB', self.label[-1])

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        epoch = kwargs.get('epochs')
        if epoch < 50:
            loss = self.loss[:1]
        else:
            loss = self.loss[1:]
        return super(EXP, self).train_batch(feature, label, learning_rate, loss=loss)


class EXP2(SuperResolution):

    def __init__(self, fixed_train_hr_size=None, condition_img=None, name='self_gan', **kwargs):
        super(EXP2, self).__init__(**kwargs)
        self.name = name
        self.cimg = imread(condition_img)
        self.F = 64  # filters
        self.K = 3  # kernel size
        # discriminator net
        self.Disc = GAN.Discriminator(
            self, depth=1,
            input_shape=[-1, fixed_train_hr_size, fixed_train_hr_size, self.channel])

    def _build_generative(self, inputs, layers=3):
        with tf.variable_scope('Generative'):
            shallow_feature = self.conv2d(
                inputs, self.F, self.K,
                activation='relu', kernel_initializer='he_normal')
            x = shallow_feature
            for _ in range(layers):
                x_old = x
                x = self.conv2d(x, self.F, self.K, activation='relu',
                                use_batchnorm=True, kernel_initializer='he_normal')
                x = self.conv2d(x, self.F, self.K,
                                use_batchnorm=True, kernel_initializer='he_normal')
                x += x_old
            x = self.conv2d(x, self.F, self.K,
                            use_batchnorm=True, kernel_initializer='he_normal')
            x += shallow_feature
            x = self.conv2d(x, self.channel, self.K, activation='tanh',
                            kernel_initializer='he_normal')
            return x

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(EXP2, self).build_graph()
            self.bimg = tf.placeholder(tf.float32, [None, None, None, self.channel], name='CondImg')
            # normalized to [-1, 1]
            x = self.bimg / 127.5 - 1 + self.inputs_preproc[-1]
            hr = self._build_generative(x)
            self.outputs.append(hr)
            self.outputs.append((hr + 1) * 127.5)

    def build_loss(self):
        label_norm = self.label[-1] / 127.5 - 1
        y_fake = self.Disc(self.outputs[0])
        y_real = self.Disc(label_norm)

        with tf.variable_scope('Loss'):
            g_loss, d_loss = GAN.loss_bce_gan(y_real, y_fake)
            cc = color_consistency(self.outputs[0], label_norm)
            t_loss = g_loss + 50 * cc
            var_g = tf.trainable_variables(self.name)
            var_d = tf.trainable_variables('Critic')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.loss.append(tf.train.AdamOptimizer(1e-4).minimize(
                    t_loss, self.global_steps, var_list=var_g))
                self.loss.append(tf.train.AdamOptimizer(1e-5, 0.5, 0.9).minimize(
                    d_loss, var_list=var_d))

            self.train_metric['TLoss'] = t_loss
            self.train_metric['DLoss'] = d_loss
            self.train_metric['Cc'] = cc

    def build_summary(self):
        tf.summary.image('HR', self.outputs[-1])
        tf.summary.image('IN', self.bimg)

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        img = random_crop_batch_image(self.cimg, feature.shape[0], feature.shape[1:])
        self.feed_dict.update({self.bimg: img})
        return super(EXP2, self).train_batch(feature, label, learning_rate)

    def validate_batch(self, feature, label, **kwargs):
        img = random_crop_batch_image(self.cimg, feature.shape[0], feature.shape[1:])
        self.feed_dict.update({self.bimg: img})
        return super(EXP2, self).validate_batch(feature, label, **kwargs)

    def test_batch(self, inputs, label=None, **kwargs):
        img = random_crop_batch_image(self.cimg, inputs.shape[0], inputs.shape[1:])
        self.feed_dict.update({self.bimg: img})
        super(EXP2, self).test_batch(inputs, label, **kwargs)
