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


class EXP(SuperResolution):

    def __init__(self, fixed_train_hr_size=None, name='minigan', **kwargs):
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
            hr = self._build_generative(self.inputs_preproc[-1] + self.bimg / 127.5 - 1)
            # hr += self.bimg / 127.5 - 1
            self.outputs.append(hr)
            self.outputs.append((hr + 1) * 127.5)

    def build_loss(self):
        label_norm = self.label[-1] / 127.5 - 1
        y_fake = self.Disc(self.outputs[0])
        y_real = self.Disc(label_norm)

        with tf.variable_scope('Loss'):
            g_loss, d_loss = GAN.loss_bce_gan(y_real, y_fake)
            # cc = color_consistency(self.outputs[0], self.bimg / 127.5 - 1)
            cc = tf.losses.mean_squared_error(self.outputs[0], self.bimg / 127.5 - 1)
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


class Dense:
    NONE = 0
    NORMAL = 1
    LAST = 2


class EXP3(SuperResolution):

    def __init__(self, name='LapGAN', filters=64, layers=20, norm_mode=None, init_epoch=200, **kwargs):
        super(EXP3, self).__init__(**kwargs)
        self.name = name
        self.F = filters
        self.K = 3
        self.depth = layers // 4 - 1
        self.depth = max(self.depth, 4)
        self.bn = norm_mode == 'BN'
        self.sn = norm_mode == 'SN'
        self.init_epoch = init_epoch
        self.Dx2 = GAN.Discriminator(
            self, input_shape=[-1, None, None, self.channel],
            use_bn=self.bn, use_sn=self.sn, scope='Critic_x2')
        self.Dx4 = GAN.Discriminator(
            self, input_shape=[-1, None, None, self.channel],
            use_bn=self.bn, use_sn=self.sn, scope='Critic_x4')

    def summary(self):
        if self.bn:
            tf.logging.debug("BatchNorm enabled")
        if self.sn:
            tf.logging.debug("SpectralNorm enabled")
        super(EXP3, self).summary()

    def _block(self, name, inputs, f, k, depth, use_bn, use_sn, dense, **kwargs):
        assert not (use_sn and use_bn)  # BN conflict with SN

        def res(x, bn_first=False):
            # residual block
            if bn_first:
                x = tf.layers.batch_normalization(x, training=self.training_phase)
                x = tf.nn.relu(x)
            local = self.conv2d(x, f, k, use_batchnorm=use_bn, use_sn=use_sn, kernel_initializer='he_normal')
            x = self.conv2d(x, f, k, activation='relu', use_batchnorm=use_bn, use_sn=use_sn, kernel_initializer='he_normal')
            if bn_first:
                x = self.conv2d(x, f, k, kernel_initializer='he_normal')
            else:
                x = self.conv2d(x, f, k, use_batchnorm=use_bn, use_sn=use_sn, kernel_initializer='he_normal')
            return local + x

        with tf.variable_scope(name):
            lattern_features = [inputs]
            features = inputs.shape.as_list()[-1]
            x = inputs
            for _ in range(depth):
                x = res(x, bn_first=False)
                lattern_features.append(x)
                if dense == Dense.NORMAL:
                    x = tf.concat(lattern_features, axis=-1)
            if dense == Dense.LAST:
                x = tf.concat(lattern_features, axis=-1)
            x = self.conv2d(x, features, self.K, use_batchnorm=use_bn, use_sn=use_sn, kernel_initializer='he_normal')
            if dense == Dense.NONE:
                x += inputs
            x = self.upscale(x, scale=[2, 2], direct_output=False)
            return x

    def _down_sample_g(self, inputs):
        with tf.variable_scope('DownSample', reuse=tf.AUTO_REUSE):
            x = self.conv2d(inputs, self.F, self.K, activation='relu', kernel_initializer='he_normal')
            x = self.conv2d(x, self.F, self.K, activation='relu', kernel_initializer='he_normal')
            x = self.conv2d(x, self.F, self.K, strides=2, activation='relu', kernel_initializer='he_normal')
            x = self.conv2d(x, self.channel, self.K, activation='tanh', kernel_initializer='he_normal')
            return x

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(EXP3, self).build_graph()
            x = self.inputs_preproc[-1] / 255
            x = self.conv2d(x, self.F, self.K, activation='relu', kernel_initializer='he_normal')
            x2 = self._block('LapBlock1', x, self.F, self.K, 4, self.bn, self.sn, dense=Dense.NORMAL)
            x4 = self._block('LapBlock2', x2, self.F, self.K, 4, self.bn, self.sn, dense=Dense.NORMAL)
            # x2 = self.conv2d(x2, self.F, 3, kernel_initializer='he_normal')
            sr_x2 = self.conv2d(x2, self.channel, 3, activation='tanh', kernel_initializer='he_normal')
            # x4 = self.conv2d(x4, self.F, 3, kernel_initializer='he_normal')
            sr_x4 = self.conv2d(x4, self.channel, 3, activation='tanh', kernel_initializer='he_normal')
            self.outputs = [(sr_x2 + 1) * 127.5, (sr_x4 + 1) * 127.5]
            self.x2 = sr_x2
            self.x4 = sr_x4

    def build_loss(self):
        label_x2 = tf.nn.avg_pool(self.label[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        label_x4 = self.label[-1]
        label_x2_norm = label_x2 / 127.5 - 1
        label_x4_norm = label_x4 / 127.5 - 1
        fake_x2 = self.Dx2(self.x2)
        real_x2 = self.Dx2(label_x2_norm)
        fake_x4 = self.Dx4(self.x4)
        real_x4 = self.Dx4(label_x4_norm)

        with tf.name_scope('loss'):
            mae_x2 = tf.losses.absolute_difference(label_x2, self.outputs[0])
            mae_x4 = tf.losses.absolute_difference(label_x4, self.outputs[1])
            mse_x2 = tf.reduce_mean(tf.squared_difference(label_x2, self.outputs[0]))
            mse_x4 = tf.reduce_mean(tf.squared_difference(label_x4, self.outputs[1]))

            loss_g_x2, loss_d_x2 = GAN.loss_bce_gan(real_x2, fake_x2)
            loss_g_x4, loss_d_x4 = GAN.loss_bce_gan(real_x4, fake_x4)

            loss_x2 = mae_x2 + loss_g_x2
            loss_x4 = mae_x4 + loss_g_x4

            var_g = tf.trainable_variables(self.name)
            var_dx2 = tf.trainable_variables('Critic_x2')
            var_dx4 = tf.trainable_variables('Critic_x4')

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                init_loss_x2 = tf.train.AdamOptimizer(self.learning_rate).minimize(mae_x2, self.global_steps, var_list=var_g)
                init_loss_x4 = tf.train.AdamOptimizer(self.learning_rate).minimize(mae_x4, var_list=var_g)
                gan_loss_x2 = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9).minimize(
                    loss_x2, self.global_steps, var_list=var_g)
                gan_loss_x4 = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9).minimize(
                    loss_x4, var_list=var_g)
                d_loss_x2 = tf.train.AdamOptimizer(1e-5, 0.5, 0.9).minimize(
                    loss_d_x2, var_list=var_dx2)
                d_loss_x4 = tf.train.AdamOptimizer(1e-5, 0.5, 0.9).minimize(
                    loss_d_x4, var_list=var_dx4)
                self.loss = [init_loss_x2, init_loss_x4, gan_loss_x2, gan_loss_x4, d_loss_x2, d_loss_x4]

            self.train_metric['mae_x2'] = mae_x2
            self.train_metric['mae_x4'] = mae_x4
            self.metrics['mse_x2'] = mse_x2
            self.metrics['mse_x4'] = mse_x4
            self.metrics['psnr_x2'] = tf.reduce_mean(tf.image.psnr(label_x2, self.outputs[0], max_val=255))
            self.metrics['psnr_x4'] = tf.reduce_mean(tf.image.psnr(label_x4, self.outputs[1], max_val=255))

    def build_summary(self):
        tf.summary.scalar('SRF_2/mse', self.metrics['mse_x2'])
        tf.summary.scalar('SRF_2/psnr', self.metrics['psnr_x2'])
        tf.summary.scalar('SRF_4/mse', self.metrics['mse_x4'])
        tf.summary.scalar('SRF_4/psnr', self.metrics['psnr_x4'])
        tf.summary.image('SRF_2/output', self.outputs[0])
        tf.summary.image('SRF_4/output', self.outputs[1])

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        epoch = kwargs.get('epochs')
        if epoch < self.init_epoch:
            loss = self.loss[0]
        elif epoch < self.init_epoch * 2:
            loss = self.loss[:2]
        else:
            loss = self.loss[2:]
        return super(EXP3, self).train_batch(feature, label, learning_rate, loss=loss)
