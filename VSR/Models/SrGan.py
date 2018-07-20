"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 17th 2018
Updated Date: May 25th 2018

SRGAN implementation
See https://arxiv.org/abs/1609.04802
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util.Utility import *

import tensorflow as tf


class SRGAN(SuperResolution):

    def __init__(self, glayers, dlayers, vgg_layer, init_steps=10000,
                 mse_weight=1, gan_weight=1e-3, vgg_weight=2e-6,
                 name='srgan', **kwargs):
        self.g_layers = glayers
        self.d_layers = dlayers
        self.vgg_layer = to_list(vgg_layer, 2)
        self.init_steps = init_steps
        self.mse_weight = mse_weight
        self.gan_weight = gan_weight
        self.vgg_weight = vgg_weight
        self.name = name
        super(SRGAN, self).__init__(**kwargs)

    def compile(self):
        self.vgg = Vgg(input_shape=[None, None, 3], type='vgg19')
        return super(SRGAN, self).compile()

    def summary(self):
        super(SRGAN, self).summary()
        if self.global_steps.eval() <= self.init_steps:
            print('Initializing model using mse loss...')
        else:
            print('Training model using GAN loss...')

    def build_graph(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            super(SRGAN, self).build_graph()
            x = self._build_generative(self.inputs_preproc[-1])
            self.outputs.append(x)

    def build_loss(self):
        y_true = self.label[-1]
        y_pred = self.outputs[-1]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            _, y_fake = self._build_adversial(y_pred)
            _, y_real = self._build_adversial(y_true)
        with tf.name_scope('loss'):
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            gan_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake)
            vgg_true = self.vgg(y_true, self.vgg_layer[0], self.vgg_layer[1])
            vgg_pred = self.vgg(y_pred, self.vgg_layer[0], self.vgg_layer[1])
            vgg_loss = tf.losses.mean_squared_error(vgg_true, vgg_pred)
            generative_loss = [mse * self.mse_weight, gan_loss * self.gan_weight, vgg_loss * self.vgg_weight] + \
                              tf.losses.get_regularization_losses(scope=f'{self.name}/Generative')
            generative_loss = tf.add_n(generative_loss)
            discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real) + \
                                 tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake)
            discriminator_loss = tf.add_n([discriminator_loss] + \
                                          tf.losses.get_regularization_losses(f'{self.name}/Adversarial'))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt_g = tf.train.AdamOptimizer(self.learning_rate)
                opt_d = tf.train.AdamOptimizer(self.learning_rate)
                self.loss.append(opt_g.minimize(mse, self.global_steps,
                                                var_list=tf.trainable_variables(f'{self.name}/Generative')))
                self.loss.append(opt_g.minimize(generative_loss, self.global_steps,
                                                var_list=tf.trainable_variables(f'{self.name}/Generative')))
                self.loss.append(opt_d.minimize(discriminator_loss, self.global_steps,
                                                var_list=tf.trainable_variables(f'{self.name}/Adversarial')))

            self.train_metric['g_loss'] = generative_loss
            self.train_metric['d_loss'] = discriminator_loss
            self.train_metric['init_loss'] = mse
            self.metrics['mse'] = mse
            self.metrics['gan_loss'] = gan_loss
            self.metrics['vgg_loss'] = vgg_loss
            self.metrics['d_loss'] = discriminator_loss
            self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(y_true, self.outputs[0], 255))
            self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(y_true, self.outputs[0], 255))

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/gan', self.metrics['gan_loss'])
        tf.summary.scalar('loss/vgg', self.metrics['vgg_loss'])
        tf.summary.scalar('loss/dis', self.metrics['d_loss'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        feature = to_list(feature)
        label = to_list(label)
        feed_dict = {self.training_phase: True, self.learning_rate: learning_rate}
        for i in range(len(self.inputs)):
            feed_dict[self.inputs[i]] = feature[i]
        for i in range(len(self.label)):
            feed_dict[self.label[i]] = label[i]
        if self.global_steps.eval() <= self.init_steps:
            loss_op = self.loss[:1]
        else:
            loss_op = self.loss[1:]
        loss = tf.get_default_session().run(list(self.train_metric.values()) + loss_op, feed_dict=feed_dict)
        ret = {}
        for k, v in zip(self.train_metric, loss):
            ret[k] = v
        return ret

    def _build_generative(self, inputs):
        with tf.variable_scope('Generative'):
            shallow_feature = self.conv2d(inputs, 64, 3, activation='relu', kernel_initializer='he_normal')
            x = shallow_feature
            for _ in range(self.g_layers):
                x_old = x
                x = self.conv2d(x, 64, 3, activation='relu', use_batchnorm=True, kernel_initializer='he_normal')
                x = self.conv2d(x, 64, 3, use_batchnorm=True, kernel_initializer='he_normal')
                x += x_old
            x = self.conv2d(x, 64, 3, use_batchnorm=True, kernel_initializer='he_normal')
            x += shallow_feature
            x = self.conv2d(x, 256, 3, kernel_initializer='he_normal')
            if self.scale[0] == 4 and self.scale[1] == 4:
                # the paper fixed scale factor as 4
                x = self.conv2d(x, 4, 3, activation='relu', kernel_initializer='he_normal')
                x = pixel_shift(x, 2, 1)
                x = self.conv2d(x, 256, 3, kernel_initializer='he_normal')
                x = self.conv2d(x, 4, 3, activation='relu', kernel_initializer='he_normal')
                x = pixel_shift(x, 2, 1)
            else:
                x = self.conv2d(x, self.scale[0] * self.scale[1], 3, activation='relu',
                                kernel_initializer='he_normal')
                x = pixel_shift(x, self.scale, 1)
            x = self.conv2d(x, 1, 1, kernel_initializer='he_normal')
            return x

    def _build_adversial(self, inputs):
        with tf.variable_scope('Adversarial'):
            x = self.conv2d(inputs, 64, 3, activation=tf.nn.leaky_relu, kernel_initializer='he_normal')
            filter = 64
            assert self.d_layers % 2 == 0
            strides = [1, 2] * (self.d_layers // 2)
            for i in range(1, self.d_layers):
                filter *= strides[i - 1]
                x = self.conv2d(x, filter, 3, strides=strides[i], activation=tf.nn.leaky_relu, use_batchnorm=True,
                                kernel_initializer='he_normal')
            w_init = tf.initializers.random_normal(stddev=0.02)
            x = tf.layers.dense(x, 1024, tf.nn.leaky_relu, kernel_initializer=w_init)
            x = tf.layers.dense(x, 1, kernel_initializer=w_init)
            return tf.nn.sigmoid(x), x
