"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 17th 2018
Updated Date: May 25th 2018

SRGAN implementation (CVPR 2017)
See https://arxiv.org/abs/1609.04802
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Framework import GAN
from VSR.Util.Utility import *

import tensorflow as tf


class SRGAN(SuperResolution):
    """Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

    Args:
        glayers:
        dlayers:
        vgg_layer:
        init_epoch:
        mse_weight:
        gan_weight:
        vgg_weight:
        fixed_train_hr_size:
    """

    def __init__(self, glayers=16, dlayers=8, vgg_layer=(2, 2),
                 init_epoch=100, mse_weight=1, gan_weight=1e-3,
                 use_vgg=False, vgg_weight=2e-6,
                 fixed_train_hr_size=None, name='srgan', **kwargs):
        super(SRGAN, self).__init__(**kwargs)
        self.name = name
        self.g_layers = glayers
        self.d_layers = dlayers
        self.init_epoch = init_epoch
        self.mse_weight = mse_weight
        self.gan_weight = gan_weight
        self.vgg_weight = vgg_weight
        self.vgg_layer = to_list(vgg_layer, 2)
        self.use_vgg = use_vgg
        self.vgg = None
        self.D = GAN.Discriminator(self, input_shape=[-1, fixed_train_hr_size, fixed_train_hr_size, self.channel],
                                   depth=dlayers, use_bn=True, use_bias=True)

    def compile(self):
        if self.use_vgg:
            self.vgg = Vgg(input_shape=[None, None, 3], type='vgg19')
        return super(SRGAN, self).compile()

    @staticmethod
    def _normalize(x):
        return x / 255

    @staticmethod
    def _denormalize(x):
        return x * 255

    def build_graph(self):
        super(SRGAN, self).build_graph()
        with tf.variable_scope(self.name):
            inputs_norm = self._normalize(self.inputs_preproc[-1])
            shallow_feature = self.prelu_conv2d(inputs_norm, 64, 3)
            x = shallow_feature
            for _ in range(self.g_layers):
                x = self.resblock(x, 64, 3, activation='prelu', use_batchnorm=True)
            x = self.bn_conv2d(x, 64, 3)
            x += shallow_feature
            x = self.conv2d(x, 256, 3)
            sr = self.upscale(x, direct_output=False, activator=prelu)
            sr = self.conv2d(sr, self.channel, 9)
            self.outputs.append(self._denormalize(sr))

        label_norm = self._normalize(self.label[-1])
        disc_real = self.D(label_norm)
        disc_fake = self.D(sr)

        with tf.name_scope('Loss'):
            loss_gen, loss_disc = GAN.loss_bce_gan(disc_real, disc_fake)
            mse = tf.losses.mean_squared_error(self.label[-1], self.outputs[-1])
            reg = tf.losses.get_regularization_losses()

            loss = tf.add_n([mse * self.mse_weight, loss_gen * self.gan_weight] + reg)
            if self.use_vgg:
                vgg_real = self.vgg(self.label[-1], *self.vgg_layer)
                vgg_fake = self.vgg(self.outputs[-1], *self.vgg_layer)
                loss_vgg = tf.losses.mean_squared_error(vgg_real, vgg_fake)
                loss += self.vgg_weight + loss_vgg

            var_g = tf.trainable_variables(self.name)
            var_d = tf.trainable_variables('Critic')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt_g = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9).minimize(
                    loss, self.global_steps, var_list=var_g)
                opt_d = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9).minimize(
                    loss_disc, var_list=var_d)
                self.loss = [mse, opt_g, opt_d]

        self.train_metric['g_loss'] = loss_gen
        self.train_metric['d_loss'] = loss_disc
        self.train_metric['loss'] = loss
        self.metrics['mse'] = mse
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], 255))
        self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], 255))

    def build_loss(self):
        pass

    def build_summary(self):
        tf.summary.scalar('loss/gan', self.train_metric['g_loss'])
        tf.summary.scalar('loss/dis', self.train_metric['d_loss'])
        tf.summary.scalar('mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
        tf.summary.image('SR', self.outputs[-1], 1)

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        epoch = kwargs.get('epochs')
        if epoch < self.init_epoch:
            loss = self.loss[0]
        else:
            loss = self.loss[1:]
        return super(SRGAN, self).train_batch(feature, label, learning_rate, loss=loss)
