"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 20th 2018

Single Image Super-Resolution with Feature Discrimination (ECCV 2018)
See http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Framework.GAN import loss_bce_gan
from VSR.Arch import Discriminator
from VSR.Util.Utility import Vgg, prelu

import tensorflow as tf


def _normalize(x):
    return x / 127.5 - 1


def _denormalize(x):
    return (x + 1) * 127.5


def _clip(image):
    return tf.cast(tf.clip_by_value(image, 0, 255), 'uint8')


class SRFEAT(SuperResolution):
    """SRFeat

    Args:
        glayers: number of layers in generator.
        dlayers: number of layers in discriminator.
        vgg_layer: vgg feature layer name for perceptual loss.
        init_epoch: number of initializing epochs.
    """

    def __init__(self, glayers=16, dlayers=4, vgg_layer='block5_conv4',
                 init_epoch=100, gan_weight=1e-3, vgg_weight=0.1569,
                 name='srfeat', **kwargs):
        super(SRFEAT, self).__init__(**kwargs)
        self.name = name
        self.g_layers = glayers
        self.init_epoch = init_epoch
        self.gan_weight = gan_weight
        self.vgg_weight = vgg_weight
        self.vgg_layer = vgg_layer
        self.vgg = Vgg(False, Vgg.VGG19)
        self.F = 64
        self.D = Discriminator.dcgan_d(self, [None, None, self.channel], 64,
                                       times_stride=dlayers, norm='bn',
                                       name_or_scope='Critic')
        self.DF = Discriminator.dcgan_d(self, [None, None, self.channel], 64,
                                        times_stride=dlayers, norm='bn',
                                        name_or_scope='DF')

    def build_graph(self):
        super(SRFEAT, self).build_graph()
        inputs_norm = _normalize(self.inputs_preproc[-1])
        label_norm = _normalize(self.label[-1])
        with tf.variable_scope(self.name):
            shallow_feature = self.prelu_conv2d(inputs_norm, self.F, 9)
            x = [shallow_feature]
            for _ in range(self.g_layers):
                x.append(self.resblock(x[-1], self.F, 3, activation='prelu',
                                       use_batchnorm=True))
            bottleneck = x[-1]
            for t in x[1:-1]:
                bottleneck += self.conv2d(t, self.F, 1)
            sr = self.upscale(bottleneck, direct_output=False, activator=prelu)
            sr = self.tanh_conv2d(sr, self.channel, 9)
            self.outputs.append(_denormalize(sr))

        disc_real = self.D(label_norm)
        disc_fake = self.D(sr)
        vgg_features = [self.vgg(self.outputs[0], self.vgg_layer)]
        vgg_features += [self.vgg(self.label[0], self.vgg_layer)]
        vgg_fake = self.DF(vgg_features[0])
        vgg_real = self.DF(vgg_features[1])

        with tf.name_scope('Loss'):
            loss_gen, loss_disc = loss_bce_gan(disc_real, disc_fake)
            vgg_loss_g, vgg_loss_d = loss_bce_gan(vgg_real, vgg_fake)
            mse = tf.losses.mean_squared_error(label_norm, sr)
            loss_d = loss_disc + vgg_loss_d
            loss_g = loss_gen + vgg_loss_g
            loss_vgg = tf.losses.mean_squared_error(*vgg_features)
            loss = tf.stack([loss_g, loss_vgg])
            loss = tf.reduce_sum(loss * [self.gan_weight, self.vgg_weight])

            var_g = tf.trainable_variables(self.name)
            var_d = tf.trainable_variables('Critic')
            var_df = tf.trainable_variables('DF')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt_i = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    mse, self.global_steps, var_list=var_g)
                opt_g = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    loss, self.global_steps, var_list=var_g)
                opt_d = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    loss_d, var_list=var_d + var_df)
                self.loss = [opt_i, opt_d, opt_g]

        self.train_metric['g_loss'] = loss_g
        self.train_metric['d_loss'] = loss_d
        self.train_metric['vgg_loss'] = loss_vgg
        self.train_metric['loss'] = loss
        self.metrics['psnr'] = tf.reduce_mean(
            tf.image.psnr(self.label[-1], self.outputs[-1], 255))
        self.metrics['ssim'] = tf.reduce_mean(
            tf.image.ssim(self.label[-1], self.outputs[-1], 255))

    def build_loss(self):
        pass

    def build_summary(self):
        super(SRFEAT, self).build_summary()
        tf.summary.image('SR', _clip(self.outputs[-1]))

    def build_saver(self):
        var_d = tf.global_variables('Critic')
        var_df = tf.global_variables('DF')
        var_g = tf.global_variables(self.name)
        loss = tf.global_variables('Loss')
        steps = [self.global_steps]
        self.savers.update({
            'Critic': tf.train.Saver(var_d + var_df, max_to_keep=1),
            'Gen': tf.train.Saver(var_g, max_to_keep=1),
            'Misc': tf.train.Saver(loss + steps, max_to_keep=1),
        })

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        epoch = kwargs.get('epochs')
        if epoch <= self.init_epoch:
            loss = self.loss[0]
        else:
            loss = self.loss[1:]
        return super(SRFEAT, self).train_batch(feature, label, learning_rate,
                                               loss=loss)
