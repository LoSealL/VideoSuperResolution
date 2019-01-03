"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 19th 2018

For ICLR 2019 Reproducibility Challenge
"""

import tensorflow as tf
import numpy as np
import tqdm
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Framework.Trainer import VSR
from VSR.Framework.Callbacks import save_batch_image
from VSR.Framework.GAN import (
    loss_bce_gan, loss_wgan, gradient_penalty, loss_lsgan, loss_relative_lsgan,
    loss_relative_bce_gan, inception_score
)
from VSR.Arch import Discriminator
from VSR.Util.Config import Config
from VSR.Util.Utility import to_list


class GAN(SuperResolution):
    """Base class of GAN.
    Args:
        name: model name.
        patch_size: generated image size.
        z_dim: latent space dimension.
        init_filter: filter size. (ideally, 512 for 32x32, 1024 for 64x64).
        linear: boolean, toggle FC layer after random vector.
        norm_g: normalization of G.
        norm_d: normalization of D.
        use_bias: boolean, use bias variables.
        optimizer: str: 'adam', 'rmsprop', 'momentum', 'sgd'.
        arch: G/D architecture: 'dcgan' or 'resnet'.
        nd_iter: number of D updates for each G update.
    """
    def __init__(self, name='gan', patch_size=32, z_dim=128, init_filter=512,
                 linear=False, norm_g=None, norm_d=None, use_bias=False,
                 optimizer=None, arch=None, nd_iter=1, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.name = name
        self._trainer = GanTrainer
        self.output_size = patch_size
        self.z_dim = z_dim
        self.init_filter = init_filter
        self.linear = linear
        self.bias = use_bias
        self.nd_iter = nd_iter
        if isinstance(norm_g, str):
            self.bn = np.any([word in norm_g for word in ('bn', 'batch')])
            self.sn = np.any([word in norm_g for word in ('sn', 'spectral')])
        self.d_outputs = []  # (real, fake)
        self.g_outputs = []  # (real, fake)
        self.opt = optimizer
        if self.opt is None:
            self.opt = Config(name='adam')
        if arch is None or arch == 'dcgan':
            self.G = self.dcgan_g
            self.D = Discriminator.dcgan_d(
                self, [patch_size, patch_size, self.channel],
                norm=norm_d, name_or_scope='D')
        elif arch == 'resnet':
            self.G = self.resnet_g
            self.D = Discriminator.resnet_d(
                self, [patch_size, patch_size, self.channel], times_pooling=4,
                norm=norm_d, name_or_scope='D')

    @staticmethod
    def _normalize(x):
        return x / 127.5 - 1

    @staticmethod
    def _denormalize(x):
        return (x + 1) * 127.5

    def dcgan_g(self, inputs):
        with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
            f = self.init_filter
            size = 4
            n_up = int(np.log2(self.output_size // size)) + 1
            kwargs = dict(use_sn=self.sn,
                          kernel_initializer='random_normal_0.02')
            x = self.dense(inputs, f * size * size, use_sn=self.sn,
                           kernel_initializer='random_normal_0.02')
            if self.bn:
                x = self.batch_norm(x, self.training_phase, epsilon=2e-5)
            x = tf.nn.relu(x)
            x = tf.reshape(x, [-1, size, size, f])
            for i in range(1, n_up):
                x = self.deconv2d(x, f // 2**i, 4, 2, **kwargs)
                if self.bn:
                    x = self.batch_norm(x, self.training_phase, epsilon=2e-5)
                x = tf.nn.relu(x)
            x = tf.nn.relu(x)
            x = self.deconv2d(x, self.channel, 3, 1, **kwargs)
            x = tf.nn.tanh(x)
        return x

    def resnet_g(self, inputs):
        with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
            f = self.init_filter // 2
            size = 4
            n_up = int(np.log2(self.output_size // size))
            x = self.dense(inputs, f * size * size, use_sn=self.sn,
                           kernel_initializer='random_normal_0.02')
            x = tf.reshape(x, [-1, size, size, f])
            for _ in range(n_up):
                # up
                x = self.upscale(x, 'nearest', 2)
                x = self.resblock(x, 256, 3, activation='relu',
                                  use_batchnorm=self.bn,
                                  use_sn=self.sn, use_bias=self.bias)
            x = self.batch_norm(x, self.training_phase)
            x = tf.nn.relu(x)
            x = self.tanh_conv2d(x, self.channel, 3)
            return x

    def build_graph(self):
        self.inputs.append(
            tf.placeholder('float32', (None, self.z_dim,), name='input/noise'))
        self.label.append(tf.placeholder(
            'float32', [None, self.output_size, self.output_size, self.channel],
            name='label/image'))
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            fake_image = self.G(self.inputs[0])
            real_image = self._normalize(self.label[0])

            real_disc = self.D(real_image)
            fake_disc = self.D(fake_image)
            self.outputs.append(self._denormalize(fake_image))
            self.d_outputs = (real_disc, fake_disc)
            self.g_outputs = (real_image, fake_image)

        self.p_fake = tf.reduce_mean(tf.sigmoid(fake_disc))
        self.p_real = tf.reduce_mean(tf.sigmoid(real_disc))

    def _build_loss(self, g_loss, d_loss):
        # used in sub-class
        var_d = tf.trainable_variables(self.name + '/D')
        var_g = tf.trainable_variables(self.name + '/G')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_d = self.get_optimizer(self.opt)
            opt_g = self.get_optimizer(self.opt)
            op_d = opt_d.minimize(d_loss, self.global_steps, var_list=var_d)
            op_g = opt_g.minimize(g_loss, var_list=var_g)
            self.loss = [op_g, op_d]

        self.train_metric = {'gloss': g_loss, 'd_loss': d_loss,
                             'p_real': self.p_real, 'p_fake': self.p_fake}
        self.metrics['inception-score'] = inception_score(self.outputs[0], 1)
        self.metrics['p_real'] = self.p_real
        self.metrics['p_fake'] = self.p_fake

    def build_summary(self):
        tf.summary.scalar('FakeD', self.p_fake)
        tf.summary.scalar('RealD', self.p_real)
        tf.summary.scalar('Inception_Score', self.metrics['inception-score'])

    def build_saver(self):
        var_g = tf.global_variables(self.name + '/G')
        var_d = tf.global_variables(self.name + '/D')
        var_loss = tf.global_variables('Loss')
        steps = [self.global_steps]
        self.savers['gen'] = tf.train.Saver(var_g + steps, max_to_keep=1)
        self.savers['disc'] = tf.train.Saver(var_d, max_to_keep=1)
        self.savers['loss'] = tf.train.Saver(var_loss, max_to_keep=1)

    def get_optimizer(self, config=None):
        if config is None:
            config = self.opt
        name = config.name
        if name == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate, **config)
        elif name == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learning_rate, **config)
        elif name == 'momentum':
            return tf.train.MomentumOptimizer(self.learning_rate, **config)
        elif name == 'sgd':
            return tf.train.GradientDescentOptimizer(
                self.learning_rate, **config)
        return None

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        feature = to_list(feature)
        label = to_list(label)
        self.feed_dict.update(
            {self.training_phase: True, self.learning_rate: learning_rate})
        for i in range(len(self.inputs)):
            self.feed_dict[self.inputs[i]] = feature[i]
        for i in range(len(self.label)):
            self.feed_dict[self.label[i]] = label[i]
        loss = kwargs.get('loss') or self.loss
        loss = to_list(loss)
        step = kwargs['steps']
        sess = tf.get_default_session()
        if step % self.nd_iter == 0:
            # update G-net
            sess.run(loss[0], feed_dict=self.feed_dict)
        # update D-net
        sess.run(loss[1:], feed_dict=self.feed_dict)
        loss = sess.run(list(self.train_metric.values()),
                        feed_dict=self.feed_dict)
        ret = {}
        for k, v in zip(self.train_metric, loss):
            ret[k] = v
        return ret


class SGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_bce_gan(*self.d_outputs)
            self._build_loss(g_loss, d_loss)


class SGANGP(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_bce_gan(*self.d_outputs)
            with tf.variable_scope(self.name, reuse=True):
                gp = gradient_penalty(*self.g_outputs, self.D, lamb=10)
            d_loss += gp
            self._build_loss(g_loss, d_loss)


class LSGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_lsgan(*self.d_outputs)
            self._build_loss(g_loss, d_loss)


class WGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_wgan(*self.d_outputs)
            self._build_loss(g_loss, d_loss)
            # weights clip
            var_d = tf.trainable_variables(self.name + '/D')
            clip_bounds = [-.01, .01]
            clip_ops = [tf.assign(var, tf.clip_by_value(var, *clip_bounds)) for
                        var in var_d]
            clip_disc_weights = tf.group(*clip_ops)
            self.loss.append(clip_disc_weights)

    def build_saver(self):
        var_g = tf.global_variables(self.name + '/G')
        var_d = tf.global_variables(self.name + '/D')
        steps = [self.global_steps]
        self.savers['gen'] = tf.train.Saver(var_g + steps, max_to_keep=1)
        self.savers['disc'] = tf.train.Saver(var_d, max_to_keep=1)


class WGANGP(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_wgan(*self.d_outputs)
            with tf.variable_scope(self.name, reuse=True):
                gp = gradient_penalty(*self.g_outputs, self.D, lamb=10)
            d_loss += gp
            self._build_loss(g_loss, d_loss)


class RGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_relative_bce_gan(*self.d_outputs,
                                                   average=False)
            self._build_loss(g_loss, d_loss)


class RGANGP(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_relative_bce_gan(*self.d_outputs,
                                                   average=False)
            with tf.variable_scope(self.name, reuse=True):
                gp = gradient_penalty(*self.g_outputs, self.D, lamb=10)
            d_loss += gp
            self._build_loss(g_loss, d_loss)


class RaGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_relative_bce_gan(*self.d_outputs,
                                                   average=True)
            self._build_loss(g_loss, d_loss)


class RaGANGP(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_relative_bce_gan(*self.d_outputs,
                                                   average=True)
            with tf.variable_scope(self.name, reuse=True):
                gp = gradient_penalty(*self.g_outputs, self.D, lamb=10)
            d_loss += gp
            self._build_loss(g_loss, d_loss)


class RLSGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_relative_lsgan(*self.d_outputs, average=False)
            self._build_loss(g_loss, d_loss)


class RaLSGAN(GAN):
    def build_loss(self):
        with tf.name_scope('Loss'):
            g_loss, d_loss = loss_relative_lsgan(*self.d_outputs, average=True)
            self._build_loss(g_loss, d_loss)


class GanTrainer(VSR):
    def query_config(self, config, **kwargs):
        # add [batch, patch_size] to collector `self.v`
        self.v.batch = config.batch
        self.v.patch_size = config.patch_size
        return super(GanTrainer, self).query_config(config, **kwargs)

    def fit_init(self):
        # disable data augmentation of GAN training
        self.v.train_loader.aug = False
        return super(GanTrainer, self).fit_init()

    def fn_train_each_step(self, label=None, feature=None, name=None):
        """override this method for:
          - sample feature from random noise (uniform distributed from [-1,1]).
          - pass step number to `train_batch` call.
        """
        v = self.v
        feature = np.random.uniform(-1, 1, [v.batch, self._m.z_dim])
        for fn in v.label_callbacks:
            label = fn(label, name=name)
        loss = self._m.train_batch(feature, label, learning_rate=v.lr,
                                   epochs=v.epoch, steps=v.global_step)
        v.global_step = self._m.global_steps.eval()
        # uncomment this if you want to record everything into tensorboard.
        # v.summary_writer.add_summary(self._m.summary(), v.global_step)
        for _k, _v in loss.items():
            v.avg_meas[_k] = \
                v.avg_meas[_k] + [_v] if v.avg_meas.get(_k) else [_v]
            loss[_k] = '{:08.5f}'.format(_v)
        v.loss = loss

    def fn_benchmark_body(self):
        """override this method for:
          - sample feature from random noise (uniform distributed from [-1,1]).
          - save an image grid during validation ([8] x [batch / 8])
        """
        v = self.v
        it = v.loader.make_one_shot_iterator(v.memory_limit, shuffle=True)
        if v.loader.method == 'val':
            v.output_callbacks += [save_batch_image(self._logd)]
        for label, _, name in tqdm.tqdm(it, 'Test', ascii=True):
            feature = np.random.uniform(-1, 1, [v.batch, self._m.z_dim])
            self.fn_benchmark_each_step(label, feature, name)
        if v.loader.method == 'val':
            v.output_callbacks.pop(-1)

    def infer(self, loader, config, **kwargs):
        """there's nothing to infer from GAN"""
        pass
