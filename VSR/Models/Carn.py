"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 11th 2018

Cascading Residual Network (ECCV 2018)
See https://arxiv.org/abs/1803.08664
"""

from ..Framework.SuperResolution import SuperResolution

import tensorflow as tf


class CARN(SuperResolution):
    """Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network

    Args:
        recursive: A boolean, specifies whether use shared Residual-E weights
        groups: number of groups in group-convolution
        n_residual: number of Residual-E layers in a cascading block
        n_block: number of cascading blocks
    """

    def __init__(self, name='carn',
                 recursive=False,
                 groups=1,
                 n_residual=3,
                 n_blocks=3,
                 filters=64,
                 **kwargs):
        super(CARN, self).__init__(**kwargs)
        self.name = name
        self.recur = recursive
        self.groups = groups
        self.n_residual = n_residual
        self.n_blocks = n_blocks
        self.F = filters

    def _residual_e(self, inputs, reuse=False, **kwargs):
        if reuse: reuse = tf.AUTO_REUSE
        with tf.variable_scope(kwargs.get('name'), 'Residual-E', reuse=reuse):
            """Fake code since there is no efficient group conv2d in TF!
            see https://github.com/tensorflow/tensorflow/pull/10482
            and https://github.com/tensorflow/tensorflow/issues/3332
            >>> x = group_conv2d(inputs, self.F, 3, self.groups, activation='relu', kernel_initializer='he_uniform')
            >>> x = group_conv2d(x, self.F, 3, self.groups, activation='relu', kernel_initializer='he_uniform')
            >>> x = conv2d(x, inputs.shape[-1], 1, kernel_initializer='he_uniform')
            """
            # use a normal residual instead
            x = self.conv2d(inputs, self.F, 3, activation='relu', kernel_initializer='he_uniform')
            x = self.conv2d(x, inputs.shape[-1], 3, kernel_initializer='he_uniform')
            inputs += x
            return tf.nn.relu(inputs)

    def _cascading(self, inputs, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'CascadingBlock'):
            feat = [inputs]
            outp_1x1 = inputs
            name = 'SharedResE' if self.recur else None
            F = inputs.shape[-1]
            for i in range(self.n_residual):
                x = self._residual_e(outp_1x1, reuse=self.recur, name=name)
                feat.append(x)
                x = tf.concat(feat, axis=-1)
                outp_1x1 = self.conv2d(x, F, 1, kernel_initializer='he_uniform')
            return outp_1x1

    def build_graph(self):
        super(CARN, self).build_graph()
        with tf.variable_scope(self.name):
            x = self.inputs_preproc[-1] / 255
            outp_1x1 = self.conv2d(x, self.F, 3, kernel_initializer='he_uniform')
            feat = [x]
            F = outp_1x1.shape[-1]
            for i in range(self.n_blocks):
                x = self._cascading(outp_1x1)
                feat.append(x)
                x = tf.concat(feat, axis=-1)
                outp_1x1 = self.conv2d(x, F, 1, kernel_initializer='he_uniform')
            sr = self.upscale(outp_1x1, direct_output=False)
            sr = self.conv2d(sr, self.channel, 3)
            self.outputs.append(sr * 255)

    def build_loss(self):
        with tf.name_scope('Loss'):
            l1_loss = tf.losses.absolute_difference(self.label[0], self.outputs[0])
            re_loss = tf.losses.get_regularization_losses()
            mse = tf.losses.mean_squared_error(self.label[0], self.outputs[0])
            loss = tf.add_n(re_loss + [l1_loss], name='Loss')

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(opt)

        # tensorboard
        self.train_metric['loss'] = loss
        self.train_metric['l1'] = l1_loss
        self.metrics['mse'] = mse
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[0], self.outputs[0], 255))
        self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[0], self.outputs[0], 255))

    def build_summary(self):
        tf.summary.scalar('loss', self.train_metric['loss'])
        tf.summary.scalar('mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
        tf.summary.image('SR', self.outputs[0], 1)

    def build_saver(self):
        self.savers[self.name] = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
