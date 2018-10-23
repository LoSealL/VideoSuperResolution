"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 11th 2018

MemNet (ICCV 2017)
See https://arxiv.org/abs/1708.02209
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import to_list, bicubic_rescale

import tensorflow as tf


class MEMNET(SuperResolution):
    """MemNet: A Persistent Memory Network for Image Restoration

    Args:
        n_memblock: number of memory blocks
        n_recur: number of recursive unit in each memory block
    """

    def __init__(self, name='memnet', n_memblock=6, n_recur=6,
                 filters=64, **kwargs):
        super(MEMNET, self).__init__(**kwargs)
        self.name = name
        self.recur = n_recur
        self.n_memblock = n_memblock
        self.F = filters

    def _recursive_unit(self, inputs, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'RecursiveUnit'):
            R = [inputs]
            for _ in range(self.recur):
                R.append(self.resblock(R[-1], self.F, 3, activation='relu'))
            b_short = tf.concat(R[1:], axis=-1, name='short_memory')
            return b_short

    def _gate_unit(self, short, long, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'GateUnit'):
            long = to_list(long)
            b_gate = tf.concat(long + [short], axis=-1)
            return self.relu_conv2d(b_gate, self.F, 1)

    def _memory_block(self, short, long, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'MemoryBlock'):
            long = to_list(long)
            short = self._recursive_unit(short)
            return self._gate_unit(short, long)

    def _reconstruct(self, x, inputs, **kwargs):
        with tf.variable_scope('Reconstruct', reuse=tf.AUTO_REUSE):
            inputs = self.relu_conv2d(inputs, self.F, 3)
            sr = self.conv2d(inputs, self.channel, 3)
            return sr + x

    def build_graph(self):
        super(MEMNET, self).build_graph()
        with tf.variable_scope(self.name):
            input_norm = self.inputs_preproc[-1] / 255
            input_norm = bicubic_rescale(input_norm, self.scale)
            sf = self.conv2d(input_norm, self.F, 3)
            feat = [sf]
            for i in range(self.n_memblock):
                feat.append(self._memory_block(feat[-1], feat))
            for fm in feat[1:]:
                srm = self._reconstruct(input_norm, fm)
                self.outputs.append(srm * 255)
            weights = tf.Variable([1.0 / self.n_memblock] * self.n_memblock, dtype=tf.float32)
            final = tf.add_n([self.outputs[i] * weights[i] for i in range(self.n_memblock)])
            self.outputs.append(final)

    def build_loss(self):
        with tf.name_scope('Loss'):
            alpha = 1 / (1 + self.n_memblock)
            mse_losses = [tf.losses.mean_squared_error(self.label[-1], self.outputs[-1], weights=alpha / 2)]
            mse_losses += [tf.losses.mean_squared_error(self.label[0], o, weights=(1 - alpha) / 2) for o in
                           self.outputs[:-1]]
            re_loss = tf.losses.get_regularization_losses()
            loss = tf.add_n(re_loss + mse_losses, name='Loss')

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(opt)

        # tensorboard
        self.train_metric['loss'] = loss
        self.metrics['mse'] = mse_losses[0] * 2 / alpha
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], 255))
        self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], 255))

    def build_summary(self):
        tf.summary.scalar('loss', self.train_metric['loss'])
        tf.summary.scalar('mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.scalar('ssim', self.metrics['ssim'])
        tf.summary.image('SR', self.outputs[-1], 1)
