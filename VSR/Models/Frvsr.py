"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug 20th 2018

Frame-Recurrent Video Super-Resolution (CVPR 2018)
See https://arxiv.org/abs/1801.04590
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util import *

import tensorflow as tf


class FRVSR(SuperResolution):
    def __init__(self, name='frvsr', depth=10, **kwargs):
        super(FRVSR, self).__init__(**kwargs)
        self.name = name
        self.depth = depth

    def _Fnet(self, f0, f1):
        with tf.variable_scope('FNet', reuse=tf.AUTO_REUSE):
            x = tf.concat([f0, f1], axis=-1)
            F = [32, 64, 128, 256]
            for _f in F:
                x = self.conv2d(x, _f, 3, activation='lrelu', kernel_initializer='he_normal')
                x = self.conv2d(x, _f, 3, activation='lrelu', kernel_initializer='he_normal')
                if _f < 256:
                    x = tf.layers.max_pooling2d(x, 2, 2)
            x = self.upscale(x, 'nearest', scale=[2, 2], direct_output=False)
            G = [128, 64]
            for _g in G:
                x = self.conv2d(x, _g, 3, activation='lrelu', kernel_initializer='he_normal')
                x = self.conv2d(x, _g, 3, activation='lrelu', kernel_initializer='he_normal')
                x = self.upscale(x, 'nearest', scale=[2, 2], direct_output=False)
            x = self.conv2d(x, 32, 3, activation='lrelu', kernel_initializer='he_normal')
            x = self.conv2d(x, 2, 3, activation='tanh', kernel_initializer='he_normal')
            return x

    def _SRnet(self, inputs):
        with tf.variable_scope('SRNet', reuse=tf.AUTO_REUSE):
            x = self.conv2d(inputs, 64, 3, activation='relu', kernel_initializer='he_normal')
            for _ in range(10):
                x_old = x
                x = self.conv2d(x, 64, 3, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 64, 3, kernel_initializer='he_normal')
                x += x_old
            x = self.upscale(x, 'deconv', direct_output=False, activator=tf.nn.relu)
            x = self.conv2d(x, self.channel, 3, kernel_initializer='he_normal')
            return x

    def _warp(self, inputs, warp):
        return inputs

    def build_graph(self):
        with tf.variable_scope(self.name):
            self.inputs.append(tf.placeholder(tf.float32, [None, None, None, None, None]))
            self.inputs[-1].set_shape([None, self.depth, None, None, self.channel])
            B = tf.shape(self.inputs[-1])[0]
            H = tf.shape(self.inputs[-1])[2]
            W = tf.shape(self.inputs[-1])[3]
            i0 = self.inputs[-1][:, 0, ...]
            blank = tf.zeros([B, H, W, self.channel * self.scale[0] * self.scale[1]])
            outp = [self._SRnet(tf.concat([i0, blank], axis=-1))]
            lr_warp = []
            for i in range(1, self.depth):
                ix = self.inputs[-1][:, i, ...]
                ip = self.inputs[-1][:, i - 1, ...]
                warp_i = self._Fnet(ip, ix)
                warp_x = self.upscale(warp_i, direct_output=False)
                x = tf.space_to_depth(self._warp(outp[i - 1], warp_x), self.scale)
                outp.append(self._SRnet(tf.concat([ix, x], axis=-1)))
                lr_warp.append(self._warp(ip, warp_i))
            self.lr_warp = tf.stack(lr_warp, axis=1)
            self.outputs = [tf.stack(outp, axis=1)]

    def build_loss(self):
        self.label.append(tf.placeholder(tf.float32, [None, self.depth, None, None, self.channel]))
        with tf.name_scope('loss'):
            mse = tf.losses.mean_squared_error(self.label[-1], self.outputs[-1])
            motion_loss = tf.losses.mean_squared_error(self.inputs[-1][:, :self.depth - 1, ...], self.lr_warp)

            loss = mse + motion_loss
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(op)

            self.metrics['mse'] = mse
            self.metrics['me_mse'] = motion_loss
            self.train_metric['loss'] = loss
