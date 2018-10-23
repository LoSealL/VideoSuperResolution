"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 12th 2018

Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation (CVPR 2017)
See https://arxiv.org/abs/1611.05250
"""

from VSR.Framework.SuperResolution import SuperResolution
from VSR.Framework.Motion import warp, viz_flow
from VSR.Util.Utility import *

import tensorflow as tf


class VESPCN(SuperResolution):
    """Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation

    Args:
        depth:
    """

    def __init__(self, name='vespcn', depth=3, beta=1, gamma=0.01, **kwargs):
        super(VESPCN, self).__init__(**kwargs)
        self.name = name
        self.depth = depth
        self.beta = beta
        self.gamma = gamma

    def _flow_coarse(self, x0, x1, **kwargs):
        with tf.variable_scope('Flow/Coarse', reuse=tf.AUTO_REUSE):
            x = tf.concat([x0, x1], axis=-1)
            x = self.relu_conv2d(x, 24, 5, strides=2)
            x = self.relu_conv2d(x, 24, 3, strides=1)
            x = self.relu_conv2d(x, 24, 5, strides=2)
            x = self.relu_conv2d(x, 24, 3, strides=1)
            x = self.tanh_conv2d(x, 32, 3, strides=1)
            flow = pixel_shift(x, 4, 2)
            return flow

    def _flow_fine(self, x0, x1, flow, w, **kwargs):
        with tf.variable_scope('Flow/Fine', reuse=tf.AUTO_REUSE):
            x = tf.concat([x0, x1, flow, w], axis=-1)
            x = self.relu_conv2d(x, 24, 5, strides=2)
            x = self.relu_conv2d(x, 24, 3)
            x = self.relu_conv2d(x, 24, 3)
            x = self.relu_conv2d(x, 24, 3)
            x = self.tanh_conv2d(x, 8, 3)
            x = pixel_shift(x, 2, 2)
            return x

    def _me(self, x0, x1, **kwargs):
        with tf.variable_scope('MotionEstimation', reuse=tf.AUTO_REUSE):
            flow0 = self._flow_coarse(x0, x1)
            u0, v0 = flow0[..., 0], flow0[..., 1]
            w0 = warp(x1, u0, v0, True)
            flow_d = self._flow_fine(x0, x1, flow0, w0)
            flow1 = flow0 + flow_d
            u1, v1 = flow1[..., 0], flow1[..., 1]
            w1 = warp(x1, u1, v1, True)
            return w1, flow1

    def _stn(self, inputs, **kwargs):
        with tf.variable_scope(kwargs.get('name'), 'SpatialTemporalNet'):
            x = self.conv2d(inputs, 64, 3)
            sf = self.conv2d(x, 64, 3)
            x = self.resblock(sf, 64, 3, activation='relu', placement='front')
            x = self.resblock(x, 64, 3, activation='relu', placement='front')
            x = self.resblock(x, 64, 3, activation='relu', placement='front')
            sr = self.upscale(x + sf, direct_output=False)
            sr = self.conv2d(sr, self.channel, 3)
            return sr

    def _strange_huber_loss(self, inputs, epsilon=0.01):
        """The "Huber loss" used in paper."""
        diff_x = inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
        diff_y = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
        diff_x2 = diff_x ** 2
        diff_y2 = diff_y ** 2
        loss = tf.reduce_sum(diff_x2, axis=[1, 2, 3]) + tf.reduce_sum(diff_y2, axis=[1, 2, 3]) + epsilon
        return tf.reduce_mean(tf.sqrt(loss))

    def build_graph(self):
        self.inputs.append(tf.placeholder(tf.float32, [None, self.depth, None, None, self.channel], name='input/lr'))
        self.label.append(tf.placeholder(tf.float32, [None, self.depth, None, None, self.channel], name='label'))
        center = (self.depth - 1) // 2
        input_center = self.inputs[0][:, center, ...]
        label_center = self.label[0][:, center, ...]
        with tf.variable_scope(self.name):
            frames = tf.split(self.inputs[0], self.depth, axis=1)
            frames = [tf.squeeze(f, axis=1) for f in frames]
            warps = []
            flows = []
            for i in range(self.depth):
                if i == center: continue
                w, f = self._me(input_center, frames[i])
                warps.append(w)
                flows.append(f)
            ef = tf.concat(warps + [input_center], axis=-1)
            sr = self._stn(ef)
            self.outputs = [*flows, *warps, *frames, input_center, sr]
            self.WARP = warps

        with tf.name_scope('Loss'):
            loss_l2 = tf.losses.mean_squared_error(label_center, sr)
            loss_re = tf.losses.get_regularization_losses()
            loss_warps = [tf.losses.mean_squared_error(input_center, w) for w in warps]
            loss_flows = [self._strange_huber_loss(f) for f in flows]
            loss_me = tf.add_n([w * self.beta + f * self.gamma for w, f in zip(loss_warps, loss_flows)])
            loss = tf.add_n([loss_l2, loss_me] + loss_re)

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(opt)

        self.train_metric['l2'] = loss_l2
        self.train_metric['loss'] = loss
        self.train_metric['me'] = loss_me
        self.metrics['mse'] = loss_l2
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(label_center, sr, 255))

    def build_loss(self):
        pass

    def build_summary(self):
        tf.summary.scalar('loss', self.train_metric['loss'])
        tf.summary.scalar('mse', self.metrics['mse'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
        tf.summary.image('flow', viz_flow(self.outputs[0]), 1)
        tf.summary.image('SR', self.outputs[-1], 1)
        tf.summary.image('WARP/0', self.WARP[0], 1)
        tf.summary.image('WARP/1', self.WARP[1], 1)
