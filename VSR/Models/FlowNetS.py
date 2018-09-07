"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug 31st 2018

Submodel of FlowNet 2.0
See https://arxiv.org/abs/1612.01925
"""

from ..Framework.SuperResolution import SuperResolution
from ..Framework.Motion import warp
from ..Util import *

import tensorflow as tf
from functools import partial


class FlowNetS(SuperResolution):
    def __init__(self, name='FlowNetS', depth=2, scale=None, **kwargs):
        super(FlowNetS, self).__init__(scale=1, **kwargs)
        self.name = name
        self.depth = depth
        self.feature_index = 0
        self.label_index = 1

    def _flownetS(self):
        with tf.variable_scope('FlowNetS'):
            self.inputs.append(tf.placeholder(tf.float32, [None, self.depth, None, None, self.channel]))
            self.label.append(tf.placeholder(tf.float32, [None, 1, None, None, 2]))

            # subtract mean of each patch
            input_mean = tf.reduce_mean(self.inputs[0], axis=[1, 2], keepdims=True)
            input_norm = (self.inputs[0] - input_mean) / 255
            # concat input 2 images
            input_split = tf.split(input_norm, self.depth, axis=1)
            input_concat = tf.concat(input_split, axis=-1)
            input_concat = tf.squeeze(input_concat, axis=1)

            bconv = partial(self.conv2d, kernel_initializer='he_normal', use_batchnorm=True)
            predict_flow = partial(self.conv2d, filters=2, kernel_size=3, kernel_initializer='he_normal')

            with tf.variable_scope('Flow'):
                x1 = bconv(input_concat, 64, 7, strides=2, activation='lrelu')
                x2 = bconv(x1, 128, 5, strides=2, activation='lrelu')
                x3 = bconv(x2, 256, 5, strides=2, activation='lrelu')
                x3_1 = bconv(x3, 256, 3, activation='lrelu')
                x4 = bconv(x3_1, 512, 3, strides=2, activation='lrelu')
                x4_1 = bconv(x4, 512, 3, activation='lrelu')
                x5 = bconv(x4_1, 512, 3, strides=2, activation='lrelu')
                x5_1 = bconv(x5, 512, 3, activation='lrelu')
                x6 = bconv(x5_1, 1024, 3, strides=2, activation='lrelu')
                x6_1 = bconv(x6, 1024, 3, activation='lrelu')

            with tf.variable_scope('Refine'):
                flow6 = predict_flow(x6_1)
                flow6_up = self.deconv2d(flow6, 2, 4, 2, use_bias=False)
                x6_up = self.deconv2d(x6_1, 512, 4, 2, activation='lrelu', kernel_initializer='he_normal')

                concat5 = tf.concat([x5_1, x6_up, flow6_up], axis=-1)
                flow5 = predict_flow(concat5)
                flow5_up = self.deconv2d(flow5, 2, 4, 2, use_bias=False)
                x5_up = self.deconv2d(concat5, 256, 4, 2, activation='lrelu', kernel_initializer='he_normal')

                concat4 = tf.concat([x4_1, x5_up, flow5_up], axis=-1)
                flow4 = predict_flow(concat4)
                flow4_up = self.deconv2d(flow4, 2, 4, 2, use_bias=False)
                x4_up = self.deconv2d(concat4, 128, 4, 2, activation='lrelu', kernel_initializer='he_normal')

                concat3 = tf.concat([x3_1, x4_up, flow4_up], axis=-1)
                flow3 = predict_flow(concat3)
                flow3_up = self.deconv2d(flow3, 2, 4, 2, use_bias=False)
                x3_up = self.deconv2d(concat3, 64, 4, 2, activation='lrelu', kernel_initializer='he_normal')

                concat2 = tf.concat([x2, x3_up, flow3_up], axis=-1)
                flow2 = predict_flow(concat2)

                flow = Utility.bicubic_rescale(flow2, 4)

            self.outputs = [flow, flow2, flow3, flow4, flow5, flow6]

        with tf.name_scope('loss'):
            target_flow = self.label[0][:, 0, ...] / 20
            weights = [0.32, 0.08, 0.02, 0.01, 0.005]
            weights.reverse()
            scales = [4, 8, 16, 32, 64]
            multiscale_l1loss = []
            for _f, _s, _w in zip(self.outputs[1:], scales, weights):
                scaled_flow = target_flow
                if _s > 1:
                    scaled_flow = tf.layers.average_pooling2d(target_flow, _s, _s)
                multiscale_l1loss.append(tf.losses.absolute_difference(scaled_flow, _f, _w))
            loss = tf.add_n(multiscale_l1loss, name='Loss/Multiscale_L1')
            epe = tf.losses.mean_squared_error(target_flow, flow)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(op)

            self.train_metric['Loss'] = loss
            self.metrics['EPE'] = epe
            self.metrics['Loss'] = loss

        tf.summary.scalar('EPE', self.metrics['EPE'])
        tf.summary.scalar('Loss', self.metrics['Loss'])

    def _spmc(self):
        def _Fnet(f0, f1):
            with tf.variable_scope('FNet', reuse=tf.AUTO_REUSE):
                x = tf.concat([f0, f1], axis=-1)
                x = self.conv2d(x, 24, 5, strides=2, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 24, 3, strides=1, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 24, 5, strides=2, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 24, 3, strides=1, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 32, 3, strides=1, activation='tanh', kernel_initializer='he_normal')
                coarse_warp = tf.depth_to_space(x, 4)

                return coarse_warp

        def _Snet(f0, f1, flow, warped):
            with tf.variable_scope('SRNet', reuse=tf.AUTO_REUSE):
                x = tf.concat([f0, f1, flow, warped], axis=-1)
                x = self.conv2d(x, 24, 5, strides=2, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 24, 3, strides=1, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 24, 3, strides=1, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 24, 3, strides=1, activation='relu', kernel_initializer='he_normal')
                x = self.conv2d(x, 8, 3, strides=1, activation='tanh', kernel_initializer='he_normal')
                fine_warp = tf.depth_to_space(x, 2)

                return fine_warp

        with tf.variable_scope('SPMC'):
            self.inputs.append(tf.placeholder(tf.float32, [None, self.depth, None, None, self.channel]))
            self.label.append(tf.placeholder(tf.float32, [None, 1, None, None, 2]))

            input_norm = self.inputs[0] / 255
            # concat input 2 images
            target = input_norm[:, 0, ...]
            refer = input_norm[:, 1, ...]

            coarse = _Fnet(target, refer)
            target_hat = warp(refer, coarse[..., 0], coarse[..., 1], True)
            fine = _Snet(target, refer, coarse, target_hat)
            flow = coarse + fine
            target_hat2 = warp(refer, flow[..., 0], flow[..., 1], True)
            self.outputs.append(flow)
            self.outputs.append(target_hat2 * 255)

        with tf.name_scope('loss'):
            label_flow = self.label[0][:, 0, ...]
            image_mse = tf.losses.mean_squared_error(target_hat2, target)
            image_mse_coarse = tf.losses.mean_squared_error(target_hat, target)
            flow_epe = tf.losses.mean_squared_error(label_flow, flow)

            loss = image_mse + image_mse_coarse * 1e-3 + flow_epe * 1e-4

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, self.global_steps)
                self.loss.append(op)

            self.metrics['image'] = image_mse
            self.metrics['flow'] = flow_epe
            self.train_metric['image'] = image_mse
            self.train_metric['loss'] = loss

        tf.summary.image('coarse', target_hat)
        tf.summary.image('fine', target_hat2)
        tf.summary.image('ref', target)

    def build_graph(self):
        self._flownetS()

    def build_loss(self):
        pass

    def build_summary(self):
        pass
