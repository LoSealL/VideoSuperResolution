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

    def build_graph(self):
        with tf.variable_scope(self.name):
            self.inputs.append(tf.placeholder(tf.float32, [None, self.depth, None, None, self.channel]))
            self.label.append(tf.placeholder(tf.float32, [None, 1, None, None, 2]))

            input_norm = self.inputs[0] / 255
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

                flow = self.upscale(flow2, scale=[4, 4], direct_output=False)

            self.outputs = [flow, flow2, flow3, flow4, flow5, flow6]

    def build_loss(self):
        with tf.variable_scope('warp'):
            u = self.outputs[0][..., 0]
            v = self.outputs[0][..., 1]
            ref = self.inputs[0][:, 1, ...]
            ref_hat = warp(ref, u, v, True)

        with tf.name_scope('loss'):
            target_flow = self.label[0][:, 0, ...]
            weights = [1, 0.32, 0.08, 0.02, 0.01, 0.005]
            scales = [1, 4, 8, 16, 32, 64]
            multiscale_epe = []
            for flow, scale, weight in zip(self.outputs, scales, weights):
                scaled_flow = target_flow
                if scale > 1:
                    scaled_flow = tf.layers.max_pooling2d(target_flow, scale, scale)
                epe = tf.losses.mean_squared_error(scaled_flow, flow, weights=weight)
                multiscale_epe.append(epe)
            epe_loss = tf.add_n(multiscale_epe, name='Loss/EPE')

            target_image = self.inputs[0][:, 0, ...]
            warp_loss = tf.losses.mean_squared_error(target_image, ref_hat)

            loss = epe_loss + warp_loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    loss, self.global_steps)
                self.loss.append(op)

            self.train_metric['EPE'] = epe_loss
            self.train_metric['IMG'] = warp_loss
            self.train_metric['Loss'] = loss
            self.metrics['EPE'] = multiscale_epe[0]
            self.metrics['Image'] = warp_loss

        tf.summary.scalar('EPE', self.metrics['EPE'])
        tf.summary.scalar('Image', self.metrics['Image'])
        tf.summary.image('Ref', self.inputs[0][:, 0, ...])
        tf.summary.image('Warp', ref_hat)

    def build_summary(self):
        pass
