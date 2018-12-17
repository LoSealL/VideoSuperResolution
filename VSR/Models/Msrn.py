"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 14th 2018

Multi-scale Residual Network for Image Super-Resolution
See http://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf
"""

from ..Framework.SuperResolution import SuperResolution
from ..Arch.Residual import msrb

import tensorflow as tf


def _normalize(inputs):
    rgb_mean = (0.4488, 0.4371, 0.4040)
    return inputs / 255 - rgb_mean


def _denormalize(inputs):
    rgb_mean = (0.4488, 0.4371, 0.4040)
    return (inputs + rgb_mean) * 255


class MSRN(SuperResolution):
    """Multi-scale Residual Network for Image Super-Resolution

    Args:
        n_blocks: number of MSRB blocks.
    """

    def __init__(self, n_blocks=8, name='msrn', **kwargs):
        super(MSRN, self).__init__(**kwargs)
        self.name = name
        self.blocks = n_blocks

    def build_graph(self):
        super(MSRN, self).build_graph()
        inputs_norm = _normalize(self.inputs_preproc[-1])
        with tf.variable_scope(self.name):
            features = [self.conv2d(inputs_norm, 64, 3)]
            for _ in range(self.blocks):
                x = features[-1]
                features.append(msrb(self, x))
            x = self.conv2d(tf.concat(features, -1), 64, 1)
            x = self.upscale(x, direct_output=False)
            sr = self.conv2d(x, self.channel, 3)
        self.outputs.append(_denormalize(sr))

    def build_loss(self):
        label_norm = _normalize(self.label[-1])
        sr = _normalize(self.outputs[-1])
        with tf.name_scope('Loss'):
            l1 = tf.losses.absolute_difference(label_norm, sr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op = tf.train.AdamOptimizer(self.learning_rate)
                op = op.minimize(l1, self.global_steps)
                self.loss.append(op)

        self.train_metric['l1'] = l1
        self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(
            self.label[-1], self.outputs[-1], max_val=255))
        self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(
            self.label[-1], self.outputs[-1], max_val=255))

    def build_summary(self):
        super(MSRN, self).build_summary()
        tf.summary.image('sr', self.outputs[0])
