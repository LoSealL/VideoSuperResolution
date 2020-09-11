"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 19th 2018

Non-Local Recurrent Network for Image Restoration
See https://arxiv.org/abs/1806.02919
"""

import logging

from .. import tf, tfc
from ..Arch.Residual import non_local
from ..Framework.SuperResolution import SuperResolution

LOG = logging.getLogger('VSR.Model.NLRN')


def _denormalize(inputs):
    return (inputs + 0) * 255


def _normalize(inputs):
    return inputs / 255


class NLRN(SuperResolution):
    """Non-Local Recurrent Network for Image Restoration (NIPS 2018)

    """

    def __init__(self, recurrents=12, clip=2.5, name='nlrn', **kwargs):
        super(NLRN, self).__init__(**kwargs)
        self.name = name
        self.recurrents = recurrents
        self.clip = clip
        self.filters = kwargs.get('filters', 128)

    def display(self):
        LOG.info(f"Recurrents: {self.recurrents}")

    def rnn(self, x, y):
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            x = self.batch_norm(x, self.training_phase)
            x = tf.nn.relu(x)
            x = non_local(self, x, self.filters, scaling=2)

            x = self.batch_norm(x, self.training_phase)
            x = self.bn_relu_conv2d(x, self.filters, 3)
            x = self.conv2d(x, self.filters, 3)
            return x + y

    def build_graph(self):
        super(NLRN, self).build_graph()
        with tf.variable_scope(self.name):
            inputs_norm = _normalize(self.inputs_preproc[-1])
            init_feat = self.batch_norm(inputs_norm, self.training_phase)
            x = init_feat = self.conv2d(init_feat, self.filters, 3)
            for _ in range(self.recurrents):
                x = self.rnn(x, init_feat)
            sr = self.batch_norm(x, self.training_phase)
            sr = tf.nn.relu(sr)
            sr = self.conv2d(sr, self.channel, 3)
            sr += inputs_norm

            self.outputs.append(_denormalize(sr))

    def build_loss(self):
        with tf.name_scope('Loss'):
            mse = tf.losses.mean_squared_error(
                self.outputs[-1], self.label[-1])

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdadeltaOptimizer(self.learning_rate)
                grad = opt.compute_gradients(mse)
                grad_clip = tfc.training.clip_gradient_norms(
                    grad, self.clip)
                op = opt.apply_gradients(grad_clip, self.global_steps)
                self.loss.append(op)

            self.train_metric['mse'] = mse
            self.metrics['psnr'] = tf.reduce_mean(
                tf.image.psnr(self.label[0], self.outputs[0], 255))
            self.metrics['ssim'] = tf.reduce_mean(
                tf.image.ssim(self.label[0], self.outputs[0], 255))

    def build_summary(self):
        super(NLRN, self).build_summary()
        tf.summary.image('SR', self.outputs[-1], 1)
