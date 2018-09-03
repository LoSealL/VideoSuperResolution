"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 12th 2018
Updated Date: May 25th 2018

Deep Laplacian Pyramid Networks
Ref http://vllab.ucmerced.edu/wlai24/LapSRN
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import *

import tensorflow as tf
import numpy as np


class LapSRN(SuperResolution):

    def __init__(self, layers, epsilon=1e-3, name='lapsrn', **kwargs):
        self.epsilon2 = epsilon ** 2
        self.name = name
        super(LapSRN, self).__init__(**kwargs)
        s0, s1 = self.scale
        if np.log2(s0) != np.round(np.log2(s0)) or np.log2(s1) != np.round(np.log2(s1)):
            raise ValueError(f'Scale factor must be pow of 2. Error: scale={s0},{s1}')
        assert s0 == s1
        self.level = int(np.log2(s0))
        self.layers = to_list(layers, self.level)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(LapSRN, self).build_graph()
            x = self.inputs_preproc[-1]
            residual = []
            with tf.variable_scope('FeatureExtraction'):
                for lv in range(self.level):
                    for _ in range(self.layers[lv] - 1):
                        x = self.conv2d(x, 64, 3, activation='lrelu',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer='l2')
                    x = self.deconv2d(x, 64, 4, 2, activation='lrelu',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer='l2')
                    x = self.conv2d(x, self.channel, 3,
                                    kernel_initializer='he_normal', kernel_regularizer='l2')
                    residual.append(x)
            with tf.name_scope('Reconstruction'):
                y = self.inputs_preproc[-1]
                _s = 2
                for res in residual:
                    sr = bicubic_rescale(y, _s) + res
                    _s *= 2
                    self.outputs.append(sr)
            self.outputs.reverse()

    def build_loss(self):
        with tf.name_scope('loss'):
            y_true = [self.label[-1]]
            for _ in range(1, self.level):
                y_true.append(bicubic_rescale(y_true[-1], 0.5))
            charbonnier = []
            mse = []
            for x, y in zip(self.outputs, y_true):
                charbonnier.append(tf.reduce_mean(tf.sqrt(tf.square(x - y) + self.epsilon2)))
                mse.append(tf.reduce_mean(tf.squared_difference(y, x)))
            charbonnier_loss = tf.reduce_mean(charbonnier)
            loss = tf.add_n([charbonnier_loss] + tf.losses.get_regularization_losses())
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.loss.append(opt.minimize(loss, self.global_steps))

            self.train_metric['loss'] = loss
            self.metrics['charbonnier_loss'] = charbonnier_loss
            for i in range(len(mse)):
                self.metrics[f'mse_x{2**(i+1)}'] = mse[i]
                self.metrics[f'psnr_x{2**(i+1)}'] = 10 * tf.log(255 ** 2 / mse[i]) / tf.log(10.0)

    def build_summary(self):
        tf.summary.scalar('loss', self.metrics['charbonnier_loss'])
        for i in range(self.level):
            tf.summary.scalar(f'mse_x{2**(i+1)}', self.metrics[f'mse_x{2**(i+1)}'])
