"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 17th 2018
Updated Date: May 17th 2018

"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util.Utility import *

import tensorflow as tf
import numpy as np


class Denoise(SuperResolution):

    def __init__(self, name='denoise', **kwargs):
        self.name = name
        super(Denoise, self).__init__(scale=1, **kwargs)

    def build_graph(self):
        self.inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1]))
        self.inp = tf.cast(self.inputs[-1], tf.float32, name='cast/input_lr')
        nn = list()
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(64, 3, 1, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        nn.append(tf.layers.Conv2D(1, 3, 1, padding='same',
                                   kernel_initializer=tf.keras.initializers.he_normal()))
        x = self.inp
        for _n in nn:
            x = _n(x)
            self.trainable_weights += [_n.kernel]
        self.outputs.append(x)

    def build_loss(self):
        self.label.append(tf.placeholder(
            tf.uint8, shape=[None, None, None, 1], name='input_label_gray'))
        y_true = tf.cast(self.label[-1], tf.float32, name='cast/input_label')
        diff = self.inp - y_true - self.outputs[-1]
        mse = tf.reduce_mean(tf.square(diff), name='loss/mse')
        l2_loss = tf.add_n(tf.losses.get_regularization_losses())
        loss = mse + l2_loss
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.loss.append(optimizer.minimize(loss))
        self.metrics['mse'] = mse
        self.metrics['l2'] = l2_loss
        self.metrics['psnr'] = 20 * tf.log(255.0 / tf.sqrt(mse)) / tf.log(10.0)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/l2', self.metrics['l2'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
