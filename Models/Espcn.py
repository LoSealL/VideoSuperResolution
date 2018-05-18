"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 12th 2018
Updated Date: May 12th 2018

Spatial transformer motion compensation model
Ref https://arxiv.org/abs/1501.00092
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util import Utility

import tensorflow as tf
import numpy as np


def get_perceptual_loss(block, conv):
    """Perceptual Loss Function

    paper: https://arxiv.org/abs/1603.08155
    """

    VGG = tf.keras.applications.vgg16.VGG16
    layer_name = f'block{block}_conv{conv}'
    vgg = VGG(input_shape=[None, None, 3], include_top=False)
    inp = vgg.input
    outp = vgg.get_layer(layer_name).output
    model = tf.keras.models.Model(inputs=inp, outputs=outp, name='perceptual_loss')

    def _vgg_normalize(x):
        shape = tf.shape(x)
        x = tf.cond(shape[-1] == 1, true_fn=lambda: tf.image.grayscale_to_rgb(x), false_fn=lambda: x[..., :3])
        x = x[:, :, :, ::-1] - 114.0
        return x

    def loss(y_true, y_pred):
        y_true.set_shape(y_pred.shape)
        y_true_norm = _vgg_normalize(y_true)
        y_pred_norm = _vgg_normalize(y_pred)
        feature_true = model(y_true_norm)
        feature_pred = model(y_pred_norm)
        return tf.reduce_mean(tf.square(feature_true - feature_pred))

    return loss


class Espcn(SuperResolution):

    def __init__(self, scale, channel=1, name='espcn', **kwargs):
        self.channel = channel
        self.name = name
        super(Espcn, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        self.inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None, self.channel]))
        cast_inp = tf.cast(self.inputs[-1], tf.float32)
        nn = list()
        l2_decay = 1e-4
        nn.append(tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        nn.append(tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        nn.append(tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        nn.append(tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        nn.append(tf.layers.Conv2D(self.scale[0] * self.scale[1] * self.channel, 3, padding='same',
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        x = cast_inp
        for _n in nn:
            x = _n(x)
            self.trainable_weights += [_n.kernel]
        outp = Utility.pixel_shift(x, self.scale, self.channel)
        nn = list()
        nn.append(tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        nn.append(tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        nn.append(tf.layers.Conv2D(1, 3, padding='same', activation=tf.nn.relu,
                                   kernel_initializer=Utility.he_initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_decay)))
        x = outp + tf.random_normal(tf.shape(outp), stddev=25)
        for _n in nn:
            x = _n(x)
        self.outputs = [x, outp]

    def build_loss(self):
        self.label.append(tf.placeholder(tf.uint8, shape=[None, None, None, self.channel]))
        y_true = tf.cast(self.label[-1], tf.float32)
        diff = self.outputs[1] - y_true - self.outputs[0]
        mse = tf.reduce_mean(tf.square(diff))
        regularization = tf.losses.get_regularization_losses()
        regularization = tf.add_n(regularization)
        tv = tf.image.total_variation(self.outputs[1]) * 1e-6
        loss = mse + regularization + tv
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.loss.append(optimizer.minimize(loss))
        self.metrics['mse'] = mse
        self.metrics['regular'] = regularization
        self.metrics['psnr'] = 20 * tf.log(255.0 / tf.sqrt(mse)) / tf.log(10.0)
        self.metrics['ssim'] = tf.image.ssim(y_true, self.outputs[0], max_val=255)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/regularization', self.metrics['regular'])
        tf.summary.scalar('psnr', self.metrics['psnr'])
