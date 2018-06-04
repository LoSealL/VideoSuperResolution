"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 17th 2018
Updated Date: May 25th 2018

"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util.Utility import *

import tensorflow as tf
import numpy as np


class Denoise(SuperResolution):

    def __init__(self, layers=(8, 8), noise_decay=0.1, name='denoise', **kwargs):
        self.layers = to_list(layers, 2)
        self.noise_decay = noise_decay
        self.name = name
        super(Denoise, self).__init__(**kwargs)

    def build_graph(self):
        with tf.name_scope(self.name):
            super(Denoise, self).build_graph()
            self.inputs.append(tf.placeholder(tf.float32, shape=()))
            x = self.inputs_preproc[-1]
            # with tf.name_scope('noise_extraction'):
            #     for _ in range(self.layers[0] - 1):
            #         x = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.relu,
            #                              kernel_initializer=tf.keras.initializers.he_normal(),
            #                              kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))
            #     x = tf.layers.conv2d(x, 1, 3, padding='same',
            #                          kernel_initializer=tf.keras.initializers.he_normal())
            # self.noise = x
            # self.outputs.append(x)

            noise = tf.ones_like(self.inputs_preproc[-1], dtype=tf.float32) * self.inputs[-1]
            tf.summary.image('input', x)
            x = tf.concat([self.inputs_preproc[-1], noise], axis=-1)
            with tf.name_scope('upscale'):
                for _ in range(self.layers[1] - 1):
                    x = self.conv2d(x, 128, 3, kernel_initializer='he_normal', kernel_regularizer='l2',
                                    activation='relu', use_batchnorm=True)
                x = self.conv2d(x, self.scale[0] * self.scale[1], 3, kernel_initializer='he_normal',
                                kernel_regularizer='l2')
                x = pixel_shift(x, self.scale, 1)
            self.outputs.append(x)
            tf.summary.image('output', x)

    def build_loss(self):
        with tf.name_scope('loss'):
            self.label.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1]))
            # SR loss part
            y_true = tf.cast(self.label[-1], tf.float32, name='cast/input_label')
            y_pred = self.outputs[-1]
            mse = tf.losses.mean_squared_error(y_true, y_pred)
            # Denoise loss part
            # y_lr = tf.image.resize_bicubic(y_true, tf.shape(self.inputs_preproc[-1])[1:3])
            # y_clear = self.inputs_preproc[-1] - self.outputs[0]
            # tf.summary.image('denoise', y_clear)
            # noise_loss = tf.losses.mean_squared_error(y_lr, y_clear)
            # regularization
            regular_loss = tf.add_n(tf.losses.get_regularization_losses())
            loss = mse + regular_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.loss.append(optimizer.minimize(loss, self.global_steps))
            # Adding metrics
            self.metrics['mse'] = mse
            self.metrics['noise'] = tf.constant(0)
            self.metrics['regularization'] = regular_loss
            self.metrics['psnr'] = tf.image.psnr(y_true, y_pred, max_val=255)
            self.metrics['ssim'] = tf.image.ssim(y_true, y_pred, max_val=255)

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
        tf.summary.scalar('loss/noise', self.metrics['noise'])
        tf.summary.scalar('loss/regularization', self.metrics['regularization'])
        tf.summary.scalar('psnr', tf.reduce_mean(self.metrics['psnr']))
        tf.summary.scalar('ssim', tf.reduce_mean(self.metrics['ssim']))

    def export_model_pb(self, export_dir='.', export_name='model.pb', **kwargs):
        outp = self.outputs[-1]
        if self.rgba:
            outp = tf.concat([outp / 255, self.inputs_preproc[-2]], axis=-1)
            outp = tf.image.yuv_to_rgb(outp) * 255
        else:
            outp = tf.image.grayscale_to_rgb(outp)
        outp = tf.cast(tf.clip_by_value(outp, 0, 255), tf.uint8)
        outp = tf.concat([outp, tf.zeros_like(outp)[..., 0:1]], axis=-1, name='output/hr/rgba')
        self.outputs[-1] = outp
        super(Denoise, self).export_model_pb(export_dir, f'{self.name}.pb', **kwargs)


from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *


def learning_rate_decay(lr, epochs, steps, **kwargs):
    lr = tf.train.exponential_decay(1e-2, global_step=steps, decay_steps=1000, decay_rate=0.96)
    return lr.eval()


def add_noise(x):
    stddev = np.random.randint(0, 55)
    noised = x + np.random.normal(0, stddev, x.shape)
    return np.clip(x, 0, 255), 40


if __name__ == '__main__':
    model = Denoise(scale=3, layers=12, noise_decay=1e-2, rgb_input=False).compile()
    dataset = load_datasets('../Data/datasets.json')['BSD']
    dataset.setattr(patch_size=96, strides=96, random=True, max_patches=64 * 100)
    env = Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log')
    env.feature_callbacks = [add_noise]
    env.fit(64, 200, dataset, restart=False, learning_rate_schedule=learning_rate_decay, learning_rate=1e-2)
    env.feature_callbacks = [to_gray(), add_noise]
    env.label_callbacks = [to_gray()]
    env.output_callbacks += [lambda output, **kwargs: output[0]]
    env.output_callbacks += [to_rgb()]
    env.output_callbacks += [save_image(f'../Results/{model.name}/test')]
    env.test(dataset, convert_to_gray=False)  # load image with 3 channels
    # env.export()
