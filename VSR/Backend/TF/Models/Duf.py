"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 9th 2018

Dynamic Upsampling Filters (CVPR 2018)
See http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf
"""

import numpy as np
import tensorflow as tf

from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util import *


class DUF(SuperResolution):
  """Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation

  """

  STP = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
  SP = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

  def __init__(self, layers=16, filter_size=(5, 5), depth=7, name='duf',
               **kwargs):
    super(DUF, self).__init__(**kwargs)
    self.layers = layers
    self.filter_size = Utility.to_list(filter_size, 2)
    self.depth = depth
    self.name = name

  @staticmethod
  def _normalize(x):
    return x / 255

  @staticmethod
  def _denormalize(x):
    return x * 255

  def _dyn_filter3d(self, x, F):
    """
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    """
    # make tower
    size = np.prod(self.filter_size)
    filter_localexpand_np = np.eye(size)
    filter_localexpand_np = filter_localexpand_np.reshape(
      [*self.filter_size, 1, size])
    x = tf.expand_dims(x, axis=-1)
    x_localexpand = tf.nn.conv2d(x, filter_localexpand_np, [1, 1, 1, 1],
                                 'SAME')  # b, h, w, 1*5*5
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5
    x = tf.matmul(x_localexpand, F)  # b, h, w, 1, R*R
    x = tf.squeeze(x, axis=3)  # b, h, w, R*R

    return x

  def build_graph(self):
    self.inputs.append(
      tf.placeholder(tf.float32, [None, None, None, None, self.channel],
                     name='input/lr'))
    self.label.append(
      tf.placeholder(tf.float32, [None, None, None, None, self.channel],
                     name='label/hr'))

    input_norm = self._normalize(self.inputs[-1])
    with tf.variable_scope(self.name):
      F, G = 64, 32
      t = tf.pad(input_norm, self.SP)
      t = self.conv3d(t, F, (1, 3, 3), padding='valid')
      for i in range(self.layers - 3):
        x = tf.layers.batch_normalization(t, training=self.training_phase)
        x = tf.nn.relu(x)
        x = self.conv3d(x, F, 1, activation='relu', use_batchnorm=True)
        x = self.conv3d(x, G, 3)
        t = tf.concat([t, x], axis=-1)
        F += G
      for i in range(3):
        x = tf.layers.batch_normalization(t, training=self.training_phase)
        x = tf.nn.relu(x)
        x = self.conv3d(x, F, 1, activation='relu', use_batchnorm=True)
        x = tf.pad(x, self.SP)
        x = self.conv3d(x, G, 3, padding='valid')
        t = tf.concat([t[:, 1:-1], x], axis=-1)
        F += G
      t = tf.layers.batch_normalization(t, training=self.training_phase)
      t = tf.nn.relu(t)
      t = self.conv3d(t, 256, (1, 3, 3), activation='relu')

      r = self.relu_conv3d(t, 256, 1)
      r = self.relu_conv3d(r, np.prod([self.channel, *self.scale]), 1)
      r = r[:, 0]
      r = Utility.pixel_shift(r, self.scale, self.channel)

      f = self.relu_conv3d(t, 512, 1)
      f = self.conv3d(f, np.prod(self.filter_size) * np.prod(self.scale), 1)
      ds_f = tf.shape(f)
      f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3],
                         np.prod(self.filter_size), np.prod(self.scale)])
      f = tf.nn.softmax(f, axis=4)

      sr = []
      for i in range(self.channel):
        x = self._dyn_filter3d(input_norm[:, self.depth // 2, ..., i], f[:, 0])
        sr.append(Utility.pixel_shift(x, self.scale))
      sr = tf.concat(sr, axis=-1)
      sr += r
      self.outputs.append(self._denormalize(sr))

  def build_loss(self):
    sr_norm = self._normalize(self.outputs[-1])
    label_norm = self._normalize(self.label[-1])
    label_c = label_norm[:, self.depth // 2]

    with tf.name_scope('Loss'):
      huber = tf.losses.huber_loss(label_c, sr_norm, delta=0.01)
      mse = tf.losses.mean_squared_error(label_c, sr_norm)
      reg = tf.losses.get_regularization_losses()

      loss = tf.add_n([huber] + reg)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
          loss, self.global_steps)
        self.loss.append(opt)

    self.train_metric['loss'] = loss
    self.metrics['mse'] = mse
    self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(label_c, sr_norm, 1))

  def build_summary(self):
    tf.summary.scalar('loss', self.train_metric['loss'])
    tf.summary.scalar('mse', self.metrics['mse'])
    tf.summary.scalar('psnr', self.metrics['psnr'])
    tf.summary.image('SR', self.outputs[-1], 1)
