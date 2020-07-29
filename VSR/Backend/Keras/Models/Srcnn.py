#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

import tensorflow as tf

from .Model import SuperResolution


class Srcnn(tf.keras.Model):
  def __init__(self, channel, filters):
    super(Srcnn, self).__init__()
    self.net = [
      tf.keras.layers.Conv2D(64, filters[0], padding='same',
                             activation=tf.nn.relu),
      tf.keras.layers.Conv2D(32, filters[1], padding='same',
                             activation=tf.nn.relu),
      tf.keras.layers.Conv2D(channel, filters[2], padding='same')]

  def call(self, inputs):
    x = inputs
    for layer in self.net:
      x = layer(x)
    return x


class SRCNN(SuperResolution):
  def __init__(self, channel, scale, **kwargs):
    super(SRCNN, self).__init__(scale=scale, channel=channel, name='srcnn')
    self.net = Srcnn(channel, kwargs.get('filters', (9, 1, 5)))
    self.net(tf.keras.Input([None, None, channel]))
    self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

  def train(self, inputs, labels, learning_rate=None):
    lr_image = inputs[0]
    _, H, W, _ = lr_image.shape
    bi_image = tf.image.resize(lr_image, [H * self.scale, W * self.scale],
                               tf.image.ResizeMethod.BICUBIC)
    with tf.GradientTape() as tape:
      sr = self.net(bi_image)
      pixel_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels[0], sr))
    variables = self.trainable_variables()
    grads = tape.gradient(pixel_loss, variables)
    if learning_rate:
      self.opt.learning_rate = learning_rate
    self.opt.apply_gradients(zip(grads, variables))
    return {
      'loss': pixel_loss.numpy()
    }

  def eval(self, inputs, labels=None, **kwargs):
    metrics = {}
    lr_image = inputs[0]
    _, H, W, _ = lr_image.shape
    bi_image = tf.image.resize(lr_image, [H * self.scale, W * self.scale],
                               tf.image.ResizeMethod.BICUBIC)
    sr = self.net(bi_image)
    if labels is not None:
      metrics['psnr'] = tf.image.psnr(sr, labels[0], 1.0)
      step = kwargs.get('epoch')
      tf.summary.image('sr', sr, step=step, max_outputs=1)
      tf.summary.image('bicubic', bi_image, step=step, max_outputs=1)
      tf.summary.image('gt', labels[0], step=step, max_outputs=1)
    return [sr.numpy()], metrics
