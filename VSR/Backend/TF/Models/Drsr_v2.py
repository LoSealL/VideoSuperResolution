#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 2 - 28

from functools import partial
import logging

import numpy as np
import tensorflow as tf

from VSR.Util import Config, to_list
from ..Arch.Residual import cascade_rdn
from ..Framework import Noise
from ..Framework.SuperResolution import SuperResolution
from ..Util import clip_image

_MEAN_GT = [84.1148, 68.3644, 64.8452]
_MEAN_SR = [85.6586, 68.7887, 66.5135]
LOG = logging.getLogger('VSR.Model.DRSRv2')


def _denormalize(inputs, shift):
  return inputs * 255 + shift


def _normalize(inputs, shift):
  return (inputs - shift) / 255


class DRSR(SuperResolution):
  def __init__(self, name='drsr_v2', noise_config=None, weights=(1, 10, 1e-5),
               level=1, mean_shift=(0, 0, 0), arch=None, auto_shift=None,
               **kwargs):
    super(DRSR, self).__init__(**kwargs)
    self.name = name
    self.noise = Config(scale=0, offset=0, penalty=0.5, max=0, layers=7)
    if isinstance(noise_config, (dict, Config)):
      self.noise.update(**noise_config)
      self.noise.crf = np.load(self.noise.crf)
      self.noise.offset = to_list(self.noise.offset, 4)
      self.noise.offset = [x / 255 for x in self.noise.offset]
      self.noise.max /= 255
    self.weights = weights
    self.level = level
    if mean_shift is not None:
      self.norm = partial(_normalize, shift=mean_shift)
      self.denorm = partial(_denormalize, shift=mean_shift)
    self.arch = arch
    self.auto = auto_shift
    self.to_sum = []

  def display(self):
    LOG.info(str(self.noise))

  def noise_cond(self, inputs, noise, layers, scope='NCL'):
    with tf.variable_scope(None, scope):
      x = noise
      c = inputs.shape[-1]
      for _ in range(layers - 1):
        x = self.prelu_conv2d(x, 64, 3)
      x = self.conv2d(x, c, 3)
      gamma = tf.nn.sigmoid(x)
      x = noise
      for _ in range(layers - 1):
        x = self.prelu_conv2d(x, 64, 3)
      beta = self.conv2d(x, c, 3)
      return inputs * gamma + beta

  def cond_rb(self, inputs, noise, scope='CRB'):
    with tf.variable_scope(None, scope):
      x = self.prelu_conv2d(inputs, 64, 3)
      x = self.conv2d(x, 64, 3)
      x = self.noise_cond(x, noise, 3)
      if inputs.shape[-1] != x.shape[-1]:
        sc = self.conv2d(inputs, x.shape[-1], 1,
                         kernel_initializer='he_uniform')
      else:
        sc = inputs
      return sc + x

  def cond_rdb(self, inputs, noise, scope='CRDB'):
    with tf.variable_scope(None, scope):
      x0 = self.prelu_conv2d(inputs, 64, 3)
      x1 = self.prelu_conv2d(tf.concat([inputs, x0], -1), 64, 3)
      x2 = self.conv2d(tf.concat([inputs, x0, x1], -1), 64, 3)
      x = self.noise_cond(x2, noise, 3)
      if inputs.shape[-1] != x.shape[-1]:
        sc = self.conv2d(inputs, x.shape[-1], 1,
                         kernel_initializer='he_uniform')
      else:
        sc = inputs
      return sc + x

  def noise_estimate(self, inputs, scope='NoiseEstimator', reuse=None):
    n = self.noise
    with tf.variable_scope(None, scope, reuse=reuse):
      x = inputs
      for _ in range(n.layers):
        x = self.leaky_conv2d(x, 64, 3)
      x = self.conv2d(x, self.channel, 3)
      return x

  def noise_shift(self, inputs, layers, scope='NoiseShift', reuse=None):
    n = self.noise
    with tf.variable_scope(None, scope, reuse=reuse):
      x = inputs
      for _ in range(layers):
        x = self.leaky_conv2d(x, 64, 3)
      x = self.conv2d(x, self.channel, 3, activation=tf.nn.sigmoid)
      return x * Noise.tf_gaussian_noise(inputs, n.max)

  def local_net(self, inputs, noise, depth=4, scope='LC'):
    with tf.variable_scope(None, scope):
      fl = [inputs]
      x = inputs
      for i in range(depth):
        x = self.cond_rb(x, noise)
        fl.append(x)
        x = tf.concat(fl, axis=-1)
        x = self.conv2d(x, 64, 1, kernel_initializer='he_uniform')
      return x

  def local_net2(self, inputs, noise, depth=4, scope='LC'):
    with tf.variable_scope(None, scope):
      fl = [inputs]
      x = inputs
      for i in range(depth):
        x = self.cond_rdb(x, noise)
        fl.append(x)
        x = tf.concat(fl, axis=-1)
        x = self.conv2d(x, 64, 1, kernel_initializer='he_uniform')
      return x

  def global_net(self, inputs, noise, depth=4, scope='GC', reuse=None):
    with tf.variable_scope(None, scope, reuse=reuse):
      fl = [inputs]
      x = inputs
      for i in range(depth):
        if self.arch == 'concat':
          x = cascade_rdn(self, x, depth=3, use_ca=True)
        elif self.arch == 'crb':
          x = self.local_net(x, noise[i], 4)
        else:
          x = self.local_net2(x, noise[i], 3)
        if self.arch != 'crdb':
          fl.append(x)
          x = tf.concat(fl, axis=-1)
          x = self.conv2d(x, 64, 1, kernel_initializer='he_uniform')
      self.to_sum += fl
      if self.arch == 'crdb':
        x += inputs
      if self.auto:
        sr = self.upscale(x, direct_output=False, scale=4)
      else:
        sr = self.upscale(x, direct_output=False)
      sr = self.conv2d(sr, self.channel, 3)
      return sr, x

  def gen_noise(self, inputs, ntype, max1=0.06, max2=0.16):
    with tf.name_scope('GenNoise'):
      n = self.noise
      if ntype == 'gaussian':
        noise = Noise.tf_gaussian_noise(inputs, sigma_max=max1,
                                        channel_wise=False)
        return noise
      elif ntype == 'crf':
        crf = tf.convert_to_tensor(n.crf['crf'])
        icrf = tf.convert_to_tensor(n.crf['icrf'])
        i = tf.random_uniform([], 0, crf.shape[0], dtype=tf.int32)
        irr = Noise.tf_camera_response_function(inputs, icrf[i], max_val=1)
        noise = Noise.tf_gaussian_poisson_noise(irr, max_c=max1, max_s=max2)
        img = Noise.tf_camera_response_function(irr + noise, crf[i], max_val=1)
        return img - inputs
      else:
        raise TypeError(ntype)

  def net(self, inputs, level, scale=1, shift=(0, 0, 0, 0), reuse=None):
    with tf.variable_scope(self.name, reuse=reuse):
      level_outputs = []
      level_noise = []
      level_inputs = []
      for i in range(1, level + 1):
        with tf.variable_scope(f'Level{i:1d}'):
          noise_hyp = self.noise_estimate(inputs) * scale + \
                      Noise.tf_gaussian_noise(inputs, self.noise.offset[0])
          level_noise.append(noise_hyp)
          noise_hyp = [noise_hyp + shift[0],
                       noise_hyp + shift[1],
                       noise_hyp + shift[2],
                       noise_hyp + shift[3]]
          if i == 1:
            if self.arch == 'concat':
              inputs = tf.concat([inputs, noise_hyp[0]], axis=-1)
            entry = self.conv2d(inputs, 64, 3)
            entry = self.conv2d(entry, 64, 3)
            level_inputs.append(entry)
          y = self.global_net(level_inputs[-1], noise_hyp, 4)
          level_outputs.append(y[0])
          level_inputs.append(y[1])
      return level_noise, level_outputs

  def build_graph(self):
    super(DRSR, self).build_graph()
    inputs_norm = self.norm(self.inputs_preproc[-1])
    labels_norm = self.norm(self.label[-1])
    n = self.noise
    if n.valid:
      LOG.info("adding noise")
      awgn = self.gen_noise(inputs_norm, 'gaussian', n.max)
      gp = self.gen_noise(inputs_norm, 'crf', 5 / 255, n.max)
    else:
      awgn = gp = tf.zeros_like(inputs_norm)

    if self.level == 1:
      noise = awgn
    elif self.level == 2:
      noise = gp
    else:
      raise NotImplementedError("Unknown level!")
    with tf.variable_scope('Offset'):
      shift = []
      if not self.auto:
        for i in range(4):
          shift.append(Noise.tf_gaussian_noise(inputs_norm, n.offset[i]))
        var_shift = []
      else:
        for i in range(4):
          shift.append(self.noise_shift(inputs_norm, 8, f'NoiseShift_{i}'))
        var_shift = tf.trainable_variables('Offset')

    noise_hyp, outputs = self.net(inputs_norm + noise, 1, n.scale, shift)
    self.outputs += [tf.abs(x * 255) for x in noise_hyp + shift]
    self.outputs += [self.denorm(x) for x in outputs]

    l1_image = tf.losses.absolute_difference(outputs[-1], labels_norm)
    noise_abs_diff = tf.abs(noise_hyp[-1]) - tf.abs(noise)
    # 1: over estimated; 0: under estimated
    penalty = tf.ceil(tf.clip_by_value(noise_abs_diff, 0, 1))
    # 1 - n: over estimated; n: under estimated
    penalty = tf.abs(n.penalty - penalty)
    noise_error = penalty * tf.squared_difference(noise_hyp[-1], noise)
    l2_noise = tf.reduce_mean(noise_error)

    # tv clamp
    tv = tf.reduce_mean(tf.image.total_variation(noise_hyp[-1]))
    l_tv_max = tf.nn.relu(tv - 1000) ** 2
    l_tv_min = tf.nn.relu(200 - tv) ** 2
    tv = tv + l_tv_max + l_tv_min

    def loss_fn1():
      w = self.weights
      loss = l1_image * w[0] + l2_noise * w[1] + tv * w[2]
      var_to_opt = tf.trainable_variables(self.name + f"/Level1")
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(self.learning_rate, 0.9)
        op = op.minimize(loss, self.global_steps, var_list=var_to_opt)
        self.loss.append(op)

      self.train_metric['mae'] = l1_image
      self.train_metric['noise_error'] = l2_noise
      self.train_metric['tv'] = tv
      self.to_sum += noise_hyp

    def loss_fn2():
      w = self.weights
      tv_noise = [tf.reduce_mean(tf.image.total_variation(x)) for x in shift]
      tv_noise = tf.add_n(tv_noise) / 4
      tv_max = tf.nn.relu(tv_noise - 1000) ** 2
      tv_min = tf.nn.relu(200 - tv_noise) ** 2
      tv_noise += tv_max + tv_min
      loss = l1_image * w[0] + tv_noise * w[2]
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(self.learning_rate, 0.9)
        op = op.minimize(loss, self.global_steps, var_list=var_shift)
        self.loss.append(op)
      self.train_metric['mae'] = l1_image
      self.train_metric['tv'] = tv_noise

    with tf.name_scope('Loss'):
      if not self.auto:
        loss_fn1()
      else:
        loss_fn2()
      self.metrics['psnr'] = tf.reduce_mean(
        tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
      self.metrics['ssim'] = tf.reduce_mean(
        tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

  def build_loss(self):
    pass

  def build_summary(self):
    super(DRSR, self).build_summary()
    # tf.summary.image('lr/input', self.inputs[-1])
    tf.summary.image(f'hr/fine_1', clip_image(self.outputs[-1]))
    tf.summary.image('hr/label', clip_image(self.label[0]))

  def build_saver(self):
    var_misc = tf.global_variables('Loss') + [self.global_steps]
    self.savers.update(misc=tf.train.Saver(var_misc, max_to_keep=1))
    var_g = tf.global_variables(self.name + f"/Level1")
    self.savers.update({
      f"level_1": tf.train.Saver(var_g, max_to_keep=1)
    })
    if self.auto:
      self.savers.update(shift=tf.train.Saver(
        tf.global_variables('Offset'),
        max_to_keep=1))
