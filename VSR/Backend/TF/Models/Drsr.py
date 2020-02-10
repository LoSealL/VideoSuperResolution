#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 1 - 11
#  Degradation-restore Super-resolution Network

import numpy as np
import tensorflow as tf

from VSR.Util import Config
from ..Framework import Noise, Trainer
from ..Framework.SuperResolution import SuperResolution
from ..Util import summary_tensor_image

_MEAN_GT = [84.1148, 68.3644, 64.8452]
_MEAN_SR = [85.6586, 68.7887, 66.5135]
_MEAN = np.array(_MEAN_SR, 'float32')


def _denormalize(inputs):
  return (inputs + 0) * 255


def _normalize(inputs):
  return inputs / 255


def _clip(image):
  return tf.cast(tf.clip_by_value(image, 0, 255), 'uint8')


class DRSR(SuperResolution):
  def __init__(self, name='drsr', n_cb=4, n_crb=4,
               noise_config=None,
               weights=(1, 0.5, 0.05, 1e-3),
               finetune=2000,
               mean_shift=False,
               **kwargs):
    super(DRSR, self).__init__(**kwargs)
    self.name = name
    self.n_cb = n_cb
    self.n_crb = n_crb
    self.weights = weights
    self.finetune = finetune
    self.mean_shift = mean_shift
    self.noise = Config(scale=0, offset=0, penalty=0.7, max=0.2, layers=7)
    if isinstance(noise_config, (dict, Config)):
      self.noise.update(**noise_config)
      if self.noise.type == 'crf':
        self.noise.crf = np.load(self.noise.crf)
      self.noise.offset /= 255
      self.noise.max /= 255
    if 'tfrecords' in kwargs:
      self.tfr = kwargs['tfrecords']
      self._trainer = DrTrainer

  def display(self):
    # stats = tf.profiler.profile()
    # tf.logging.info("Total parameters: {}".format(stats.total_parameters))
    tf.logging.info("Noisy scaling {}, bias sigma {}".format(
      self.noise.scale, self.noise.offset))
    tf.logging.info("Using {}".format(self.trainer))

  def _dncnn(self, inputs):
    n = self.noise
    with tf.variable_scope('Dncnn'):
      x = inputs
      for _ in range(6):
        x = self.bn_relu_conv2d(x, 64, 3)
      x = self.conv2d(x, self.channel, 3)
    return x

  def cascade_block(self, inputs, noise, filters=64, depth=4, scope=None,
                    reuse=None):
    def _noise_condition(nc_inputs, layers=2):
      with tf.variable_scope(None, 'NCL'):
        t = noise
        for _ in range(layers - 1):
          t = self.relu_conv2d(t, 64, 3)
        t = self.conv2d(t, 64, 3)
        gamma = tf.reduce_mean(t, [1, 2], keepdims=True)
        t = noise
        for _ in range(layers - 1):
          t = self.relu_conv2d(t, 64, 3)
        beta = self.conv2d(t, 64, 3)
      return nc_inputs * gamma + beta

    def _cond_resblock(cr_inputs, kernel_size):
      with tf.variable_scope(None, 'CRB'):
        pre_inputs = cr_inputs
        cr_inputs = self.relu_conv2d(cr_inputs, filters, kernel_size)
        cr_inputs = _noise_condition(cr_inputs)
        cr_inputs = self.relu_conv2d(cr_inputs, filters, kernel_size)
        cr_inputs = _noise_condition(cr_inputs)
        return pre_inputs + cr_inputs

    with tf.variable_scope(scope, 'CB', reuse=reuse):
      feat = [inputs]
      for i in range(depth):
        x = _cond_resblock(inputs, 3)
        feat.append(x)
        inputs = self.conv2d(tf.concat(feat, axis=-1), filters, 1,
                             kernel_initializer='he_uniform')
      # inputs = self.conv2d(inputs, filters, 3)
      return inputs

  def _upsample(self, inputs, noise):
    x = [self.conv2d(inputs, 64, 7)]
    for i in range(self.n_cb):
      x += [self.cascade_block(x[i], noise, depth=self.n_crb)]
    # bottleneck
    df = [self.conv2d(n, 32, 1, kernel_initializer='he_uniform')
          for n in x[:-1]]
    df.append(x[-1])
    summary_tensor_image(x[-1], 'last_before_bn')
    bottleneck = tf.concat(df, axis=-1, name='bottleneck')
    sr = self.upscale(bottleneck, direct_output=False)
    summary_tensor_image(sr, 'after_bn')
    sr = self.conv2d(sr, self.channel, 3)
    return sr, x

  def _unet(self, inputs, noise):
    with tf.variable_scope('Unet'):
      x0 = self.conv2d(inputs, 64, 7)
      x1 = self.cascade_block(x0, noise, depth=self.n_crb)
      x1s = tf.layers.average_pooling2d(x1, 2, 2)
      n1s = tf.layers.average_pooling2d(noise, 2, 2)
      x2 = self.cascade_block(x1s, n1s, depth=self.n_crb)
      x2s = tf.layers.average_pooling2d(x2, 2, 2)
      n2s = tf.layers.average_pooling2d(noise, 4, 4)
      x3 = self.cascade_block(x2s, n2s, depth=self.n_crb)
      x3u = self.deconv2d(x3, 64, 3, strides=2)
      x3u1 = tf.concat([x3u, x1s], -1)
      x3u2 = self.conv2d(x3u1, 64, 3)
      x4 = self.cascade_block(x3u2, n1s, depth=self.n_crb)
      x4u = self.deconv2d(x4, 64, 3, strides=2)
      x4u1 = tf.concat([x4u, x0], -1)
      x4u2 = self.conv2d(x4u1, 64, 3)
      x5 = self.conv2d(x4u2, self.channel, 3)
    return x5, None

  def _get_noise(self, inputs):
    n = self.noise
    if n.type == 'gaussian':
      sigma = tf.random_uniform([], maxval=n.max)
      noise = tf.random_normal(tf.shape(inputs), stddev=sigma)
      img = inputs + noise
      return img, noise
    elif n.type == 'crf':
      crf = tf.convert_to_tensor(n.crf['crf'])
      icrf = tf.convert_to_tensor(n.crf['icrf'])
      i = tf.random_uniform([], 0, crf.shape[0], dtype=tf.int32)
      irr = Noise.tf_camera_response_function(inputs, icrf[i], max_val=1)
      noise = Noise.tf_gaussian_poisson_noise(irr, max_c=n.max)
      img = Noise.tf_camera_response_function(irr + noise, crf[i], max_val=1)
      return img, img - inputs
    else:
      raise TypeError(n.type)

  def build_graph(self):
    super(DRSR, self).build_graph()
    inputs_norm = _normalize(self.inputs_preproc[-1])
    labels_norm = _normalize(self.label[-1])
    if self.mean_shift:
      inputs_norm -= _MEAN / 255
      labels_norm -= _MEAN / 255
    n = self.noise
    inputs_noise, noise = self._get_noise(inputs_norm)
    nn = self._upsample
    with tf.variable_scope('Offset'):
      x = inputs_norm
      for _ in range(n.layers):
        x = self.relu_conv2d(x, 64, 3,
                             kernel_initializer=tf.initializers.random_normal(
                               stddev=0.01))
      offset = self.conv2d(x, self.channel, 3,
                           kernel_initializer=tf.initializers.random_normal(
                             stddev=0.01))
      offset *= Noise.tf_gaussian_noise(offset, n.offset2)

    with tf.variable_scope(self.name):
      zero = self._dncnn(inputs_norm)
      zero_shift = zero + offset * n.scale + \
                   Noise.tf_gaussian_noise(zero, n.offset)
      clean = nn(inputs_norm, zero_shift)
    with tf.variable_scope(self.name, reuse=True):
      noisy = self._dncnn(inputs_noise)
      dirty = nn(inputs_noise, noisy)
    if self.finetune == -1:
      with tf.variable_scope(self.name, reuse=True):
        s = 2
        inputs_s2 = tf.layers.average_pooling2d(inputs_norm, s, s)
        zero_s2 = self._dncnn(inputs_s2)
        zero_shift_s2 = zero_s2 + Noise.tf_gaussian_noise(zero_s2, n.offset)
        clean_s2 = nn(inputs_s2, zero_shift_s2)
        noise_s2 = inputs_norm - clean_s2[0]
      with tf.variable_scope('Fine'):
        x = self.conv2d(inputs_norm, 64, 3)
        x = self.cascade_block(x, noise_s2, depth=6)
        x = self.conv2d(x, self.channel, 3)
        clean_fine = [x, x]
      self.outputs.append(_denormalize(clean_s2[0]))
      self.outputs.append(_denormalize(clean_fine[0]))
    else:
      self.outputs.append(_denormalize(tf.abs(zero)))
      self.outputs.append(_denormalize(clean[0]))

    if self.mean_shift:
      self.outputs = [x + _MEAN for x in self.outputs]

    def loss1():
      l1_with_noise = tf.losses.absolute_difference(dirty[0], labels_norm)
      l1_fine_tune = tf.losses.absolute_difference(clean[0], labels_norm)
      penalty = tf.clip_by_value(2 * tf.ceil(tf.nn.relu(noisy - noise)),
                                 0, 1)
      penalty = tf.abs(self.noise.penalty - penalty)
      noise_identity = penalty * tf.squared_difference(noisy, noise)
      noise_identity = tf.reduce_mean(noise_identity)
      noise_tv = tf.reduce_mean(tf.image.total_variation(noisy))
      # tv clamp
      l_tv_max = tf.nn.relu(noise_tv - 1000) ** 2
      l_tv_min = tf.nn.relu(100 - noise_tv) ** 2
      noise_tv += l_tv_max + l_tv_min
      loss = tf.stack([l1_with_noise, noise_identity, noise_tv])
      loss *= self.weights[:-1]
      loss = tf.reduce_sum(loss)
      self.train_metric['l1/noisy'] = l1_with_noise
      self.train_metric['l1/finet'] = l1_fine_tune
      self.train_metric['ni'] = noise_identity
      self.train_metric['nt'] = noise_tv

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      var_g = tf.trainable_variables(self.name)
      var_o = tf.trainable_variables('Offset')
      with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(self.learning_rate, 0.9)
        op = op.minimize(loss, self.global_steps, var_list=var_g)
        self.loss.append(op)
        op = tf.train.AdamOptimizer(self.learning_rate, 0.9)
        op = op.minimize(l1_fine_tune, self.global_steps, var_list=var_o)
        self.loss.append(op)

    def loss2():
      l1_clean = tf.losses.mean_squared_error(clean[0], labels_norm)
      var_g = tf.trainable_variables(self.name)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(self.learning_rate, 0.9)
        op = op.minimize(l1_clean, self.global_steps, var_list=var_g)
        self.loss += [op, op]
      self.train_metric['l1/tune'] = l1_clean

    def loss3():
      l1_clean = tf.losses.mean_squared_error(clean_fine[0], labels_norm)
      var_f = tf.trainable_variables('Fine')
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(self.learning_rate, 0.9)
        op = op.minimize(l1_clean, self.global_steps, var_list=var_f)
        self.loss += [op, op]
      self.train_metric['l1/tune'] = l1_clean
      tf.summary.image('hr/coarse', _clip(self.outputs[-2]))

    with tf.name_scope('Loss'):
      if self.finetune == -1:
        loss3()
      elif 'DrTrainer' in str(self.trainer):
        loss2()
      else:
        loss1()
    self.metrics['psnr1'] = tf.reduce_mean(
      tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
    tf.summary.image('noisy/zero', zero)

  def build_loss(self):
    pass

  def build_summary(self):
    super(DRSR, self).build_summary()
    tf.summary.image('lr/input', self.inputs[-1])
    tf.summary.image('hr/fine', _clip(self.outputs[-1]))
    tf.summary.image('hr/label', _clip(self.label[0]))

  def build_saver(self):
    var_g = tf.global_variables(self.name)
    steps = [self.global_steps]
    loss = tf.global_variables('Loss')
    self.savers.update(drsr_g=tf.train.Saver(var_g, max_to_keep=1),
                       misc=tf.train.Saver(steps + loss, max_to_keep=1))
    if self.finetune == -1:
      var_f = tf.global_variables('Fine')
      self.savers.update(drsr_f=tf.train.Saver(var_f, max_to_keep=1))

  def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
    epochs = kwargs.get('epochs')
    if epochs < self.finetune:
      loss = self.loss[0]
    else:
      loss = self.loss[1]
    return super(DRSR, self).train_batch(feature, label, learning_rate,
                                         loss=loss)


class DrTrainer(Trainer.VSR):
  def fn_train_each_step(self, label=None, feature=None, name=None,
                         post=None):
    if not self.model.tfr:
      return super(DrTrainer, self).fn_train_each_step(
        label, feature, name, post)
    v = self.v
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
      post = fn(post, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    loss = self._m.train_batch(post, label, learning_rate=v.lr,
                               epochs=v.epoch)
    v.global_step = self._m.global_steps.eval()
    for _k, _v in loss.items():
      v.avg_meas[_k] = \
        v.avg_meas[_k] + [_v] if v.avg_meas.get(_k) else [_v]
      loss[_k] = '{:08.5f}'.format(_v)
    v.loss = loss

  def fn_benchmark_each_step(self, label=None, feature=None, name=None,
                             post=None):
    if not self.model.tfr:
      return super(DrTrainer, self).fn_benchmark_each_step(
        label, feature, name, post)
    v = self.v
    origin_feat = feature
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
      # post = fn(post, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    outputs, metrics = self._m.test_batch(post, label, epochs=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    for fn in v.output_callbacks:
      outputs = fn(outputs, input=origin_feat, label=label, name=name,
                   mode=v.color_format, subdir=v.subdir)
