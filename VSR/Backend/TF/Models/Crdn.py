"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Mar. 20th 2019

Cascaded Residual Dense Network (NTIRE 2019)
"""

from .. import tf
from ..Arch.Residual import cascade_rdn
from ..Framework.SuperResolution import SuperResolution
from ..Util import clip_image


def _denormalize(inputs):
  return (inputs + 0) * 255


def _normalize(inputs):
  return (inputs - 0) / 255


class CRDN(SuperResolution):
  """A Cascaded Residual Dense Network"""

  def __init__(self, name='crdn', **kwargs):
    super(CRDN, self).__init__(**kwargs)
    self.name = name
    self.n_cb = 6

  def upsample(self, inputs, skips, method='nearest', scale=2):
    with tf.variable_scope(None, f'UpX{scale}'):
      c = int(inputs.shape[-1])
      up0 = self.upscale(inputs, method, scale, direct_output=False)
      up1 = self.conv2d(up0, c, 3)
      fs0 = tf.concat([up1, skips], axis=-1)
      fs1 = self.conv2d(fs0, c // 2, 3)
      return fs1

  def build_graph(self):
    super(CRDN, self).build_graph()
    depth = self.n_cb
    filters = 32
    inputs = _normalize(self.inputs_preproc[-1])
    with tf.variable_scope(self.name):
      x = self.conv2d(inputs, filters, 7)
      entry = self.conv2d(x, filters, 5)
      x_list = [entry]
      f = filters
      for i in range(depth // 2):
        x = cascade_rdn(self, x, depth=3, use_ca=True, filters=f)
        x_list.append(x)
        f *= 2
        x = self.conv2d(x, f, 3, strides=2, name='Down%d' % i)
      x = cascade_rdn(self, x, depth=3, use_ca=True, filters=f)
      x = cascade_rdn(self, x, depth=3, use_ca=True, filters=f)
      for i in range(depth // 2):
        f //= 2
        x = self.upsample(x, x_list.pop())
        x = cascade_rdn(self, x, depth=3, use_ca=True, filters=f)

      assert len(x_list) == 1, f'length of x_list is not 1: {len(x_list)}'
      assert f == filters
      x += x_list.pop()
      sr = self.conv2d(x, filters, 3)
      sr = self.conv2d(sr, self.channel, 3)

    self.outputs.append(_denormalize(sr))

  def build_loss(self):
    with tf.name_scope('Loss'):
      l1 = tf.losses.absolute_difference(self.label[-1], self.outputs[-1])
      op = tf.train.AdamOptimizer(self.learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        opt = op.minimize(l1, self.global_steps)
        self.loss.append(opt)

      self.train_metric['l1'] = l1
      self.metrics['psnr'] = tf.reduce_mean(
          tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
      self.metrics['ssim'] = tf.reduce_mean(
          tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))

  def build_summary(self):
    super(CRDN, self).build_summary()
    tf.summary.image('sr', clip_image(self.outputs[-1]))
    tf.summary.image('lr', clip_image(self.inputs_preproc[-1]))
    tf.summary.image('hq', clip_image(self.label[-1]))

  def build_saver(self):
    var_g = tf.global_variables(self.name)
    misc = tf.global_variables('Loss') + [self.global_steps]
    self.savers['misc'] = tf.train.Saver(misc, max_to_keep=1)
    self.savers[self.name] = tf.train.Saver(var_g, max_to_keep=1)
