#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/25 下午2:13

from .. import tf
from ..Framework.SuperResolution import SuperResolution


class FFDNet(SuperResolution):
  """FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising
  By Kai Zhang. (IEEE TIP 2018)

  Args:
    sigma: in training phase, this is the max sigma level added to clean images,
      in testing phase, this is input noise level, correspond to pixel [0, 255].
    space_down: block size for space-to-depth (default 2, same as paper's).
    layers: convolutional layers used in the network.
    training: set to false when evaluating.
  """

  def __init__(self, sigma, space_down=2, layers=15, training=True,
               name='ffdnet', **kwargs):
    self.name = name
    self.sigma = sigma
    self.space_down = space_down
    self.layers = layers
    self.training = training
    if 'scale' in kwargs:
      kwargs.pop('scale')
    super(FFDNet, self).__init__(scale=1, **kwargs)

  def build_graph(self):
    super(FFDNet, self).build_graph()  # build inputs placeholder
    with tf.variable_scope(self.name):
      # build layers
      inputs = self.inputs_preproc[-1] / 255
      if self.training:
        sigma = tf.random_uniform((), maxval=self.sigma / 255)
        inputs += tf.random_normal(tf.shape(inputs)) * sigma
      else:
        sigma = self.sigma / 255
      inputs = tf.space_to_depth(inputs, block_size=self.space_down)
      noise_map = tf.ones_like(inputs)[..., 0:1] * sigma
      x = tf.concat([inputs, noise_map], axis=-1)
      x = self.relu_conv2d(x, 64, 3)
      for i in range(1, self.layers - 1):
        x = self.bn_relu_conv2d(x, 64, 3, use_bias=False)
      # the last layer w/o BN and ReLU
      x = self.conv2d(x, self.channel * self.space_down ** 2, 3)
      denoised = tf.depth_to_space(x, block_size=self.space_down)
      self.outputs.append(denoised * 255)

  def build_loss(self):
    with tf.name_scope('loss'):
      mse, loss = super(FFDNet, self).build_loss()
      self.train_metric['loss'] = loss
      self.metrics['mse'] = mse
      self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(
        self.label[-1], self.outputs[-1], max_val=255))
      self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(
        self.label[-1], self.outputs[-1], max_val=255))
