"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 17th 2018

Architectures of common residual blocks used in SR researches
"""
import tensorflow as tf

from ..Framework.LayersHelper import Layers
from ..Util.Utility import to_list


def rdn(layers: Layers, inputs, depth, scaling=1.0,
        scope=None, reuse=None, **kwargs):
  """Residual Dense Block (CVPR18)"""

  k = kwargs.pop('kernel_size', 3)
  f = kwargs.pop('filters', 64)
  act = kwargs.pop('activation', 'relu')
  kwargs.pop('name', None)
  with tf.variable_scope(scope, 'RDN', reuse=reuse):
    fl = [inputs]
    for i in range(depth - 1):
      x = layers.conv2d(tf.concat(fl, -1), f, k, activation=act, **kwargs)
      fl.append(x)
    x = layers.conv2d(tf.concat(fl, -1), f, k, **kwargs)
    return x * scaling + inputs


def rcab(layers: Layers, inputs, ratio=16, scope=None, reuse=None, **kwargs):
  """Residual channel attention block (ECCV18)

  """

  k = kwargs.pop('kernel_size', 3)
  f = kwargs.pop('filters', 64)
  act = kwargs.pop('activation', 'relu')
  with tf.variable_scope(scope, 'RCAB', reuse=reuse):
    x = layers.conv2d(inputs, f, k, activation=act, **kwargs)
    y = layers.conv2d(x, f, k, **kwargs)
    x = tf.reduce_mean(y, axis=[1, 2], keepdims=True)
    x = layers.conv2d(x, f // ratio, 1, activation=act, **kwargs)
    x = layers.conv2d(x, f, 1, activation=tf.nn.sigmoid, **kwargs)
    y *= x
  return y + inputs


def msrb(layers: Layers, inputs, scope=None, reuse=None, **kwargs):
  """Multi-scale residual block (ECCV18)

  """

  k1, k2 = kwargs.pop('kernel_size', (3, 5))
  f = kwargs.pop('filters', 64)
  act = kwargs.pop('activation', 'relu')
  with tf.variable_scope(scope, 'MSRB', reuse=reuse):
    s1 = layers.conv2d(inputs, f, k1, activation=act)
    p1 = layers.conv2d(inputs, f, k2, activation=act)

    s2 = layers.conv2d(tf.concat([s1, p1], -1), f * 2, k1, activation=act)
    p2 = layers.conv2d(tf.concat([p1, s1], -1), f * 2, k2, activation=act)

    s = layers.conv2d(tf.concat([s2, p2], -1), f, 1)
  return s + inputs


def cascade_block(layers: Layers, inputs, depth=4,
                  scope=None, reuse=None, **kwargs):
  """Cascading residual block (ECCV18)

  """

  k = kwargs.pop('kernel_size', 3)
  f = kwargs.pop('filters', 64)
  act = kwargs.pop('activation', 'relu')
  with tf.variable_scope(scope, 'CARB', reuse=reuse):
    feat = [inputs]
    for i in range(depth):
      x = layers.resblock(inputs, f, k, activation=act)
      feat.append(x)
      inputs = layers.conv2d(
        tf.concat(feat, axis=-1), f, 1,
        kernel_initializer='he_uniform')
    inputs = layers.conv2d(inputs, f, k)
    return inputs


def cascade_rdn(layers: Layers, inputs, depth, use_ca=False,
                scope=None, reuse=None, **kwargs):
  """Cascaded residual dense block.
  Args:
    layers: child class of Layers
    inputs: input tensor
    depth: an int or list of 2 ints, representing number of rdbs
    use_ca: insert channel attention layer
    scope: scope name
    reuse: reuse variables
  """

  k = kwargs.pop('kernel_size', 3)
  f = kwargs.pop('filters', 64)
  act = kwargs.pop('activation', 'relu')
  kwargs.pop('name', None)
  depth = to_list(depth, 2)
  with tf.variable_scope(scope, 'CascadeRDN', reuse=reuse):
    fl = [inputs]
    x = inputs
    for i in range(depth[0]):
      x = rdn(layers, x, depth[1], kernel_size=k, filters=f, activation=act,
              **kwargs)
      if use_ca:
        x = rcab(layers, x, kernel_size=k, filters=f, activation=act, **kwargs)
      fl.append(x)
      x = tf.concat(fl, -1)
      x = layers.conv2d(x, f, 1, **kwargs)
    return x


def non_local(layers: Layers, inputs, filters=64, func=None, scaling=1,
              pooling=None, use_bn=None, scope=None, reuse=None):
  """Non-local neural networks (CVPR 18)

  Note:
      y_i = \frac{1}{C(x)}\Sigma_{j}(f(x_i, x_j)g(x_j))
      A pairwise function f computes a scalar between i and all j. The unary
      function g computes a representation of the input signal at the position
      j. The response is normalized by a factor C(x).

  Args:
      layers: parent object.
      inputs: input tensor.
      filters: output filter numbers.
      func: a string in ('gaussian', 'embedded', 'dot', 'concat'),
        representing the pairwise function f. Default 'embedded'.
      scaling: scaling channel numbers down.
      pooling: sub-sampling x by max-pooling, `pooling` represents strides.
      use_bn: if True, batch-normalizing last embedding output.
      scope: variable scope for this block.
      reuse: reuse flag.
  """

  def embedding(x: tf.Tensor, c, scale):
    if len(x.shape) == 4:
      return layers.conv2d(x, c // scale, 1)
    elif len(x.shape) == 5:
      return layers.conv3d(x, c // scale, 1)
    else:
      raise ValueError('input tensor dimensions must be 4 or 5.')

  def flatten(x: tf.Tensor):
    c = x.shape[-1]
    b = tf.shape(x)[0]
    return tf.reshape(x, [b, -1, c])

  def reduce(x: tf.Tensor):
    if len(x.shape) == 4:
      if pooling is not None:
        return tf.layers.max_pooling2d(x, pooling, pooling)
      else:
        return x
    elif len(x.shape) == 5:
      if pooling is not None:
        return tf.layers.max_pooling3d(x,
                                       [1, pooling, pooling],
                                       [1, pooling, pooling])
      else:
        return x
    else:
      raise ValueError('input tensor dimensions must be 4 or 5.')

  def gaussian(x: tf.Tensor):
    x0 = flatten(x)
    x1 = flatten(reduce(x))
    return tf.nn.softmax(tf.matmul(x0, x1, transpose_b=True), -1)

  def embedded(x: tf.Tensor):
    theta = flatten(embedding(x, filters, scaling))
    phi = flatten(embedding(reduce(x), filters, scaling))
    return tf.nn.softmax(tf.matmul(theta, phi, transpose_b=True), -1)

  def dot(x: tf.Tensor):
    theta = flatten(embedding(x, filters, scaling))
    phi = flatten(embedding(reduce(x), filters, scaling))
    n = tf.shape(phi)[1]
    return tf.truediv(tf.matmul(theta, phi, transpose_b=True), n)

  def concat(x: tf.Tensor):
    raise NotImplementedError

  with tf.variable_scope(scope, 'NonLocal', reuse=reuse):
    channels = inputs.get_shape().as_list()[-1]
    shape = tf.shape(inputs)
    g = embedding(inputs, filters, scaling)
    g = reduce(g)
    if func == 'gaussian':
      f = gaussian
    elif func == 'dot':
      f = dot
    elif func == 'concat':
      f = concat
    else:
      f = embedded
    corr = f(inputs)
    y = tf.matmul(corr, g)
    y = tf.reshape(y, [shape[0], shape[1], shape[2], -1])
    y = embedding(y, channels, 1)
    if use_bn:
      y = layers.batch_norm(y, layers.training_phase)
    return inputs + y
