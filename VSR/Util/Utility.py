"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 8th 2018

utility functions
"""
from typing import Generator
import tensorflow as tf
import numpy as np


def to_list(x, repeat=1):
    """convert x to list object

    Args:
         x: any object to convert
         repeat: if x is to make as [x], repeat `repeat` elements in the list
    """
    if isinstance(x, (Generator, tuple, set)):
        return list(x)
    elif isinstance(x, list):
        return x
    elif isinstance(x, dict):
        return list(x.values())
    elif x is not None:
        return [x] * repeat
    else:
        return []


def shrink_mod_scale(x, scale):
    """clip each dim of x to multiple of scale

    """
    scale = to_list(scale, 2)
    mod_x = []
    for _x, _s in zip(x, scale):
        mod_x.append(_x - _x % _s)
    return mod_x


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    """
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)


def pixel_shift(t, scale, c):
    with tf.name_scope('SubPixel'):
        r = to_list(scale, 2)
        shape = tf.shape(t)
        H, W = shape[1], shape[2]
        C = c
        t = tf.reshape(t, [-1, H, W, *r, C])
        # Here we are different from Equation 4 from the paper. That equation
        # is equivalent to switching 3 and 4 in `perm`. But I feel my
        # implementation is more natural.
        t = tf.transpose(t, perm=[0, 1, 3, 2, 4, 5])  # B, H, r, W, r, C
        t = tf.reshape(t, [-1, H * r[1], W * r[0], C])
        return t


def bicubic_rescale(img, scale):
    with tf.name_scope('Bicubic'):
        shape = tf.shape(img)
        scale = to_list(scale, 2)
        shape_enlarge = tf.cast(shape, tf.float32) * [1, *scale, 1]
        shape_enlarge = tf.cast(shape_enlarge, tf.int32)
        return tf.image.resize_bicubic(img, shape_enlarge[1:3])


def prelu(x, name=None):
    with tf.name_scope('PReLU'):
        alphas = tf.Variable(tf.constant(0.1, shape=[x.shape[-1]]), name=name)
        return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5


def guassian_kernel(kernel_size, width):
    """generate a guassian kernel"""

    kernel_size = np.asarray(to_list(kernel_size, 2), np.int32)
    kernel_size = kernel_size - kernel_size % 2
    half_ksize = kernel_size // 2
    x, y = np.mgrid[-half_ksize[0]:half_ksize[0] + 1, -half_ksize[1]:half_ksize[1] + 1]
    return np.exp(-(x ** 2 + y ** 2) / 2 * width) / (2 * np.pi * width ** 2)


def wgan_gp(y_true, y_pred, discriminator, lamb=10):
    """W-GAN Gradient penalty"""

    if not callable(discriminator):
        raise TypeError('Discriminator is not a callable!')
    diff = y_pred - y_true
    alpha = tf.random_uniform(tf.shape(diff), minval=0., maxval=1.)
    interp = y_true + alpha * diff
    gradients = tf.gradients(discriminator(interp), [interp])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), reduction_indices=[1]))
    gp = tf.reduce_mean((slopes - 1.) ** 2.)
    return lamb * gp


class Vgg:
    """use pre-trained VGG network from keras.application.vgg16
    to obtain outputs of specific layers
    """

    def __init__(self, include_top=False, input_shape=None, type='vgg16'):
        with tf.variable_scope('VGG'):
            if np.size(input_shape) > 3:
                input_shape = input_shape[-3:]
            elif np.size(input_shape) < 3:
                raise ValueError('input shape must be [H, W, 3]')
            if type == 'vgg16':
                self._m = tf.keras.applications.vgg16.VGG16(
                    include_top=include_top, input_shape=input_shape)
            elif type == 'vgg19':
                self._m = tf.keras.applications.vgg19.VGG19(
                    include_top=include_top, input_shape=input_shape)
            self._vgg_mean = [103.939, 116.779, 123.68]
            self.include_top = include_top

    def __call__(self, x, *args, **kwargs):
        return self.call(x, *args, **kwargs)

    def call(self, x, conv, block, yuv_to_rgb_convert=False):
        """get the output of a pre-trained VGG16 network

            Args:
                conv: the convolution layer in block, start by 0 in each block
                block: the block number, range from [1, 5]
                yuv_to_rgb_convert: need to convert from yuv to rgb

            Return:
                the output of given layer

            Note:
                if `conv` and `block` is lists, return a list of outputs
            """
        with tf.variable_scope('VGG'):
            x = self._normalize(x, yuv_to_rgb_convert)
            block = to_list(block)
            conv = to_list(conv)
            outputs = []
            for b, c in zip(block, conv):
                layer_name = f'block{b}_conv{c}'
                layer = self._m.get_layer(layer_name)
                outputs.append(layer.output)
            m = tf.keras.Model(self._m.input, outputs, name='VGG')
            return m(x)

    def _normalize(self, x, yuv_to_rgb_convert):
        if yuv_to_rgb_convert:
            x = tf.image.yuv_to_rgb(x)
        if x.shape[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)
        if self.include_top:
            x = tf.image.resize_bicubic(x, (224, 224))
        # RGB->BGR
        x = tf.cast(x, tf.float32)
        x = x[..., ::-1] - self._vgg_mean
        return x


class ConvolutionDeltaOrthogonal(tf.keras.initializers.Initializer):
    """Initializer that generates a delta orthogonal kernel for ConvNets.

    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero.

    Args:
      gain: multiplicative factor to apply to the orthogonal matrix. Default is 1.
        The 2-norm of an input is multiplied by a factor of 'sqrt(gain)' after
        applying this convolution.
      dtype: The type of the output.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
    """

    def __init__(self, gain=1.0, seed=None, dtype=tf.float32):
        self.gain = gain
        self.dtype = dtype
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        # Check the shape
        if len(shape) < 3 or len(shape) > 5:
            raise ValueError("The tensor to initialize must be at least "
                             "three-dimensional and at most five-dimensional")

        if shape[-2] > shape[-1]:
            raise ValueError("In_filters cannot be greater than out_filters.")

        # Generate a random matrix
        a = tf.random_normal([shape[-1], shape[-1]],
                             dtype=dtype, seed=self.seed)
        # Compute the qr factorization
        q, r = tf.qr(a, full_matrices=False)
        # Make Q uniform
        d = tf.diag_part(r)
        # ph = D / math_ops.abs(D)
        q *= tf.sign(d)
        q = q[:shape[-2], :]
        q *= tf.sqrt(tf.cast(self.gain, dtype=dtype))
        if len(shape) == 3:
            weight = tf.scatter_nd([[(shape[0] - 1) // 2]],
                                   tf.expand_dims(q, 0), shape)
        elif len(shape) == 4:
            weight = tf.scatter_nd([[(shape[0] - 1) // 2, (shape[1] - 1) // 2]],
                                   tf.expand_dims(q, 0), shape)
        else:
            weight = tf.scatter_nd([[(shape[0] - 1) // 2, (shape[1] - 1) // 2,
                                     (shape[2] - 1) // 2]],
                                   tf.expand_dims(q, 0), shape)
        return weight

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}
