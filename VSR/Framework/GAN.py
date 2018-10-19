"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: July 20th 2018

Conventional Generator and Discriminator as well as objective function
for generative adversarial networks 
"""

from .SuperResolution import SuperResolution
import tensorflow as tf
import numpy as np


def Discriminator(net,
                  input_shape=None,
                  filters=64,
                  depth=3,
                  use_bias=False,
                  use_bn=True,
                  use_sn=False,
                  scope='Critic'):
    """A simple D-net, for image generation usage
    
      Args:
          net: your base class of the caller to this method
          input_shape: identify the shape of the image if the dense layer is used.
                       if the input_shape is None, the dense layer is replaced by
                       global average pooling layer
          filters: the filter number in the 1st layer
          depth: layers = (depth + 1) * 2
          use_bias: use bias in convolution
          use_bn: use batch normalization
          use_sn: use spectral normalization
          scope: name of the scope

      Return:
          the **callable** which returns the prediction and feature maps of each layer
    """

    def critic(inputs):
        assert isinstance(net, SuperResolution)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if input_shape[1] and input_shape[2]:
                if input_shape[0] is None:
                    input_shape[0] = -1
                inputs = tf.reshape(inputs, input_shape)
            F = filters
            N = [net.conv2d(inputs, F, 3, activation='lrelu',
                            use_bias=use_bias, use_sn=use_sn,
                            kernel_initializer='he_normal')]
            for _ in range(depth):
                N.append(net.conv2d(N[-1], F, 4, strides=2, activation='lrelu',
                                    use_sn=use_sn,
                                    use_batchnorm=use_bn,
                                    use_bias=use_bias,
                                    kernel_initializer='he_normal'))
                F *= 2
                N.append(net.conv2d(N[-1], F, 4, activation='lrelu',
                                    use_batchnorm=use_bn,
                                    use_sn=use_sn,
                                    use_bias=use_bias,
                                    kernel_initializer='he_normal'))
            N.append(net.conv2d(N[-1], F, 4, strides=2, activation='lrelu',
                                use_batchnorm=True,
                                use_sn=use_sn,
                                use_bias=use_bias,
                                kernel_initializer='he_normal'))
            if input_shape[1] and input_shape[2]:
                x = tf.layers.flatten(N[-1])
                x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)
                x = tf.layers.dense(x, 1)
            else:
                x = net.conv2d(N[-1], 1, 3)
                x = tf.reduce_mean(x, [1, 2, 3])
            return x

    return critic


def ProjectDiscriminator(net,
                         input_shape=None,
                         use_sn=False,
                         scale=2,
                         scope='Critic'):
    def resblock(inputs, filters, kernel_size=3, activation=tf.nn.relu, downsample=False):
        h = inputs
        h = activation(h)
        h = net.conv2d(h, filters, kernel_size, kernel_initializer='he_normal', use_sn=use_sn)
        h = activation(h)
        h = net.conv2d(h, filters, kernel_size, kernel_initializer='he_normal', use_sn=use_sn)
        sc = net.conv2d(inputs, filters, 1, kernel_initializer='he_normal', use_sn=use_sn)
        h += sc
        if downsample:
            h = tf.layers.average_pooling2d(h, 2, 2)
        return h

    def critic(inputs, condition=None):
        assert isinstance(net, SuperResolution)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if input_shape[1] and input_shape[2]:
                if input_shape[0] is None:
                    input_shape[0] = -1
                inputs = tf.reshape(inputs, input_shape)
            x = inputs
            x = resblock(x, 64, downsample=True)
            x = resblock(x, 64)
            if scale == 2: t = x
            x = resblock(x, 128, downsample=True)
            x = resblock(x, 128)
            if scale == 4: t = x
            x = resblock(x, 256, downsample=True)
            x = resblock(x, 512, downsample=True)
            x = resblock(x, 1024, downsample=True)
            x = resblock(x, 1024)
            x = tf.nn.relu(x)
            if input_shape[1] and input_shape[2]:
                x = tf.layers.flatten(x)
                x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)
                x = tf.layers.dense(x, 1)
            else:
                x = net.conv2d(x, 1, 3)
                x = tf.reduce_mean(x, [1, 2, 3])
            if condition is not None:
                t = net.conv2d(t, net.channel, 3, kernel_initializer='he_normal', use_sn=use_sn)
                t = tf.layers.flatten(t)
                t = tf.matmul(t, tf.layers.flatten(condition), transpose_b=True)
                return x + t
            return x

    return critic


def loss_bce_gan(y_real, y_fake):
    """Original GAN loss with BCE"""

    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real) + \
             tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake)

    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake)
    return g_loss, d_loss


def loss_relative_bce_gan(y_real, y_fake, average=False):
    """R(A)GAN"""

    if average:
        d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real - tf.reduce_mean(y_fake)) + \
                 tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake - tf.reduce_mean(y_real))

        g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake - tf.reduce_mean(y_real)) + \
                 tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_real), y_real - tf.reduce_mean(y_fake))
    else:
        d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real - y_fake) + \
                 tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake - y_real)

        g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake - y_real)
    return g_loss, d_loss


def loss_wgan(y_real, y_fake):
    """W-GAN"""

    d_loss = tf.reduce_mean(y_fake - y_real)
    g_loss = -tf.reduce_mean(y_fake)

    return g_loss, d_loss


def gradient_penalty(y_true, y_pred, graph_fn, lamb=10):
    """Gradient penalty"""

    if not callable(graph_fn):
        raise TypeError('graph callee is not a callable!')

    diff = y_pred - y_true
    alpha = tf.random_uniform(tf.shape(diff), minval=0., maxval=1.)
    interp = y_true + alpha * diff
    gradients = tf.gradients(graph_fn(interp), [interp])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), reduction_indices=[1]))
    gp = tf.reduce_mean((slopes - 1.) ** 2.)
    return lamb * gp


def loss_lsgan(y_real, y_fake):
    """Least-Square GAN"""

    d_loss = tf.reduce_mean((y_real - 1) ** 2) + tf.reduce_mean(y_fake ** 2)
    g_loss = tf.reduce_mean((y_fake - 1) ** 2)
    return g_loss, d_loss


def loss_relative_lsgan(y_real, y_fake, average=False):
    """R(A)LSGAN"""

    if average:
        d_loss = tf.reduce_mean((y_real - tf.reduce_mean(y_fake) - 1) ** 2) + \
                 tf.reduce_mean((y_fake - tf.reduce_mean(y_real) + 1) ** 2)

        g_loss = tf.reduce_mean((y_real - tf.reduce_mean(y_fake) + 1) ** 2) + \
                 tf.reduce_mean((y_fake - tf.reduce_mean(y_real) - 1) ** 2)
    else:
        d_loss = tf.reduce_mean((y_real - y_fake - 1) ** 2)
        g_loss = tf.reduce_mean((y_fake - y_real - 1) ** 2)
    return g_loss, d_loss


def loss_sensitive_gan(y_real, y_fake):
    raise NotImplementedError
