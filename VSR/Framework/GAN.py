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


def Discriminator(net, input_shape=None, scope='Critic', use_bias=False):
    """A simple D-net, for image generation usage
    
      Args:
          net: your base class of the caller to this method
          input_shape: identify the shape of the image if the dense layer is used.
                       if the input_shape is None, the dense layer is replaced by
                       global average pooling layer
          scope: name of the scope
          use_bias: use bias in convolution

      Return:
          the **callable** which returns the prediction and feature maps of each layer
    """

    def critic(inputs):
        assert isinstance(net, SuperResolution)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if input_shape is not None:
                x = tf.reshape(inputs, input_shape)
            else:
                x = inputs
            fet = []
            x = net.conv2d(x, 64, 3, activation='lrelu', use_batchnorm=False, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 64, 3, strides=2, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 128, 4, strides=1, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 128, 4, strides=2, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 256, 4, strides=1, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 256, 4, strides=2, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 512, 4, strides=1, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            x = net.conv2d(x, 512, 4, strides=2, activation='lrelu', use_batchnorm=True, use_bias=use_bias,
                           kernel_initializer='he_normal')
            fet.append(x)
            if input_shape:
                x = tf.layers.flatten(x)
                x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)
                x = tf.layers.dense(x, 1)
            else:
                x = net.conv2d(x, 1, 3)
                x = tf.reduce_mean(x, [1, 2, 3])
            return x, fet

    return critic


def loss_gan(y_true, y_pred, discriminator):
    """Original GAN loss with BCE"""

    if not callable(discriminator):
        raise TypeError('Discriminator is not a callable!')

    y_real = discriminator(y_true)
    y_fake = discriminator(y_pred)

    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real) + \
             tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake)
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake)
    return g_loss, d_loss


def loss_wgan(y_true, y_pred, discriminator):
    """W-GAN"""

    if not callable(discriminator):
        raise TypeError('Discriminator is not a callable!')

    y_real = discriminator(y_true)
    y_fake = discriminator(y_pred)

    d_loss = tf.reduce_mean(y_fake - y_real)
    g_loss = -tf.reduce_mean(y_fake)

    return g_loss, d_loss


def loss_wgan_gp(y_true, y_pred, discriminator, lamb=10):
    """W-GAN Gradient penalty"""

    g_loss, d_loss = loss_wgan(y_true, y_pred, discriminator)

    diff = y_pred - y_true
    alpha = tf.random_uniform(tf.shape(diff), minval=0., maxval=1.)
    interp = y_true + alpha * diff
    gradients = tf.gradients(discriminator(interp), [interp])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), reduction_indices=[1]))
    gp = tf.reduce_mean((slopes - 1.) ** 2.)
    return g_loss, d_loss + lamb * gp


def loss_lsgan(y_true, y_pred, discriminator):
    """LSGAN"""

    if not callable(discriminator):
        raise TypeError('Discriminator is not a callable!')

    y_real = discriminator(y_true)
    y_fake = discriminator(y_pred)
    d_loss = tf.reduce_mean((y_real - 1) ** 2) + tf.reduce_mean(y_fake ** 2)
    g_loss = tf.reduce_mean((y_fake - 1) ** 2)
    return g_loss, d_loss


def loss_relative_lsgan(y_true, y_pred, discriminator, average=False):
    """R(A)LSGAN"""

    if not callable(discriminator):
        raise TypeError('Discriminator is not a callable!')

    y_real = discriminator(y_true)
    y_fake = discriminator(y_pred)
    if average:
        d_loss = tf.reduce_mean((y_real - tf.reduce_mean(y_fake) - 1) ** 2) + \
                 tf.reduce_mean((y_fake - tf.reduce_mean(y_real) + 1) ** 2)

        g_loss = tf.reduce_mean((y_real - tf.reduce_mean(y_fake) + 1) ** 2) + \
                 tf.reduce_mean((y_fake - tf.reduce_mean(y_real) - 1) ** 2)
    else:
        d_loss = tf.reduce_mean((y_real - y_fake - 1) ** 2)
        g_loss = tf.reduce_mean((y_fake - y_real - 1) ** 2)
    return g_loss, d_loss
