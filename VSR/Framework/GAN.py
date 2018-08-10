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


def Discriminator(net,
                  input_shape=None,
                  filters=64,
                  depth=3,
                  scope='Critic', use_bias=False):
    """A simple D-net, for image generation usage
    
      Args:
          net: your base class of the caller to this method
          input_shape: identify the shape of the image if the dense layer is used.
                       if the input_shape is None, the dense layer is replaced by
                       global average pooling layer
          filters: the filter number in the 1st layer
          depth: layers = (depth + 1) * 2
          scope: name of the scope
          use_bias: use bias in convolution

      Return:
          the **callable** which returns the prediction and feature maps of each layer
    """

    def critic(inputs):
        assert isinstance(net, SuperResolution)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if input_shape[1] and input_shape[2]:
                inputs = tf.reshape(inputs, input_shape)
            F = filters
            N = [net.conv2d(inputs, F, 3, activation='lrelu',
                            use_bias=use_bias,
                            kernel_initializer='he_normal')]
            for _ in range(depth):
                N.append(net.conv2d(N[-1], F, 4, strides=2,
                                    activation='lrelu',
                                    use_batchnorm=True, use_bias=use_bias,
                                    kernel_initializer='he_normal'))
                F *= 2
                N.append(net.conv2d(N[-1], F, 4, activation='lrelu',
                                    use_batchnorm=True, use_bias=use_bias,
                                    kernel_initializer='he_normal'))
            N.append(net.conv2d(N[-1], F, 4, strides=2,
                                activation='lrelu',
                                use_batchnorm=True, use_bias=use_bias,
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


def DiscAE(net,
           input_shape=None,
           filters=64,
           depth=3,
           scope='Critic', use_bias=False):
    raise DeprecationWarning("This is not right")
    assert isinstance(net, SuperResolution)

    def critic(inputs):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            F = filters
            N = [net.conv2d(inputs, F, 3, activation='lrelu',
                            use_bias=use_bias,
                            kernel_initializer='he_normal')]
            for _ in range(depth):
                N.append(net.conv2d(N[-1], F, 4, strides=2,
                                    activation='lrelu',
                                    use_batchnorm=True, use_bias=use_bias,
                                    kernel_initializer='he_normal'))
                F *= 2
                N.append(net.conv2d(N[-1], F, 4, activation='lrelu',
                                    use_batchnorm=True, use_bias=use_bias,
                                    kernel_initializer='he_normal'))
            M = [net.conv2d(N[-1], F, 4, activation='lrelu',
                            use_batchnorm=True, use_bias=use_bias,
                            kernel_initializer='he_normal')]
            for i in range(depth):
                F //= 2
                M.append(net.conv2d(M[-1], F, 4, activation='lrelu',
                                    use_batchnorm=True, use_bias=use_bias,
                                    kernel_initializer='he_normal'))
                x = tf.concat([N[-i * 2 - 2], M[-1]], axis=-1)
                M.append(net.deconv2d(x, F, 4, strides=2,
                                      activation='lrelu',
                                      use_batchnorm=True, use_bias=use_bias,
                                      kernel_initializer='he_normal'))
            x = net.conv2d(M[-1], net.channel, 3, use_bias=use_bias,
                           kernel_initializer='he_normal')
            return x

    return critic


def loss_bce_gan(y_real, y_fake):
    """Original GAN loss with BCE"""

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
