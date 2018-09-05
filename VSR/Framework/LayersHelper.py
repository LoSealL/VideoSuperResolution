"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 5th 2018

commonly used layers helper
"""
import tensorflow as tf

from ..Util.Utility import (
    to_list, prelu, pixel_shift, SpectralNorm
)


class Layers(object):
    def conv2d(self, x,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='same',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               use_batchnorm=False,
               use_sn=False,
               kernel_initializer=None,
               kernel_regularizer=None,
               **kwargs):
        """wrap a convolution for common use case"""

        ki, kr = self._kernel(kernel_initializer, kernel_regularizer)
        nn = tf.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, data_format=data_format,
                              dilation_rate=dilation_rate, use_bias=use_bias,
                              kernel_initializer=ki, kernel_regularizer=kr, **kwargs)
        nn.build(x.shape.as_list())
        if use_sn:
            nn.kernel = SpectralNorm()(nn.kernel)
        x = nn(x)
        if use_batchnorm:
            x = tf.layers.batch_normalization(x, training=self.training_phase)
        activator = self._act(activation)
        if activation:
            x = activator(x)
        return x

    def deconv2d(self, x,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 use_batchnorm=False,
                 use_sn=False,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 **kwargs):
        """warp a conv2d_transpose op for simplicity usage"""

        ki, kr = self._kernel(kernel_initializer, kernel_regularizer)
        nn = tf.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                                       data_format=data_format, use_bias=use_bias,
                                       kernel_initializer=ki, kernel_regularizer=kr, **kwargs)
        nn.build(x.shape.as_list())
        if use_sn:
            nn.kernel = SpectralNorm()(nn.kernel)
        x = nn(x)
        if use_batchnorm:
            x = tf.layers.batch_normalization(x, training=self.training_phase)
        activator = self._act(activation)
        if activation:
            x = activator(x)
        return x

    def _act(self, activation):
        activator = None
        if activation:
            if isinstance(activation, str):
                if activation == 'relu':
                    activator = tf.nn.relu
                elif activation == 'tanh':
                    activator = tf.nn.tanh
                elif activation == 'prelu':
                    activator = prelu
                elif activation == 'lrelu':
                    activator = tf.nn.leaky_relu
            elif callable(activation):
                activator = activation
            else:
                raise ValueError('invalid activation!')
        return activator

    def _kernel(self, kernel_initializer, kernel_regularizer):
        ki = None
        if isinstance(kernel_initializer, str):
            if kernel_initializer == 'he_normal':
                ki = tf.keras.initializers.he_normal()
        elif callable(kernel_initializer):
            ki = kernel_initializer
        elif kernel_initializer:
            raise ValueError('invalid kernel initializer!')
        kr = None
        if isinstance(kernel_regularizer, str):
            if kernel_regularizer == 'l1':
                kr = tf.keras.regularizers.l1(self.weight_decay) if self.weight_decay else None
            elif kernel_regularizer == 'l2':
                kr = tf.keras.regularizers.l2(self.weight_decay) if self.weight_decay else None
        elif callable(kernel_regularizer):
            kr = kernel_regularizer
        elif kernel_regularizer:
            raise ValueError('invalid kernel regularizer!')
        return ki, kr

    def upscale(self, image, method='espcn', scale=None, direct_output=True, **kwargs):
        """Image up-scale layer

        Upsample `image` width and height by scale factor `scale[0]` and `scale[1]`.
        Perform upsample progressively: i.e. x12:= x2->x2->x3

        Args:
            image: tensors to upsample
            method: method could be 'espcn', 'nearest' or 'deconv'
            scale: None or int or [int, int]. If None, `scale`=`self.scale`
            direct_output: output channel is the desired RGB or Grayscale, if False, keep the same channels as `image`
        """
        _allowed_method = ('espcn', 'nearest', 'deconv')
        assert str(method).lower() in _allowed_method
        method = str(method).lower()
        act = kwargs.get('activator')

        scale_x, scale_y = to_list(scale, 2) or self.scale
        features = self.channel if direct_output else image.shape.as_list()[-1]
        while scale_x > 1 or scale_y > 1:
            if scale_x % 2 == 1 or scale_y % 2 == 1:
                if method == 'espcn':
                    image = pixel_shift(self.conv2d(
                        image, features * scale_x * scale_y, 3,
                        kernel_initializer='he_normal',
                        kernel_regularizer='l2'), [scale_x, scale_y], features)
                elif method == 'nearest':
                    image = pixel_shift(
                        tf.concat([image] * scale_x * scale_y, -1),
                        [scale_x, scale_y],
                        image.shape[-1])
                elif method == 'deconv':
                    image = self.deconv2d(image, features, 3,
                                          strides=[scale_y, scale_x],
                                          kernel_initializer='he_normal')
                if act:
                    image = act(image)
                break
            else:
                scale_x //= 2
                scale_y //= 2
                if method == 'espcn':
                    image = pixel_shift(self.conv2d(
                        image, features * 4, 3,
                        kernel_initializer='he_normal',
                        kernel_regularizer='l2'), [2, 2], features)
                elif method == 'nearest':
                    image = pixel_shift(
                        tf.concat([image] * 4, -1),
                        [2, 2],
                        image.shape[-1])
                elif method == 'deconv':
                    image = self.deconv2d(image, features, 3,
                                          strides=2,
                                          kernel_initializer='he_normal')
                if act:
                    image = act(image)
        return image
