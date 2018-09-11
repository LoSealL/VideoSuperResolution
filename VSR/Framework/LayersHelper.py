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

    def non_local(self, inputs, channel_scale=8, **kwargs):
        """Non-local block
        Refer to CVPR2018 "Non-local Neural Networks": https://arxiv.org/abs/1711.07971
        and "Self-Attention Generative Adversarial Networks": https://arxiv.org/abs/1805.08318

        Args:
            inputs: A tensor representing input feature maps
            channel_scale: An integer representing scale factor from inputs to embedded channel numbers
            kwargs: optional arguments for `conv2d`

        Return:
            Non-local residual, scaled by a trainable `gamma`, which is zero initialized.
        """
        try:
            name = kwargs.pop('name')
        except KeyError:
            name = None
        with tf.variable_scope(name, 'NonLocal'):
            C = inputs.shape[-1]
            shape = tf.shape(inputs)
            g = self.conv2d(inputs, C, 1, **kwargs)
            theta = self.conv2d(inputs, C // channel_scale, 1, **kwargs)
            phi = self.conv2d(inputs, C // channel_scale, 1, **kwargs)
            theta = tf.reshape(theta, [shape[0], -1, C // channel_scale])  # N*C
            phi = tf.reshape(phi, [shape[0], -1, C // channel_scale])  # N*C
            beta = tf.matmul(theta, phi, transpose_b=True)  # N*N
            beta = tf.nn.softmax(beta, axis=-1)
            non_local = tf.matmul(beta, tf.reshape(g, [shape[0], -1, C]))  # N*C
            non_local = tf.reshape(non_local, shape)  # H*W*C
            gamma = tf.Variable(0, dtype=tf.float32, name='gamma')
            non_local *= gamma
        return non_local

    """ frequently used bindings """

    def __getattr__(self, item):
        from functools import partial as P
        """Make an alignment for easy call. You can add more patterns as below.
        
        >>> Layers.relu_conv2d = Layers.conv2d(activation='relu')
        >>> Layers.bn_conv2d = Layers.conv2d(use_batchnorm=True)
        >>> Layers.sn_leaky_conv2d = Layers.conv2d(use_sn=True, activation='lrelu')
        
        NOTE: orders do not matter.
        """
        if 'conv2d' in item:
            items = item.split('_')
            kwargs = {
                'kernel_initializer': 'he_normal',
                'kernel_regularizer': 'l2',
                'use_batchnorm': False,
                'use_sn': False,
            }
            if 'bn' in items or 'batchnorm' in items:
                kwargs.update(use_batchnorm=True)
            if 'sn' in items or 'spectralnorm' in items:
                kwargs.update(use_sn=True)
            if 'relu' in items:
                kwargs.update(activation='relu')
            if 'leaky' in items or 'lrelu' in items or 'leakyrelu' in items:
                kwargs.update(activation='lrelu')
            if 'tanh' in items:
                kwargs.update(activation='tanh')
            return P(self.conv2d, **kwargs)

        return None

    def resblock(self, x,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 use_batchnorm=False,
                 use_sn=False,
                 kernel_initializer='he_normal',
                 kernel_regularizer='l2',
                 bn_placement=None,
                 **kwargs):
        """make a residual block

        Args:
            bn_placement: 'front' or 'behind', use BN layer in front of or behind after the 1st conv2d layer.
        """

        kwargs.update({
            'strides': strides,
            'padding': padding,
            'data_format': data_format,
            'activation': activation,
            'use_bias': use_bias,
            'use_batchnorm': use_batchnorm,
            'use_sn': use_sn,
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer
        })
        if bn_placement is None: bn_placement = 'behind'
        assert bn_placement in ('front', 'behind')
        try:
            name = kwargs.pop('name')
        except KeyError:
            name = None
        with tf.variable_scope(name, 'ResBlock'):
            ori = x
            if bn_placement == 'front':
                act = self._act(activation)
                x = tf.layers.batch_normalization(x, training=self.training_phase)
                if act: x = act(x)
            x = self.conv2d(x, filters, kernel_size, **kwargs)
            kwargs.pop('activation')
            if bn_placement == 'front': kwargs.pop('use_batchnorm')
            x = self.conv2d(x, filters, kernel_size, **kwargs)
            if ori.shape[-1] != x.shape[-1]:
                # short cut
                ori = self.conv2d(ori, x.shape[-1], 1, kernel_initializer=kernel_initializer)
            ori += x
        return ori
