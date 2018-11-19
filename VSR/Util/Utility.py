"""
Copyright: Wenyi Tang 2017-2018
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


def str_to_bytes(s):
    """convert string to byte unit. Case insensitive.

    >>> str_to_bytes('2GB')
      2147483648
    >>> str_to_bytes('1kb')
      1024
    """
    s = s.replace(' ', '')
    if s[-1].isalpha() and s[-2].isalpha():
        _unit = s[-2:].upper()
        _num = s[:-2]
    elif s[-1].isalpha():
        _unit = s[-1].upper()
        _num = s[:-1]
    else:
        return float(s)
    if not _unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'):
        raise ValueError('invalid unit', _unit)
    carry = {'B': 1,
             'KB': 1024,
             'MB': 1024 ** 2,
             'GB': 1024 ** 3,
             'TB': 1024 ** 4,
             'PB': 1024 ** 5,
             'EB': 1024 ** 6,
             'ZB': 1024 ** 7,
             'YB': 1024 ** 8}
    return float(_num) * carry[_unit]


def shrink_mod_scale(x, scale):
    """clip each dim of x to multiple of scale"""
    return [_x - _x % _s for _x, _s in zip(x, to_list(scale, 2))]


def repeat(x, n):
    """Repeats a 2D tensor. Copy from Keras

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

      Args:
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

      Returns:
        A tensor.
    """
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)


def pixel_shift(image, scale, channel=1):
    """Efficient Sub-pixel Convolution, see paper: https://arxiv.org/abs/1609.05158

      Args:
          image: A 4-D tensor of [N, H, W, C*scale[0]*scale[1]]
          scale: A scalar or 1-D tensor with 2 elements, the scale factor for width and height respectively
          channel: specify the channel number

      Return:
          A 4-D tensor of [N, H*scale[1], W*scale[0], C]
    """

    with tf.name_scope('PixelShift'):
        r = to_list(scale, 2)
        shape = tf.shape(image)
        H, W = shape[1], shape[2]
        image = tf.reshape(image, [-1, H, W, r[1], r[0], channel])
        image = tf.transpose(image, perm=[0, 1, 3, 2, 4, 5])  # B, H, r, W, r, C
        image = tf.reshape(image, [-1, H * r[1], W * r[0], channel])
        return image


def crop_to_batch(image, scale):
    """Crop image into `scale[0]*scale[1]` parts, and concat into batch dimension

      Args:
          image: A 4-D tensor with [N, H, W, C]
          scale: A 1-D tensor or scalar of scale factor for width and height
    """

    scale = to_list(scale, 2)
    with tf.name_scope('BatchEnhance'):
        hs = tf.split(image, scale[1], axis=1)
        image = tf.concat(hs, axis=0)
        rs = tf.split(image, scale[0], axis=2)
        return tf.concat(rs, axis=0)


def bicubic_rescale(img, scale):
    """Resize image in tensorflow.

    NOTE: tf.image.resize_bicubic uses different boundary to PIL.Image.resize,
      try to use resize_area without aligned corners.
    """
    with tf.name_scope('Bicubic'):
        shape = tf.shape(img)
        scale = to_list(scale, 2)
        shape_enlarge = tf.to_float(shape) * [1, *scale, 1]
        shape_enlarge = tf.to_int32(shape_enlarge)
        return tf.image.resize_bicubic(img, shape_enlarge[1:3], False)


def _bicubic_filter(x, a=0.5):
    # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    if x < 0:
        x = -x
    if x < 1:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0


def upsample(img, scale):
    scale = to_list(scale, 2)
    with tf.name_scope(f'UpsampleX{scale[0]}'):
        shape = img.shape
        imgx = tf.pad(img, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")
        imgx = tf.pad(imgx, [[0, 0], [0, 1], [0, 0], [0, 0]], "CONSTANT")
        dh = [(i + 0.5) / scale[1] for i in range(scale[1])]
        dw = [(i + 0.5) / scale[0] for i in range(scale[0])]
        ph, pw = [], []
        for d in dh:
            _v = np.array([-1, 0, 1, 2], np.float32) - d + 0.5
            if d < 0.5:
                _v[-1] = 2
            else:
                _v[0] = 2
            _k = np.asarray([_bicubic_filter(v) for v in _v], np.float32)
            _k /= np.sum(_k) + 1e-12
            _k = _k.reshape([4, 1, 1])
            if shape[-1] == 3:
                zero = tf.zeros_like(_k)
                _r = tf.concat([_k, zero, zero], -1)
                _g = tf.concat([zero, _k, zero], -1)
                _b = tf.concat([zero, zero, _k], -1)
                _k = tf.stack([_r, _g, _b], -1)
            else:
                _k = tf.expand_dims(_k, -1)
            ph += [tf.nn.conv2d(imgx, _k, (1, 1, 1, 1), 'VALID', name='Hori')]
        img = pixel_shift(tf.concat(ph, -1), [1, scale[1]], shape[-1])
        img = tf.round(img)
        imgx = tf.pad(img, [[0, 0], [0, 0], [1, 1], [0, 0]], "CONSTANT")
        imgx = tf.pad(imgx, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")
        for d in dw:
            _v = np.array([-1, 0, 1, 2], np.float32) - d + 0.5
            if d < 0.5:
                _v[-1] = 2
            else:
                _v[0] = 2
            _k = np.asarray([_bicubic_filter(v) for v in _v], np.float32)
            _k /= np.sum(_k) + 1e-12
            _k = _k.reshape([4, 1, 1])
            if shape[-1] == 3:
                zero = tf.zeros_like(_k)
                _r = tf.concat([_k, zero, zero], -1)
                _g = tf.concat([zero, _k, zero], -1)
                _b = tf.concat([zero, zero, _k], -1)
                _k = tf.stack([_r, _g, _b], -1)
            else:
                _k = tf.expand_dims(_k, -1)
            _k = tf.transpose(_k, [1, 0, 2, 3])
            pw += [tf.nn.conv2d(imgx, _k, (1, 1, 1, 1), 'VALID', name='Vert')]
        img = pixel_shift(tf.concat(pw, -1), [scale[0], 1], shape[-1])
        return tf.round(img)


def prelu(x, initialize=0, name=None, scope='PReLU'):
    """Parametric ReLU"""
    with tf.variable_scope(name, scope):
        alphas = tf.get_variable(
            'Variable', shape=(x.shape[-1],), dtype='float32',
            initializer=tf.initializers.constant(initialize))
        return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5


def gaussian_kernel(kernel_size, width):
    """generate a gaussian kernel"""

    kernel_size = np.asarray(to_list(kernel_size, 2), np.int32)
    kernel_size = kernel_size - kernel_size % 2
    half_ksize = kernel_size // 2
    x, y = np.mgrid[-half_ksize[0]:half_ksize[0] + 1,
           -half_ksize[1]:half_ksize[1] + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / 2 * width) / (2 * np.pi * width ** 2)
    return kernel / kernel.sum()


def imfilter(image, kernel, name=None):
    """Image filter"""
    with tf.variable_scope('imfilter'):
        _c = image.shape[-1]
        _k = tf.expand_dims(kernel, -1)
        _k = tf.expand_dims(_k, -1)
        _p = tf.zeros_like(_k)
        _m = []
        # apply kernel to each channel separately
        for i in range(_c):
            t = [_p] * _c
            t[i] = _k
            _m.append(tf.concat(t, -1))
        _k = tf.concat(_m, -2)
        _k = tf.cast(_k, dtype=image.dtype)
        return tf.nn.conv2d(image, _k, strides=[1, 1, 1, 1], padding='SAME',
                            name=name)


def pixel_norm(images, epsilon=1.0e-8, scale=1.0, bias=0):
    """Pixel normalization.

    For each pixel a[i,j,k] of image in HWC format, normalize its value to
    b[i,j,k] = a[i,j,k] / SQRT(SUM_k(a[i,j,k]^2) / C + eps).

    Args:
      images: A 4D `Tensor` of NHWC format.
      epsilon: A small positive number to avoid division by zero.
      scale: scale the normalized value
      bias: add bias to output

    Returns:
      A 4D `Tensor` with pixel-wise normalized channels.
    """
    return images * scale * tf.rsqrt(
        tf.reduce_mean(tf.square(images), axis=3,
                       keepdims=True) + epsilon) + bias


def color_consistency(feature, label, lambd=5):
    """Color consistency regularization (from StackGAN++)

    See: https://arxiv.org/abs/1710.10916
    """

    m1, m2 = tf.reduce_mean(feature, [1, 2], True), tf.reduce_mean(label,
                                                                   [1, 2], True)
    B, H, W, C = tf.shape(feature)[0], tf.shape(feature)[1], tf.shape(feature)[
        2], tf.shape(feature)[3]
    f_hat = tf.reshape(feature - m1, [B, -1, C])
    l_hat = tf.reshape(label - m2, [B, -1, C])
    c1 = tf.matmul(f_hat, f_hat, True) / tf.cast(H * W, tf.float32)
    c2 = tf.matmul(l_hat, l_hat, True) / tf.cast(H * W, tf.float32)
    cc = tf.losses.mean_squared_error(m1, m2) + tf.losses.mean_squared_error(c1,
                                                                             c2,
                                                                             lambd)
    return cc


def pop_dict_wo_keyerror(d, key):
    value = d.get(key)
    if value is not None:
        d.pop(key)
    return value


def summary_tensor_image(x, name, shape):
    """summary a tensor

    split each channel and form a huge image
    """
    raise NotImplementedError


def _make_vector(x, patch=3, stride=1):
    """[B, H, W, C]->[B, H, W, c*k1*k2]"""
    k1, k2 = to_list(patch, 2)
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    padded_x = tf.pad(x, [[0, 0], [k1 // 2] * 2, [k2 // 2] * 2, [0, 0]])
    vec = []
    for i in range(k1):
        for j in range(k2):
            vec.append(padded_x[:, i:i + h:stride, j:j + w:stride, :])
    return tf.concat(vec, axis=-1)


def _make_displacement(x, patch=3, max_dis=1, stride1=1, stride2=1):
    """[B, H, W, C]->[B, H, W, V, d*d]"""
    k1, k2 = to_list(patch, 2)
    d1, d2 = to_list(max_dis, 2)
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    padding = [[0, 0], [k1 // 2 + d1] * 2, [k2 // 2 + d2] * 2, [0, 0]]
    padded_x = tf.pad(x, padding)
    disp = []
    vec = []
    for i in range(0, 2 * d1 + 1, stride2):
        for j in range(0, 2 * d2 + 1, stride2):
            for k in range(k1):
                for l in range(k2):
                    vec.append(padded_x[
                               :,
                               i + k:i + k + h:stride1,
                               j + l:j + l + w:stride1,
                               :])
            disp.append(tf.concat(vec, -1))
            vec.clear()
    return tf.stack(disp, axis=-1)


def correlation(f1, f2, patch, max_displacement, stride1=1, stride2=1):
    """calculate correlation between feature map "f1" and "f2".
    See "FlowNet: Learning Optical Flow with Convolutional Networks" for
    details.

    Args:
        f1: a 4-D tensor with shape [B, H, W, C]
        f2: a 4-D tensor with shape [B, H, W, C]
        patch: an integer or a list like [k1, k2], window size for comparison
        max_displacement: an integer, representing the max searching distance
        stride1: stride for patch
        stride2: stride for displacement

    Returns:
        a 4-D correlation tensor with shape [B, H, W, d*d]
    """
    channel = f1.shape[-1]
    norm = np.prod(to_list(patch, 2) + [channel])
    v1 = _make_vector(f1, patch, stride1)
    v1 = tf.expand_dims(v1, -2)
    v2 = _make_displacement(f2, patch, max_displacement, stride1, stride2)
    corr = tf.matmul(v1, v2) / tf.to_float(norm)
    return tf.squeeze(corr, axis=-2)


class Vgg:
    """use pre-trained VGG network from keras.application.vgg16
    to obtain outputs of specific layers
    """

    def __init__(self, include_top=False, input_shape=None, type='vgg19'):
        with tf.variable_scope('VGG'):
            from tensorflow.keras import utils as keras_utils
            if np.size(input_shape) > 3:
                input_shape = input_shape[-3:]
            elif np.size(input_shape) < 3:
                raise ValueError('input shape must be [H, W, 3]')
            if type == 'vgg16':
                if include_top:
                    from keras_applications.vgg16 import WEIGHTS_PATH as w_16
                    self.weights_path = keras_utils.get_file(
                        'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                        w_16,
                        cache_subdir='models',
                        file_hash='64373286793e3c8b2b4e3219cbf3544b')
                else:
                    from keras_applications.vgg16 import \
                        WEIGHTS_PATH_NO_TOP as w_16_notop
                    self.weights_path = keras_utils.get_file(
                        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        w_16_notop,
                        cache_subdir='models',
                        file_hash='6d6bbae143d832006294945121d1f1fc')
                self._m = tf.keras.applications.vgg16.VGG16(
                    include_top=include_top, input_shape=tuple(input_shape))
            elif type == 'vgg19':
                if include_top:
                    from keras_applications.vgg19 import WEIGHTS_PATH as w_19
                    self.weights_path = keras_utils.get_file(
                        'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                        w_19,
                        cache_subdir='models',
                        file_hash='cbe5617147190e668d6c5d5026f83318')
                else:
                    from keras_applications.vgg19 import \
                        WEIGHTS_PATH_NO_TOP as w_19_notop
                    self.weights_path = keras_utils.get_file(
                        'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        w_19_notop,
                        cache_subdir='models',
                        file_hash='253f8cb515780f3b799900260a226db6')
                self._m = tf.keras.applications.vgg19.VGG19(
                    include_top=include_top, input_shape=tuple(input_shape))
            self._vgg_mean = [103.939, 116.779, 123.68]
            self.include_top = include_top
            self.model = None

    def __call__(self, x, *args, **kwargs):
        return self.call(x, *args, **kwargs)

    def call(self, x, conv=None, block=None, yuv_to_rgb_convert=False):
        """get the output of a pre-trained VGG16 network

            Args:
                conv: the convolution layer in block, start by 0 in each block
                block: the block number, range from [1, 5]
                yuv_to_rgb_convert: need to convert from yuv to rgb

            Return:
                the output of given layer

            Note:
                if `conv` and `block` are lists, return a list of outputs;
                if `conv` and `block` are None, the output is of the last softmax layer
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
            if not outputs:
                outputs = self._m.outputs
            self.model = tf.keras.Model(self._m.input, outputs, name='VGG')
            return self.model(x)

    def restore(self, *args, **kwargs):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('No session initialized')
        self.model.load_weights(self.weights_path)

    def save(self, *args, **kwargs):
        pass

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


class SpectralNorm(tf.keras.constraints.Constraint):
    """Spectral normalization constraint.
      Ref: https://arxiv.org/pdf/1802.05957

      Args:
          iteration: power iteration, default 1
      Note:
          use SN as a kernel constraint seems not work well
          I now use like this:
          >>> nn = tf.layers.Conv2D(...)
          >>> nn.build(input_shape=x.shape.as_list())
          >>> if use_sn: nn.kernel = SpectralNorm()(nn.kernel)
          >>> y = nn(x)
    """

    def __init__(self, iteration=1):
        self.pi = iteration

    def __call__(self, w):
        scope = w.op.name.replace(':', '') + '/snorm'
        with tf.variable_scope(scope):
            w_shape = w.shape.as_list()
            w = tf.reshape(w, [-1, w_shape[-1]])
            u = tf.get_variable(
                'u',
                shape=(w.shape[0], 1),
                dtype=w.dtype,
                collections=[tf.GraphKeys.MODEL_VARIABLES,
                             tf.GraphKeys.GLOBAL_VARIABLES],
                initializer=tf.random_normal_initializer(),
                trainable=False)
            u_hat = u
            v_hat = None
            for i in range(self.pi):
                # power iteration
                v_hat = tf.nn.l2_normalize(tf.matmul(tf.transpose(w), u_hat),
                                           axis=None, epsilon=1e-12)
                u_hat = tf.nn.l2_normalize(tf.matmul(w, v_hat),
                                           axis=None, epsilon=1e-12)
            u_hat = tf.stop_gradient(u_hat)
            v_hat = tf.stop_gradient(v_hat)
            sigma = tf.matmul(tf.matmul(tf.transpose(u_hat), w), v_hat)
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
            return w_norm

    def get_config(self):
        return {"iteration": self.pi}
