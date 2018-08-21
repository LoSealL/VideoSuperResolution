"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug 21st 2018

Utility for motion compensation
"""
import tensorflow as tf


def _grid(width, height, bounds=(-1.0, 1.0)):
    """generate a normalized mesh grid

      Args:
          width: width of the pixels(mesh)
          height: height of the pixels
          bounds: normalized lower and upper bounds
      Return:
          This should be equivalent to:
          >>>  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
          >>>                         np.linspace(-1, 1, height))
          >>>  ones = np.ones(np.prod(x_t.shape))
          >>>  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(*bounds, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(*bounds, height), 1),
                    tf.ones(shape=tf.stack([1, width])))

    # x_t_flat = tf.reshape(x_t, (1, -1))
    # y_t_flat = tf.reshape(y_t, (1, -1))
    #
    # ones = tf.ones_like(x_t_flat)
    # grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
    grid = tf.stack([x_t, y_t], axis=-1)
    return grid


def _sample(image, x, y):
    """bilinear sample image at coord

      Args:
          image: a 4-D tensor of shape [B H W C]
          x, y: a 3-D tensor of shape [B H W]

      Return:
          sampled images
    """

    shape = tf.shape(image)
    B = shape[0]
    H = shape[1]
    W = shape[2]

    x0 = tf.cast(tf.floor(x), dtype=tf.int32)
    y0 = tf.cast(tf.floor(y), dtype=tf.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, W - 1)
    y0 = tf.clip_by_value(y0, 0, H - 1)
    x1 = tf.clip_by_value(x1, 0, W - 1)
    y1 = tf.clip_by_value(y1, 0, H - 1)

    batch_idx = tf.reshape(tf.range(0, B), [B, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, H, W])
    gather_00 = tf.stack([batch_idx, y0, x0], axis=-1)
    gather_01 = tf.stack([batch_idx, y0, x1], axis=-1)
    gather_10 = tf.stack([batch_idx, y1, x0], axis=-1)
    gather_11 = tf.stack([batch_idx, y1, x1], axis=-1)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    w00 = tf.expand_dims((x1 - x) * (y1 - y), -1)
    w01 = tf.expand_dims((x - x0) * (y1 - y), -1)
    w10 = tf.expand_dims((x1 - x) * (y - y0), -1)
    w11 = tf.expand_dims((x - x0) * (y - y0), -1)

    p00 = tf.gather_nd(image, gather_00) * w00
    p01 = tf.gather_nd(image, gather_01) * w01
    p10 = tf.gather_nd(image, gather_10) * w10
    p11 = tf.gather_nd(image, gather_11) * w11

    return tf.add_n([p00, p01, p10, p11])


def warp(image, coordinate, additive_warp=False):
    shape = tf.shape(image)
    H = tf.cast(shape[1], tf.float32)
    W = tf.cast(shape[2], tf.float32)

    if additive_warp:
        coordinate += _grid(W, H)

    warp_pos = (coordinate + 1) * 0.5  # [0, 1]
    x = warp_pos[..., 0]
    y = warp_pos[..., 1]
    x *= (W - 1)
    y *= (H - 1)

    return _sample(image, x, y)
