"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug 21st 2018

Utility for motion compensation
"""
import numpy as np
import tensorflow as tf


def _grid_norm(width, height, bounds=(-1.0, 1.0)):
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
                  tf.transpose(tf.expand_dims(
                      tf.linspace(*bounds, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(*bounds, height), 1),
                  tf.ones(shape=tf.stack([1, width])))

  grid = tf.stack([x_t, y_t], axis=-1)
  return grid


def _grid(width, height, batch=1, dtype=None, with_batch=False):
  """generate a mesh grid

    Args:
        batch: batch size
        width: width of the pixels(mesh)
        height: height of the pixels
    Return:
        A grid of shape [B, H, W, 3]
  """
  b = tf.range(0, batch)
  h = tf.range(0, height)
  w = tf.range(0, width)
  grid = tf.meshgrid(w, h, b)
  grid.reverse()
  grid = tf.stack(grid, -1)
  grid = tf.transpose(grid, [2, 0, 1, 3])
  if dtype:
    grid = tf.cast(grid, dtype)
  if with_batch:
    return grid
  return grid[..., 1:]


def _sample(image, x, y):
  """bilinear sample image at coordinate x, y

    Args:
        image: a 4-D tensor of shape [B H W C]
        x, y: a 3-D tensor of shape [B H W], pixel index of image

    Return:
        sampled images, that
        output[i, j] = image[y[i], x[j]]
  """

  shape = tf.shape(image)
  batch = shape[0]
  h = shape[1]
  w = shape[2]

  x = tf.to_float(x)
  y = tf.to_float(y)
  image = tf.to_float(image)
  x0 = tf.to_int32(tf.floor(x))
  y0 = tf.to_int32(tf.floor(y))
  x1 = x0 + 1
  y1 = y0 + 1

  w00 = tf.expand_dims((tf.to_float(x1) - x) * (tf.to_float(y1) - y), -1)
  w01 = tf.expand_dims((x - tf.to_float(x0)) * (tf.to_float(y1) - y), -1)
  w10 = tf.expand_dims((tf.to_float(x1) - x) * (y - tf.to_float(y0)), -1)
  w11 = tf.expand_dims((x - tf.to_float(x0)) * (y - tf.to_float(y0)), -1)

  x0 = tf.clip_by_value(x0, 0, w - 1)
  y0 = tf.clip_by_value(y0, 0, h - 1)
  x1 = tf.clip_by_value(x1, 0, w - 1)
  y1 = tf.clip_by_value(y1, 0, h - 1)

  batch_idx = tf.reshape(tf.range(0, batch), [batch, 1, 1])
  batch_idx = tf.tile(batch_idx, [1, h, w])
  gather_00 = tf.stack([batch_idx, y0, x0], axis=-1)
  gather_01 = tf.stack([batch_idx, y0, x1], axis=-1)
  gather_10 = tf.stack([batch_idx, y1, x0], axis=-1)
  gather_11 = tf.stack([batch_idx, y1, x1], axis=-1)

  p00 = tf.gather_nd(image, gather_00) * w00
  p01 = tf.gather_nd(image, gather_01) * w01
  p10 = tf.gather_nd(image, gather_10) * w10
  p11 = tf.gather_nd(image, gather_11) * w11

  return tf.add_n([p00, p01, p10, p11])


def _move(image, x, y):
  """move source image to target coordinate x, y"""
  shape = tf.shape(image)
  batch = shape[0]
  h = shape[1]
  w = shape[2]

  x = tf.to_float(x)
  y = tf.to_float(y)
  image = tf.to_float(image)
  x0 = tf.to_int32(tf.floor(x))
  y0 = tf.to_int32(tf.floor(y))
  x1 = x0 + 1
  y1 = y0 + 1

  w00 = tf.expand_dims((tf.to_float(x1) - x) * (tf.to_float(y1) - y), -1)
  w01 = tf.expand_dims((x - tf.to_float(x0)) * (tf.to_float(y1) - y), -1)
  w10 = tf.expand_dims((tf.to_float(x1) - x) * (y - tf.to_float(y0)), -1)
  w11 = tf.expand_dims((x - tf.to_float(x0)) * (y - tf.to_float(y0)), -1)

  x0 = tf.clip_by_value(x0, 0, w - 1)
  y0 = tf.clip_by_value(y0, 0, h - 1)
  x1 = tf.clip_by_value(x1, 0, w - 1)
  y1 = tf.clip_by_value(y1, 0, h - 1)

  batch_idx = tf.reshape(tf.range(0, batch), [batch, 1, 1])
  batch_idx = tf.tile(batch_idx, [1, h, w])
  scatter_00 = tf.stack([batch_idx, y0, x0], axis=-1)
  scatter_01 = tf.stack([batch_idx, y0, x1], axis=-1)
  scatter_10 = tf.stack([batch_idx, y1, x0], axis=-1)
  scatter_11 = tf.stack([batch_idx, y1, x1], axis=-1)

  p00 = tf.scatter_nd(scatter_00, image * w00, shape)
  p01 = tf.scatter_nd(scatter_01, image * w01, shape)
  p10 = tf.scatter_nd(scatter_10, image * w10, shape)
  p11 = tf.scatter_nd(scatter_11, image * w11, shape)

  return tf.add_n([p00, p01, p10, p11])


def warp(image, u, v, additive_warp=True, normalized=False):
  """warp the image with flow(u, v)

  If flow=[u, v], representing motion from img1 to img2
  then `warp(img2, u, v)->img1~`

  Args:
       image: a 4-D tensor [B, H, W, C], images to warp
       u: horizontal motion vectors of optical flow
       v: vertical motion vectors of optical flow
       additive_warp: a boolean, if False, regard [u, v]
         as destination coordinate rather than motion
         vectors.
       normalized: a boolean, if True, regard [u, v] as
       [-1, 1] and scaled to [-W, W], [-H, H] respectively.

  Note: usually nobody uses a normalized optical flow...
  """
  shape = tf.shape(image)
  b, h, w = shape[0], shape[1], shape[2]

  if normalized:
    if not additive_warp:
      u = (u + 1) * 0.5
      v = (v + 1) * 0.5
    u *= tf.to_float(w)
    v *= tf.to_float(h)

  if additive_warp:
    grids = _grid(w, h, dtype=tf.float32)
    u += grids[..., 1]
    v += grids[..., 0]

  return _sample(image, u, v)


def epe(label, predict):
  """End-point error of optical flow"""
  ux, vx = predict[..., 0], predict[..., 1]
  uy, vy = label[..., 0], label[..., 1]
  diff = tf.squared_difference(ux, uy) + tf.squared_difference(vx, vy)
  return tf.sqrt(diff, name='EPE')


def viz_flow(flow):
  """Visualize optical flow in TF"""
  from VSR.Util.VisualizeOpticalFlow import _color_wheel
  with tf.name_scope('VizFlow'):
    color_wheel = _color_wheel().astype('float32')
    n_cols = color_wheel.shape[0]
    u, v = flow[..., 0], flow[..., 1]
    rot = tf.atan2(-v, -u) / np.pi
    fk = (rot + 1) / 2 * (n_cols - 1)  # -1~1 mapped to 0~n_cols
    k0 = tf.to_int32(fk)  # 0, 1, 2, ..., n_cols
    k1 = tf.mod(k0 + 1, n_cols)
    f = fk - tf.to_float(k0)
    f = tf.expand_dims(f, -1)
    col0 = tf.gather_nd(color_wheel, tf.expand_dims(k0, -1))
    col1 = tf.gather_nd(color_wheel, tf.expand_dims(k1, -1))
    col = (1 - f) * col0 + f * col1
  return col
