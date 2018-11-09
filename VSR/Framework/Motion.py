"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug 21st 2018

Utility for motion compensation
"""
import tensorflow as tf
import numpy as np

try:
    import png
except ImportError:
    tf.logging.warning('Unable to import pypng, cannot read 16bit png')


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
    B = shape[0]
    H = shape[1]
    W = shape[2]

    x = tf.to_float(x)
    y = tf.to_float(y)
    image = tf.to_float(image)
    x0 = tf.to_int32(tf.floor(x))
    y0 = tf.to_int32(tf.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, W)
    y0 = tf.clip_by_value(y0, 0, H)
    x1 = tf.clip_by_value(x1, 0, W)
    y1 = tf.clip_by_value(y1, 0, H)

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


def _move(image, x, y):
    """move source image to target coordinate x, y"""
    shape = tf.shape(image)
    B = shape[0]
    H = shape[1]
    W = shape[2]

    x = tf.to_float(x)
    y = tf.to_float(y)
    image = tf.to_float(image)
    x0 = tf.to_int32(tf.floor(x))
    y0 = tf.to_int32(tf.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, W)
    y0 = tf.clip_by_value(y0, 0, H)
    x1 = tf.clip_by_value(x1, 0, W)
    y1 = tf.clip_by_value(y1, 0, H)

    batch_idx = tf.reshape(tf.range(0, B), [B, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, H, W])
    scatter_00 = tf.stack([batch_idx, y0, x0], axis=-1)
    scatter_01 = tf.stack([batch_idx, y0, x1], axis=-1)
    scatter_10 = tf.stack([batch_idx, y1, x0], axis=-1)
    scatter_11 = tf.stack([batch_idx, y1, x1], axis=-1)

    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    w00 = tf.expand_dims((x1 - x) * (y1 - y), -1)
    w01 = tf.expand_dims((x - x0) * (y1 - y), -1)
    w10 = tf.expand_dims((x1 - x) * (y - y0), -1)
    w11 = tf.expand_dims((x - x0) * (y - y0), -1)

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
    B, H, W = shape[0], shape[1], shape[2]

    if normalized:
        if not additive_warp:
            u = (u + 1) * 0.5
            v = (v + 1) * 0.5
        u *= tf.to_float(W)
        v *= tf.to_float(H)

    if additive_warp:
        G = _grid(W, H, dtype=tf.float32)
        u += G[..., 1]
        v += G[..., 0]

    return _sample(image, u, v)


def epe(label, predict):
    """End-point error of optical flow"""
    ux, vx = predict[..., 0], predict[..., 1]
    uy, vy = label[..., 0], label[..., 1]
    diff = tf.squared_difference(ux, uy) + tf.squared_difference(vx, vy)
    return tf.sqrt(diff, name='EPE')


def viz_flow(flow):
    """Visualize optical flow in TF"""
    from .Callbacks import _color_wheel
    with tf.name_scope('VizFlow'):
        colorwheel = _color_wheel().astype('float32')
        ncols = colorwheel.shape[0]
        u, v = flow[..., 0], flow[..., 1]
        rot = tf.atan2(-v, -u) / np.pi
        fk = (rot + 1) / 2 * (ncols - 1)  # -1~1 maped to 0~ncols
        k0 = tf.to_int32(fk)  # 0, 1, 2, ..., ncols
        k1 = tf.mod(k0 + 1, ncols)
        f = fk - tf.to_float(k0)
        f = tf.expand_dims(f, -1)
        col0 = tf.gather_nd(colorwheel, tf.expand_dims(k0, -1))
        col1 = tf.gather_nd(colorwheel, tf.expand_dims(k1, -1))
        col = (1 - f) * col0 + f * col1
    return col


def open_flo(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def write_flo(filename, uv, v=None):
    """ Write optical flow to file.
    
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    TAG_CHAR = np.array([202021.25], np.float32)

    if v is None:
        u = uv[..., 0]
        v = uv[..., 1]
    else:
        u = uv
    height, width = u.shape
    with open(filename, 'wb') as f:
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


def open_png16(fn):
    """Read 16bit png file"""

    reader = png.Reader(fn)
    data = reader.asDirect()
    pixels = []
    for row in data[2]:
        row = np.reshape(np.asarray(row), [-1, 3])
        pixels += [row]
    return np.stack(pixels, 0)


class KITTI:
    @staticmethod
    def open_flow(fn):
        flow = open_png16(fn)
        valid = flow[..., -1]
        u = flow[..., 0].astype('float32')
        v = flow[..., 1].astype('float32')
        u = (u - 2 ** 15) / 64 * valid
        v = (v - 2 ** 15) / 64 * valid
        return np.stack([u, v], -1)
