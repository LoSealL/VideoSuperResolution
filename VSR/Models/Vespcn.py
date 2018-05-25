"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 12th 2018
Updated Date: May 12th 2018

Video-ESPCN
Ref https://arxiv.org/abs/1501.00092
"""
from VSR.Framework.SuperResolution import SuperResolution
from VSR.Util import Utility
from VSR.Util.Utility import repeat

import tensorflow as tf
import numpy as np


class Spmc:
    def __init__(self):
        self.nn = {
            'coarse': {
                'kernel': [5, 3, 5, 3, 3],
                'filters': [24, 24, 24, 24, 32],
                'strides': [2, 1, 2, 1, 1]
            },
            'fine': {
                'kernel': [5, 3, 3, 3, 3],
                'filters': [24, 24, 24, 24, 8],
                'strides': [2, 1, 1, 1, 1]
            }
        }
        self.weights = {}
        self.bias = {}
        # build weights
        c = 2  # coarse input channel
        self.weights['coarse'] = []
        self.bias['coarse'] = []
        for k, f in zip(self.nn['coarse']['kernel'], self.nn['coarse']['filters']):
            w = tf.initializers.orthogonal(np.sqrt(2))(shape=[k, k, c, f])
            b = Utility.bias([f])
            c = f
            self.weights['coarse'].append(w)
            self.bias['coarse'].append(b)

        c = 5  # fine input channel
        self.weights['fine'] = []
        self.bias['fine'] = []
        for k, f in zip(self.nn['fine']['kernel'], self.nn['fine']['filters']):
            w = tf.initializers.orthogonal(np.sqrt(2))(shape=[k, k, c, f])
            b = Utility.bias([f])
            c = f
            self.weights['fine'].append(w)
            self.bias['fine'].append(b)

    def __call__(self, x, **kwargs):
        stacked_inp = tf.split(x, 2, axis=-1)
        coarse_me = self._coarse_flow(x)
        coarse_y = self._motion_compensate(tf.concat([stacked_inp[1], coarse_me], axis=-1))
        fine_me = self._fine_flow(tf.concat([*stacked_inp, coarse_me, coarse_y], axis=-1))
        total_me = coarse_me + fine_me
        fine_y = self._motion_compensate(tf.concat([stacked_inp[1], total_me], axis=-1))
        return fine_y

    def _coarse_flow(self, x):
        w = self.weights['coarse']
        b = self.bias['coarse']
        s = self.nn['coarse']['strides']
        inp = x
        for _w, _b, _s in zip(w[:-1], b[:-1], s[:-1]):
            t = tf.nn.conv2d(inp, _w, (1, _s, _s, 1), 'SAME', name='spmc/coarse/conv2d') + _b
            inp = tf.nn.relu(t)
        inp = tf.nn.tanh(tf.nn.conv2d(inp, w[-1], (1, s[-1], s[-1], 1), 'SAME') + b[-1])
        return Utility.pixel_shift(inp, 4, 2)

    def _fine_flow(self, x):
        w = self.weights['fine']
        b = self.bias['fine']
        s = self.nn['fine']['strides']
        inp = x
        for _w, _b, _s in zip(w[:-1], b[:-1], s[:-1]):
            t = tf.nn.conv2d(inp, _w, (1, _s, _s, 1), 'SAME', name='spmc/fine/conv2d') + _b
            inp = tf.nn.relu(t)
        inp = tf.nn.tanh(tf.nn.conv2d(inp, w[-1], (1, s[-1], s[-1], 1), 'SAME') + b[-1])
        return Utility.pixel_shift(inp, 2, 2)

    @staticmethod
    def _gen_grid(shape, warp):
        """generate coord grid

        :param shape: the frame shape
        :param warp: warp tensor by flow estimation
        :return: the grid contains 4 tensor, upper-left, upper-right,
                  lower-right, lower-left (clock-wise respectively);
                  the scale factor is the interpolation of each pixel.
        """
        grid, scale = [], []
        h = tf.expand_dims(tf.range(0, shape[1], dtype=tf.int32), -1)
        w = tf.expand_dims(tf.range(0, shape[2], dtype=tf.int32), -1)
        b = tf.range(0, shape[0], dtype=tf.int32)
        grid_x = tf.transpose(repeat(w, shape[1])[..., 0])
        grid_y = repeat(h, shape[2])[..., 0]
        # make batch
        grid_x = tf.transpose(repeat(grid_x, shape[0]), [1, 0, 2])
        grid_y = tf.transpose(repeat(grid_y, shape[0]), [1, 0, 2])
        grid_batch = tf.transpose(tf.ones(shape[:-1]), [1, 2, 0]) * tf.cast(b, tf.float32)
        warp_x = warp[..., 0]
        warp_y = warp[..., 1]
        warp_xf = tf.cast(warp_x, tf.float32)
        warp_yf = tf.cast(warp_y, tf.float32)
        grid_xf = tf.cast(grid_x, tf.float32)
        grid_yf = tf.cast(grid_y, tf.float32)
        # equal to tf.floor
        coord_x = tf.clip_by_value(tf.cast(grid_xf + warp_xf, tf.int32), 0, shape[2] - 2)
        coord_y = tf.clip_by_value(tf.cast(grid_yf + warp_yf, tf.int32), 0, shape[1] - 2)
        coord_batch = tf.cast(tf.transpose(grid_batch, [2, 0, 1]), tf.int32)
        diff_x = grid_xf + warp_xf - tf.cast(coord_x, tf.float32)
        ndiff_x = 1 - diff_x
        diff_y = grid_yf + warp_yf - tf.cast(coord_y, tf.float32)
        ndiff_y = 1 - diff_y
        grid.append(tf.stack([coord_batch, coord_y, coord_x], axis=-1))  # upper-left
        grid.append(tf.stack([coord_batch, coord_y, coord_x + 1], axis=-1))  # upper-right
        grid.append(tf.stack([coord_batch, coord_y + 1, coord_x + 1], axis=-1))  # lower-right
        grid.append(tf.stack([coord_batch, coord_y + 1, coord_x], axis=-1))  # lower-left
        scale = [diff_x, ndiff_x, diff_y, ndiff_y]
        return grid, [tf.expand_dims(s, axis=-1) for s in scale]

    def _motion_compensate(self, x):
        """warp frame x

        :param x: is the input frame to be warped
        :return: a tensor with same shape as x
        """
        assert x.shape[-1] == 3  # [H, W, (gray, coordx, coordy)]
        grid, scale = self._gen_grid(tf.shape(x), x[..., 1:])
        inp = x[..., 0:1]
        y = tf.zeros_like(inp)
        # sample x to y according to self.grid
        y += tf.gather_nd(inp, grid[0]) * scale[1] * scale[3]
        y += tf.gather_nd(inp, grid[1]) * scale[0] * scale[3]
        y += tf.gather_nd(inp, grid[2]) * scale[0] * scale[2]
        y += tf.gather_nd(inp, grid[3]) * scale[1] * scale[2]
        return y


class VESPCN(SuperResolution):

    def __init__(self, scale=3, depth=3, name='vespcn', **kwargs):
        self.name = name
        self.depth = depth
        self.spmc = Spmc()
        self.compensated_frames = []
        super(VESPCN, self).__init__(scale=scale, **kwargs)

    def build_graph(self):
        self.inputs.append(tf.placeholder(tf.uint8, shape=[None, self.depth, None, None, 1]))
        cast_inp = tf.cast(self.inputs[-1], dtype=tf.float32)
        stacked_inp = tf.split(cast_inp, self.depth, axis=1)
        stacked_inp = [tf.squeeze(i, axis=1) for i in stacked_inp]
        mid_frame = stacked_inp[(self.depth - 1) // 2]
        for f in stacked_inp:
            if f is mid_frame:
                self.compensated_frames.append(f)
                continue
            spmc_in = tf.concat([mid_frame, f], axis=-1)
            spmc_out = self.spmc(spmc_in)
            self.compensated_frames.append(spmc_out)
        assert len(self.compensated_frames) == self.depth
        nn = list()
        nn.append(tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.tanh,
                                   kernel_initializer=Utility.he_initializer))
        nn.append(tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.tanh,
                                   kernel_initializer=Utility.he_initializer))
        nn.append(tf.layers.Conv2D(self.scale[0] * self.scale[1], 3, padding='same',
                                   kernel_initializer=Utility.he_initializer))
        x = tf.concat(self.compensated_frames, axis=-1)
        for _n in nn:
            x = _n(x)
            self.trainable_weights += [_n.kernel]
        outp = Utility.pixel_shift(x, self.scale, 1)
        self.outputs.append(outp)

    def build_loss(self):
        self.label.append(tf.placeholder(tf.uint8, shape=[None, self.depth, None, None, 1]))
        mid_frame = self.compensated_frames[(self.depth - 1) // 2]

        cast_label = tf.cast(self.label[-1], tf.float32)
        mid_label = cast_label[:, (self.depth - 1) // 2, ...]
        mse = tf.reduce_mean(tf.square(self.outputs[-1] - mid_label))

        spmc_mse = []
        for f in self.compensated_frames:
            if f is mid_frame:
                continue
            spmc_mse.append(tf.reduce_mean(tf.square(mid_frame - f)))
        spmc_mse = tf.add_n(spmc_mse)

        loss = mse + spmc_mse
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.loss.append(optimizer.minimize(loss))
        self.metrics['mse'] = mse
        self.metrics['spmc/mse'] = spmc_mse

    def build_summary(self):
        tf.summary.scalar('loss/mse', self.metrics['mse'])
