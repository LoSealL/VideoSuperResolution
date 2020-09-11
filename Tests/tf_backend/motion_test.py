"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

unit test for VSR.Framework.Motion package
"""
import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

from VSR.Backend.TF.Framework import Motion as M
from VSR.DataLoader.FloDecoder import KITTI, open_flo

TEST_FLO_FILE = './data/flying_chair/flow/0000.flo'
TEST_PNG16_FILE = './data/kitti_car/f_01.png'


class MotionTest(unittest.TestCase):
    def _assert_same(self, x, y, epsilon=1e-6):
        d = tf.reduce_mean(tf.abs(x - y))
        self.assertLessEqual(d.eval(), epsilon)

    def test_grid(self):
        G = np.meshgrid(range(5), range(5))
        G = np.stack(G, -1)
        G_bar = M._grid(5, 5)[0]
        with tf.Session() as sess:
            G_bar = sess.run(G_bar)
        self.assertTrue(np.all(G == G_bar.transpose([1, 0, 2])))

    def test_sample(self):
        G = np.meshgrid(range(5), range(5))
        G = np.stack(G, -1)
        G = np.expand_dims(G, 0)
        G.transpose([0, 2, 1, 3])

        X = np.random.rand(1, 5, 5, 3).astype('float32')
        X_bar = M._sample(X, G[..., 0], G[..., 1])

        with tf.Session() as sess:
            X_bar = sess.run(X_bar)
        self.assertTrue(np.all(X == X_bar))

    def test_warp_car(self):
        flow = KITTI.open_flow(TEST_PNG16_FILE)
        car = KITTI.open_png16('./data/kitti_car/c_11.png')
        flow = flow.reshape([1, *flow.shape])
        car = car.reshape([1, *car.shape])
        car_bar = M.warp(car, flow[..., 0], flow[..., 1], True)
        with tf.Session() as sess:
            car_bar = sess.run(car_bar)[0]
            car_bar = car_bar.astype('uint8')
            # Image.fromarray(car_bar, 'RGB').show()

    def test_warp_chair(self):
        flow = open_flo(TEST_FLO_FILE)
        img1 = Image.open('./data/flying_chair/pair/0000/img1.png')
        ch0 = np.array(img1).astype('float32')
        flow = flow.reshape([1, *flow.shape])
        ch0 = np.expand_dims(ch0, 0)
        ch1 = M.warp(ch0, flow[..., 0], flow[..., 1], True)
        with tf.Session() as sess:
            ch1 = sess.run(ch1)[0]
            ch1 = ch1.astype('uint8')
            # Image.fromarray(ch1, 'RGB').show()
