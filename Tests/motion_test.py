"""
unit test for VSR.Framework.Motion package
"""
import os
import unittest

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')
from VSR.Backend.TF.Framework import Motion as M
from VSR.Backend.Torch.Models.Ops import Motion as MT
from VSR.DataLoader.FloDecoder import open_flo, KITTI

import tensorflow as tf
import torch
import numpy as np
from PIL import Image

TEST_FLO_FILE = './data/flying_chair/flow/0000.flo'
TEST_PNG16_FILE = './data/kitti_car/f_01.png'


class MotionTest(unittest.TestCase):
  def _assert_same(self, x, y, epsilon=1e-6):
    if isinstance(x, torch.Tensor):
      d = (x - y).abs().mean()
      self.assertLessEqual(d.cpu().numpy(), epsilon)
    elif isinstance(x, tf.Tensor):
      d = tf.reduce_mean(tf.abs(x - y))
      self.assertLessEqual(d.eval(), epsilon)

  def test_open_flo(self):
    X = open_flo(TEST_FLO_FILE)
    self.assertEqual(X.shape, (384, 512, 2))
    self.assertLessEqual(np.abs(X[..., 0]).max(), 512)
    self.assertLessEqual(np.abs(X[..., 1]).max(), 384)

  def test_open_png16(self):
    X = KITTI.open_png16(TEST_PNG16_FILE)
    self.assertEqual(X.shape, (375, 1242, 3))
    X = KITTI.open_flow(TEST_PNG16_FILE)
    self.assertEqual(X.shape, (375, 1242, 2))
    self.assertLessEqual(np.abs(X[..., 0]).max(), 1242)
    self.assertLessEqual(np.abs(X[..., 1]).max(), 375)

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

  def test_sttn(self):
    f0 = torch.ones(1, 1, 8, 8) * 4
    f1 = torch.ones(1, 1, 8, 8) * 8
    f2 = torch.stack([f1, f0], dim=2)  # NCTHW
    tr = MT.STTN(padding_mode='border')
    d = torch.zeros(1, 8, 8)
    u = torch.zeros(1, 8, 8)
    v = torch.zeros(1, 8, 8)
    f3 = tr(f2, d, u, v).squeeze(2)
    self._assert_same(f3, f1)
    d = torch.ones(1, 8, 8) * 2
    f4 = tr(f2, d, u, v).squeeze(2)
    self._assert_same(f4, f0)

  def test_sttn_permute(self):
    f0 = torch.ones(1, 1, 8, 8) * 4
    f1 = torch.ones(1, 1, 8, 8) * 8
    f2 = torch.stack([f1, f0], dim=1)  # NTCHW
    tr = MT.STTN([0, 2, 1, 3, 4], padding_mode='border')
    d = torch.zeros(1, 8, 8)
    u = torch.zeros(1, 8, 8)
    v = torch.zeros(1, 8, 8)
    f3 = tr(f2, d, u, v).squeeze(2)
    self._assert_same(f3, f1)
    d = torch.ones(1, 8, 8) * 2
    f4 = tr(f2, d, u, v).squeeze(2)
    self._assert_same(f4, f0)


if __name__ == '__main__':
  unittest.main()
