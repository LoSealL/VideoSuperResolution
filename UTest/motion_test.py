"""
unit test for VSR.Framework.Motion package
"""
import os

if not os.getcwd().endswith('UTest'):
  os.chdir('UTest')
from VSR.Framework import Motion as M
from VSRTorch.Models.video import motion as MT

import tensorflow as tf
import torch
import numpy as np
from PIL import Image

TEST_FLO_FILE = './data/flying_chair/flow/0000.flo'
TEST_PNG16_FILE = './data/kitti_car/f_01.png'


def _assert_same(x, y, epsilon=1e-6):
  if isinstance(x, torch.Tensor):
    d = (x - y).abs().mean()
    assert d.cpu().numpy() <= epsilon
  elif isinstance(x, tf.Tensor):
    d = tf.reduce_mean(tf.abs(x - y))
    assert d.eval() <= epsilon


def test_open_flo():
  X = M.open_flo(TEST_FLO_FILE)
  assert X.shape == (384, 512, 2)


def test_open_png16():
  X = M.open_png16(TEST_PNG16_FILE)
  assert X.shape == (375, 1242, 3)


def test_grid():
  G = np.meshgrid(range(5), range(5))
  G = np.stack(G, -1)
  G_bar = M._grid(5, 5)[0]
  with tf.Session() as sess:
    G_bar = sess.run(G_bar)
  assert np.all(G == G_bar.transpose([1, 0, 2]))


def test_sample():
  G = np.meshgrid(range(5), range(5))
  G = np.stack(G, -1)
  G = np.expand_dims(G, 0)
  G.transpose([0, 2, 1, 3])

  X = np.random.rand(1, 5, 5, 3).astype('float32')
  X_bar = M._sample(X, G[..., 0], G[..., 1])

  with tf.Session() as sess:
    X_bar = sess.run(X_bar)
  assert np.all(X == X_bar)


def test_warp_car():
  flow = M.KITTI.open_flow(TEST_PNG16_FILE)
  car = M.open_png16('./data/kitti_car/c_11.png')
  flow = flow.reshape([1, *flow.shape])
  car = car.reshape([1, *car.shape])
  car_bar = M.warp(car, flow[..., 0], flow[..., 1], True)
  with tf.Session() as sess:
    car_bar = sess.run(car_bar)[0]
    car_bar = car_bar.astype('uint8')
    # Image.fromarray(car_bar, 'RGB').show()


def test_warp_chair():
  flow = M.open_flo(TEST_FLO_FILE)
  img1 = Image.open('./data/flying_chair/pair/0000/img1.png')
  ch0 = np.array(img1).astype('float32')
  flow = flow.reshape([1, *flow.shape])
  ch0 = np.expand_dims(ch0, 0)
  ch1 = M.warp(ch0, flow[..., 0], flow[..., 1], True)
  with tf.Session() as sess:
    ch1 = sess.run(ch1)[0]
    ch1 = ch1.astype('uint8')
    # Image.fromarray(ch1, 'RGB').show()


def test_sttn():
  f0 = torch.ones(1, 1, 8, 8) * 4
  f1 = torch.ones(1, 1, 8, 8) * 8
  f2 = torch.stack([f1, f0], dim=2)  # NCTHW
  tr = MT.STTN(padding_mode='border')
  d = torch.zeros(1, 8, 8)
  u = torch.zeros(1, 8, 8)
  v = torch.zeros(1, 8, 8)
  f3 = tr(f2, d, u, v).squeeze(2)
  _assert_same(f3, f1)
  d = torch.ones(1, 8, 8) * 2
  f4 = tr(f2, d, u, v).squeeze(2)
  _assert_same(f4, f0)


def test_sttn_permute():
  f0 = torch.ones(1, 1, 8, 8) * 4
  f1 = torch.ones(1, 1, 8, 8) * 8
  f2 = torch.stack([f1, f0], dim=1)  # NTCHW
  tr = MT.STTN([0, 2, 1, 3, 4], padding_mode='border')
  d = torch.zeros(1, 8, 8)
  u = torch.zeros(1, 8, 8)
  v = torch.zeros(1, 8, 8)
  f3 = tr(f2, d, u, v).squeeze(2)
  _assert_same(f3, f1)
  d = torch.ones(1, 8, 8) * 2
  f4 = tr(f2, d, u, v).squeeze(2)
  _assert_same(f4, f0)
