#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 13

import unittest

import numpy as np

from VSR.Util.Math import (anisotropic_gaussian_kernel, gaussian_kernel)

_K1 = gaussian_kernel(15, 2)
_K2 = anisotropic_gaussian_kernel(15, 1, 5, 3)


class ImFilter(unittest.TestCase):
  # For ones([4, 4])
  y_gold = np.array([
    [0.3151, 0.3776, 0.3776, 0.3151],
    [0.3776, 0.4524, 0.4524, 0.3776],
    [0.3776, 0.4524, 0.4524, 0.3776],
    [0.3151, 0.3776, 0.3776, 0.3151]
  ])
  z_gold = np.array([
    [0.3391, 0.3950, 0.3774, 0.2950],
    [0.3850, 0.4627, 0.4557, 0.3677],
    [0.3677, 0.4557, 0.4627, 0.3850],
    [0.2950, 0.3774, 0.3950, 0.3391]
  ])

  def test_torch(self):
    import torch
    from VSR.Backend.Torch.Util.Utility import imfilter

    tk1 = torch.tensor(_K1, dtype=torch.float32)
    tk2 = torch.tensor(_K2, dtype=torch.float32)
    x = torch.ones(2, 3, 4, 4, dtype=torch.float32)
    y = imfilter(x, tk1)
    z = imfilter(x, torch.stack([tk1, tk2]))
    y_ = y.detach().numpy()
    z_ = z.detach().numpy()

    self.assertTrue(np.all(y_[0] == z_[0]))
    self.assertTrue(np.all(np.abs(y_[0, 0] - self.y_gold) <= 1e-4))
    self.assertTrue(np.all(np.abs(z_[1, 0] - self.z_gold) <= 1e-4))

  def test_tf(self):
    import tensorflow as tf
    from VSR.Backend.TF.Util import imfilter

    tk1 = tf.constant(_K1, dtype=tf.float32)
    tk2 = tf.constant(_K2, dtype=tf.float32)
    x = tf.ones([2, 4, 4, 3], dtype=tf.float32)
    y = imfilter(x, tk1)
    z = imfilter(x, tk2)
    with tf.Session() as sess:
      y_, z_ = sess.run([y, z])

      self.assertTrue(np.all(np.abs(y_[0, ..., 0] - self.y_gold) <= 1e-4))
      self.assertTrue(np.all(np.abs(z_[0, ..., 0] - self.z_gold) <= 1e-4))


if __name__ == '__main__':
  unittest.main()
