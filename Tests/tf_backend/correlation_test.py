"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com

Test correlation between tensors
"""
import unittest

import tensorflow as tf

from VSR.Backend.TF.Util import _make_displacement, _make_vector, correlation


class CorrelationTest(unittest.TestCase):
    @staticmethod
    def constant():
        return tf.constant([[
            [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]],
            [[4, 4.1, 4.2], [5, 5.1, 5.2], [6, 6.1, 6.2]],
            [[7, 7.1, 7.2], [8, 8.1, 8.2], [9, 9.1, 9.2]]
        ]], 'float32')

    def test_correlation(self):
        with tf.Session() as sess:
            vec = _make_vector(self.constant()).eval()
            disp = _make_displacement(self.constant()).eval()
            x = self.constant()
            corr = correlation(x, x, 3, 1).eval()
            x = tf.ones([1, 5, 5, 1], 'float32')
            corr_stride = correlation(x, x, 3, 2, 2, 2).eval()
        self.assertEqual(vec.shape, [1, 3, 3, 27])
        self.assertEqual(disp.shape, [1, 3, 3, 1])
        self.assertEqual(corr.shape, [1, 3, 3, 4])
        self.assertEqual(corr_stride.shape, [1, 3, 3, 4])
