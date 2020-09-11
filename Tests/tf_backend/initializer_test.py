"""
Copyright: Wenyi Tang 2019-2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-3-5

Reimplement torch's initizlier to tf
"""
import unittest

import numpy as np
import tensorflow as tf

from VSR.Backend.TF.Util import TorchInitializer

tf.enable_eager_execution()


class InitializerTest(unittest.TestCase):
    def test_torch_initializer(self):
        x = np.ones([4, 16, 16, 16], np.float32)
        c2dtf = tf.layers.Conv2D(16, 3, padding='same',
                                 kernel_initializer=TorchInitializer(),
                                 bias_initializer=TorchInitializer(9 * 16))
        c2dtf.build(x.shape)
        w1 = c2dtf.kernel
        y1 = c2dtf.apply(x)

        # c2dnn = nn.Conv2d(16, 16, 3, padding=1)
        # w2 = c2dnn.weight
        # TODO: how to test distribution?
        assert True
