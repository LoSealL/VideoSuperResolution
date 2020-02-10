#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 5

import os

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from torch import nn
from VSR.Backend.TF.Util import TorchInitializer


def test_torch_initializer():
  x = np.ones([4, 16, 16, 16], np.float32)
  c2dtf = tf.layers.Conv2D(16, 3, padding='same',
                           kernel_initializer=TorchInitializer(),
                           bias_initializer=TorchInitializer(9 * 16))
  c2dtf.build(x.shape)
  w1 = c2dtf.kernel
  y1 = c2dtf.apply(x)

  c2dnn = nn.Conv2d(16, 16, 3, padding=1)
  w2 = c2dnn.weight
  assert True
