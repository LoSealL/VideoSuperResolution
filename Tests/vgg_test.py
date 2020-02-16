#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 16

import os
import unittest

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

import numpy as np
from PIL import Image

from VSR.Backend.TF.Util import Vgg

URL = 'data/set5_x2/img_001_SRF_2_LR.png'
image_boy = np.asarray(Image.open(URL))


class VggTest(unittest.TestCase):
  def import_tf(self):
    import tensorflow as tf
    return tf

  def test_vgg16(self):
    vgg = Vgg(False, vgg=Vgg.VGG16)
    x = np.random.normal(size=[16, 128, 128, 3])
    y = vgg(x)
    self.assertEqual(y.shape, (16,))

  def test_vgg19(self):
    vgg = Vgg(False, vgg=Vgg.VGG19)
    x = np.random.normal(size=[16, 128, 128, 3])
    y = vgg(x, 'block2_conv2')
    self.assertEqual(y.shape, (16, 64, 64, 128))

  def test_vgg_classify(self):
    vgg16 = Vgg(True, vgg=Vgg.VGG16)
    vgg19 = Vgg(True, vgg=Vgg.VGG19)
    x = np.expand_dims(image_boy, 0)
    y1 = vgg16(x)
    y2 = vgg19(x)
    tf = self.import_tf()
    with tf.Session() as sess:
      y1, y2 = sess.run([y1, y2])
      self.assertEqual(y2[0].tolist().index(y2.max()),
                       y1[0].tolist().index(y1.max()))

  def test_multiple_call(self):
    vgg1 = Vgg(False, vgg=Vgg.VGG16)
    vgg2 = Vgg(False, vgg=Vgg.VGG16)
    x = np.expand_dims(image_boy, 0)
    y1 = vgg1(x)
    y2 = vgg2(x)
    y3 = vgg2(x.copy())
    tf = self.import_tf()
    with tf.Session() as sess:
      y1, y2, y3 = sess.run([y1, y2, y3])
      self.assertEqual(y1, y2)
      self.assertEqual(y2, y3)


if __name__ == '__main__':
  unittest.main()
