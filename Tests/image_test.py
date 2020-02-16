import os
import unittest

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

import numpy as np
from PIL import Image
from VSR.Util import imread, rgb_to_yuv
from VSR.Backend import BACKEND

URL = 'data/set5_x2/img_001_SRF_2_LR.png'


class ImageTest(unittest.TestCase):
  def psnr(self, x: np.ndarray, y: np.ndarray):
    mse = np.mean((x - y) ** 2)
    if mse == 0:
      return np.inf
    psnr = np.log10(255 * 255 / mse) * 10
    return psnr

  def test_rgb2yuv(self):
    img = imread(URL, mode='RGB').astype('float32')
    yuv = rgb_to_yuv(img, 255, 'matlab')
    yuv_ref = imread(URL, mode='YCbCr').astype('float32')
    # should have the same shape
    self.assertEqual(yuv.shape, img.shape)
    self.assertGreaterEqual(self.psnr(yuv, yuv_ref), 30)

  def test_resize_upsample_tf(self):
    if BACKEND != 'tensorflow':
      return
    import tensorflow as tf
    tf.enable_eager_execution()
    from VSR.Backend.TF.Util import upsample

    Im = Image.open(URL)
    for X in [Im, Im.convert('L')]:
      w = X.width
      h = X.height
      for ss in [2, 3, 4, 5, 6]:
        GT = X.resize([w * ss, h * ss], Image.BICUBIC)
        gt = np.asarray(GT, dtype='float32') / 255
        x = tf.constant(np.asarray(X), dtype='float32') / 255
        y = upsample(x, ss).numpy().clip(0, 1)
        self.assertGreaterEqual(self.psnr(y, gt), 30, f"{X.mode}, {ss}")

  def test_resize_downsample_tf(self):
    if BACKEND != 'tensorflow':
      return
    import tensorflow as tf
    tf.enable_eager_execution()
    from VSR.Backend.TF.Util import downsample

    Im = Image.open(URL)
    for X in [Im, Im.convert('L')]:
      w = X.width
      h = X.height
      for ss in [2, 4, 6, 8]:
        w_ = w - w % ss
        h_ = h - h % ss
        X = X.crop([0, 0, w_, h_])
        GT = X.resize([w_ // ss, h_ // ss], Image.BICUBIC)
        gt = np.asarray(GT, dtype='float32') / 255
        x = tf.constant(np.asarray(X), dtype='float32') / 255
        y = downsample(x, ss).numpy().clip(0, 1)
        self.assertGreaterEqual(self.psnr(y, gt), 30, f"{X.mode}, {ss}")

  def test_resize_upsample_torch(self):
    if BACKEND != 'pytorch':
      return
    from VSR.Backend.Torch.Util.Utility import upsample
    import torchvision

    Im = Image.open(URL)
    trans = torchvision.transforms.ToTensor()
    for X in [Im, Im.convert('L')]:
      w = X.width
      h = X.height
      for ss in [2, 3, 4, 5, 6]:
        GT = X.resize([w * ss, h * ss], Image.BICUBIC)
        gt = trans(GT).numpy()
        x = trans(X)
        y = upsample(x, ss).numpy().clip(0, 1)
        self.assertGreaterEqual(self.psnr(y, gt), 30, f"{X.mode}, {ss}")

  def test_resize_downsample_torch(self):
    if BACKEND != 'pytorch':
      return
    from VSR.Backend.Torch.Util.Utility import downsample
    import torchvision

    Im = Image.open(URL)
    trans = torchvision.transforms.ToTensor()
    for X in [Im, Im.convert('L')]:
      w = X.width
      h = X.height
      for ss in [2, 4, 6, 8]:
        w_ = w - w % ss
        h_ = h - h % ss
        X = X.crop([0, 0, w_, h_])
        GT = X.resize([w_ // ss, h_ // ss], Image.BICUBIC)
        gt = trans(GT).numpy()
        x = trans(X)
        y = downsample(x, ss).numpy().clip(0, 1)
        self.assertGreaterEqual(self.psnr(y, gt), 30, f"{X.mode}, {ss}")


if __name__ == '__main__':
  unittest.main()
