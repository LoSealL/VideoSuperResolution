import os

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

import tensorflow as tf
tf.enable_eager_execution()

import torchvision
import numpy as np
from PIL import Image
from VSR.Util import imread, rgb_to_yuv

URL = 'data/set5_x2/img_001_SRF_2_LR.png'


def test_rgb2yuv():
  img = imread(URL)
  img = img.astype('float32')

  yuv = rgb_to_yuv(img, 255, 'matlab')
  # should have the same shape
  assert yuv.shape == img.shape


def test_resize_upsample():
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
      assert np.all(np.abs(y[16:-16, 16:-16] - gt[16:-16, 16:-16]) < 1.0e-2), \
        f"Scale: {ss}. Mode: {X.mode}"


def test_resize_downsample():
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
      assert np.all(np.abs(y[8:-8, 8:-8] - gt[8:-8, 8:-8]) < 1.0e-2), \
        f"Scale: {ss}. Mode: {X.mode}"



def test_resize_upsample_VSRT():
  from VSR.Backend.Torch.Util.Utility import upsample
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
      assert np.all(
          np.abs(y[..., 16:-16, 16:-16] - gt[..., 16:-16, 16:-16]) < 1.0e-2), \
        f"Scale: {ss}. Mode: {X.mode}"


def test_resize_downsample_VSRT():
  from VSR.Backend.Torch.Util.Utility import downsample
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
      assert np.all(np.abs(y[..., 8:-8, 8:-8] - gt[..., 8:-8, 8:-8]) < 1.0e-2), \
        f"Scale: {ss}. Mode: {X.mode}"
