"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

Image scaling test
"""
import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

from VSR.Backend.TF.Util import downsample, upsample
from VSR.Util import imread, rgb_to_yuv

tf.enable_eager_execution()


URL = 'data/set5_x2/img_001_SRF_2_LR.png'


class ImageTest(unittest.TestCase):
    @staticmethod
    def psnr(x: np.ndarray, y: np.ndarray):
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
        Im = Image.open(URL)
        for X in [Im, Im.convert('L')]:
            w = X.width
            h = X.height
            for ss in [2, 3, 4, 5, 6]:
                GT = X.resize([w * ss, h * ss], Image.BICUBIC)
                gt = np.asarray(GT, dtype='float32') / 255
                x = tf.constant(np.asarray(X), dtype='float32') / 255
                y = upsample(x, ss).numpy().clip(0, 1)
                self.assertGreaterEqual(
                    self.psnr(y, gt), 30, f"{X.mode}, {ss}")

    def test_resize_downsample_tf(self):
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
                self.assertGreaterEqual(
                    self.psnr(y, gt), 30, f"{X.mode}, {ss}")
