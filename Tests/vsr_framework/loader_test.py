"""
Copyright: Wenyi Tang 2019-2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-3

Test dataloader
"""
import unittest

import numpy as np

from VSR.DataLoader.Crop import CenterCrop, RandomCrop
from VSR.DataLoader.Dataset import Dataset
from VSR.DataLoader.Loader import Loader
from VSR.Util.ImageProcess import imresize


class LoaderTest(unittest.TestCase):
    def assert_psnr(self, ret):
        x = ret[0]['hr'][0, 0]
        y = ret[0]['lr'][0, 0]
        y = imresize(y, x.shape[-1] // y.shape[-1])
        mse = np.mean((x - y) ** 2)
        psnr = np.log10(255 * 255 / (mse + 1e-8)) * 10
        self.assertGreaterEqual(psnr, 28)

    def test_simplest_loader(self):
        d = Dataset('data/set5_x2')
        ld = Loader(d, scale=2, threads=4)
        itr = ld.make_one_shot_iterator([4, 3, 4, 4], 10, True)
        self.assertEqual(len(itr), 10)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, True)
        self.assertEqual(len(itr), 10)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assert_psnr(ret)

    def test_complex_loader(self):
        d = Dataset('data').use_like_video().include_reg('hr/xiuxian')
        hr = d.compile()
        d = Dataset('data').use_like_video().include_reg('lr/xiuxian')
        lr = d.compile()
        ld = Loader(hr, lr, threads=4)
        ld.image_augmentation()
        ld.cropper(RandomCrop(2))
        itr = ld.make_one_shot_iterator([4, 3, 3, 16, 16], 10, shuffle=True)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assert_psnr(ret)

    def test_memory_limit(self):
        d = Dataset('data/')
        d = d.include('*.png')
        data = d.compile()
        ld = Loader(data, data, threads=4)
        ld.cropper(RandomCrop(1))
        ld.image_augmentation()
        itr = ld.make_one_shot_iterator(
            [4, 3, 16, 16], 10, True, data.capacity / 2)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assert_psnr(ret)
        itr = ld.make_one_shot_iterator(
            [4, 3, 16, 16], 10, True, data.capacity / 2)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assert_psnr(ret)

    def test_no_shuffle(self):
        d = Dataset('data/').include('*.png')
        data = d.compile()
        ld = Loader(data, data, threads=4)
        ld.cropper(CenterCrop(1))
        itr1 = ld.make_one_shot_iterator([1, 3, 16, 16], -1, False)
        ret1 = list(itr1)
        self.assertEqual(len(ret1), 16)
        self.assert_psnr(ret1)
        itr2 = ld.make_one_shot_iterator([1, 3, 16, 16], -1, False)
        ret2 = list(itr2)
        self.assertEqual(len(ret2), 16)
        self.assert_psnr(ret2)
        for x, y in zip(ret1, ret2):
            self.assertTrue(np.all((x['hr'] - y['hr']) < 1e-4))

    def test_no_shuffle_limit(self):
        d = Dataset('data/')
        d = d.include('*.png')
        data = d.compile()
        ld = Loader(data, data, threads=4)
        ld.cropper(RandomCrop(1))
        ld.image_augmentation()
        itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, False,
                                        data.capacity / 2)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assert_psnr(ret)
        itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, False,
                                        data.capacity / 2)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assert_psnr(ret)

    def test_auto_deduce_shape(self):
        d = Dataset('data').include_reg('set5')
        ld = Loader(d, scale=1)
        itr = ld.make_one_shot_iterator([1, -1, -1, -1], -1)
        ret = list(itr)
        self.assertEqual(len(ret), 5)
        self.assert_psnr(ret)

    def test_load_empty_data(self):
        d = Dataset('not-found')
        ld = Loader(d, scale=1)
        itr = ld.make_one_shot_iterator([1, -1, -1, -1], -1)
        self.assertEqual(len(list(itr)), 0)
        itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10)
        ret = list(itr)
        self.assertEqual(len(ret), 10)
        self.assertFalse(ret[0]['hr'])
        self.assertFalse(ret[0]['lr'])
        self.assertFalse(ret[0]['name'])
