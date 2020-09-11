"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

unit test for VSR.Framework.Motion package
"""
import unittest

import numpy as np
import torch

from VSR.Backend.Torch.Models.Ops.Motion import STTN
from VSR.DataLoader.FloDecoder import KITTI, open_flo

TEST_FLO_FILE = './data/flying_chair/flow/0000.flo'
TEST_PNG16_FILE = './data/kitti_car/f_01.png'


class MotionTest(unittest.TestCase):
    def _assert_same(self, x, y, epsilon=1e-6):
        d = (x - y).abs().mean()
        self.assertLessEqual(d.cpu().numpy(), epsilon)

    def test_open_flo(self):
        X = open_flo(TEST_FLO_FILE)
        self.assertEqual(X.shape, (384, 512, 2))
        self.assertLessEqual(np.abs(X[..., 0]).max(), 512)
        self.assertLessEqual(np.abs(X[..., 1]).max(), 384)

    def test_open_png16(self):
        X = KITTI.open_png16(TEST_PNG16_FILE)
        self.assertEqual(X.shape, (375, 1242, 3))
        X = KITTI.open_flow(TEST_PNG16_FILE)
        self.assertEqual(X.shape, (375, 1242, 2))
        self.assertLessEqual(np.abs(X[..., 0]).max(), 1242)
        self.assertLessEqual(np.abs(X[..., 1]).max(), 375)

    def test_sttn(self):
        f0 = torch.ones(1, 1, 8, 8) * 4
        f1 = torch.ones(1, 1, 8, 8) * 8
        f2 = torch.stack([f1, f0], dim=2)  # NCTHW
        tr = STTN(padding_mode='border')
        d = torch.zeros(1, 8, 8)
        u = torch.zeros(1, 8, 8)
        v = torch.zeros(1, 8, 8)
        f3 = tr(f2, d, u, v).squeeze(2)
        self._assert_same(f3, f1)
        d = torch.ones(1, 8, 8) * 2
        f4 = tr(f2, d, u, v).squeeze(2)
        self._assert_same(f4, f0)

    def test_sttn_permute(self):
        f0 = torch.ones(1, 1, 8, 8) * 4
        f1 = torch.ones(1, 1, 8, 8) * 8
        f2 = torch.stack([f1, f0], dim=1)  # NTCHW
        tr = STTN([0, 2, 1, 3, 4], padding_mode='border')
        d = torch.zeros(1, 8, 8)
        u = torch.zeros(1, 8, 8)
        v = torch.zeros(1, 8, 8)
        f3 = tr(f2, d, u, v).squeeze(2)
        self._assert_same(f3, f1)
        d = torch.ones(1, 8, 8) * 2
        f4 = tr(f2, d, u, v).squeeze(2)
        self._assert_same(f4, f0)
