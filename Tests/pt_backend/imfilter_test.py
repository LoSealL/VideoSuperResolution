"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-13

Gaussian blur filter test
"""
import unittest

import numpy as np
import torch

from VSR.Backend.Torch.Util.Utility import imfilter
from VSR.Util.Math import anisotropic_gaussian_kernel, gaussian_kernel

_K1 = gaussian_kernel(15, 2)
_K2 = anisotropic_gaussian_kernel(15, 1, 5, 3)


class ImFilterTest(unittest.TestCase):
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
