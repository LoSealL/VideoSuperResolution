#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 16

import os
import unittest

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

try:
  import torch
  import torchvision
  from torch.nn import PixelShuffle
  from VSR.Backend.Torch.Models.Arch import SpaceToDim
except ImportError:
  exit(0)

import numpy as np
from PIL import Image


class SpaceToDimTest(unittest.TestCase):
  def test_space_to_depth(self):
    f1 = SpaceToDim(2, dim=1)
    ff = PixelShuffle(2)
    x = Image.open('data/set5_x2/img_001_SRF_2_LR.png')
    g = torchvision.transforms.ToTensor()
    h = torchvision.transforms.ToPILImage()
    z = f1(g(x).unsqueeze(0))
    y = h(ff(z)[0])
    self.assertTrue(np.all(np.array(x) == np.array(y)))

  def dummy_test_space_to_x(self):
    f1 = SpaceToDim(2, (1, 2), dim=3)
    x = torch.ones(1, 4, 4, 3)
    y = f1(x)
    self.assertEqual(y.shape, torch.Size([1, 2, 2, 12]))
    f2 = SpaceToDim(2, (1, 2), dim=0)
    y = f2(x)
    self.assertEqual(y.shape, torch.Size([4, 2, 2, 3]))


if __name__ == '__main__':
  unittest.main()
