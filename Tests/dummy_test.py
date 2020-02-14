import os

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

from VSR.Util import str_to_bytes, Config

TEST_STR = ('1.3', '2kb', '3 mb', '4GB', '9Zb', '2.3pB')
ANS = (1.3, 2048.0, 3145728.0, 4294967296.0, 10625324586456701730816.0,
       2589569785738035.2)


def dummy_test_str_to_bytes():
  for t, a in zip(TEST_STR, ANS):
    ans = str_to_bytes(t)
    print(t, ans)
    assert ans == a


def dummy_test_config():
  d = Config(a=1, b=2)
  d.update(a=2, b=3)
  d.a = 9
  d.update(Config(b=6, f=5))
  d.pop('b')
  print(d)


import torchvision, torch
from torch.nn import PixelShuffle
from PIL import Image
from VSR.Backend.Torch.Models.Arch import SpaceToDim
import numpy as np


def dummy_test_space_to_depth():
  f1 = SpaceToDim(2, dim=1)
  ff = PixelShuffle(2)
  x = Image.open('data/set5_x2/img_001_SRF_2_LR.png')
  g = torchvision.transforms.ToTensor()
  h = torchvision.transforms.ToPILImage()
  z = f1(g(x).unsqueeze(0))
  y = h(ff(z)[0])
  assert np.all(np.array(x) == np.array(y))


def dummy_test_space_to_x():
  f1 = SpaceToDim(2, (1, 2), dim=3)
  x = torch.ones(1, 4, 4, 3)
  y = f1(x)
  assert y.shape == torch.Size([1, 2, 2, 12])
  f2 = SpaceToDim(2, (1, 2), dim=0)
  y = f2(x)
  assert y.shape == torch.Size([4, 2, 2, 3])


if __name__ == '__main__':
  dummy_test_space_to_depth()
  dummy_test_space_to_x()
  exit(0)
