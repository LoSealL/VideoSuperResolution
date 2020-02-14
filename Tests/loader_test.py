#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午8:28

import os

import numpy as np

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')
from VSR.DataLoader.Loader import Loader
from VSR.DataLoader.Dataset import Dataset
from VSR.DataLoader.Crop import RandomCrop
from VSR.Util.ImageProcess import imresize


def assert_psnr(ret):
  x = ret[0]['hr'][0, 0]
  y = ret[0]['lr'][0, 0]
  y = imresize(y, x.shape[-1] // y.shape[-1])
  mse = np.mean((x - y) ** 2)
  psnr = np.log10(255 * 255 / mse) * 10
  assert psnr > 28


def test_simplest_loader():
  d = Dataset('data/set5_x2')
  ld = Loader(d, scale=2, threads=4)
  itr = ld.make_one_shot_iterator([4, 3, 4, 4], 10, True)
  assert len(itr) is 10
  ret = list(itr)
  assert len(ret) is 10
  itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, True)
  assert len(itr) is 10
  ret = list(itr)
  assert len(ret) is 10
  assert_psnr(ret)


def test_complex_loader():
  d = Dataset('data').use_like_video().include_reg('hr/xiuxian')
  hr = d.compile()
  d = Dataset('data').use_like_video().include_reg('lr/xiuxian')
  lr = d.compile()
  ld = Loader(hr, lr, threads=4)
  ld.image_augmentation()
  ld.cropper(RandomCrop(2))
  itr = ld.make_one_shot_iterator([4, 3, 3, 16, 16], 10, shuffle=True)
  ret = list(itr)
  assert len(ret) is 10
  assert_psnr(ret)


def test_memory_limit():
  d = Dataset('data/')
  d = d.include('*.png')
  data = d.compile()
  ld = Loader(data, data, threads=4)
  ld.cropper(RandomCrop(1))
  ld.image_augmentation()
  itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, True, data.capacity / 2)
  ret = list(itr)
  assert len(ret) is 10
  assert_psnr(ret)
  itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, True, data.capacity / 2)
  ret = list(itr)
  assert len(ret) is 10
  assert_psnr(ret)


def test_no_shuffle():
  d = Dataset('data/')
  d = d.include('*.png')
  data = d.compile()
  ld = Loader(data, data, threads=4)
  ld.cropper(RandomCrop(1))
  ld.image_augmentation()
  itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, False, data.capacity / 2)
  ret = list(itr)
  assert len(ret) is 10
  assert_psnr(ret)
  itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10, False, data.capacity / 2)
  ret = list(itr)
  assert len(ret) is 10
  assert_psnr(ret)


def test_auto_deduce_shape():
  d = Dataset('data').include_reg('set5')
  ld = Loader(d, scale=1)
  itr = ld.make_one_shot_iterator([1, -1, -1, -1], -1)
  ret = list(itr)
  assert len(ret) is 5
  assert_psnr(ret)


def test_load_empty_data():
  d = Dataset('not-found')
  ld = Loader(d, scale=1)
  itr = ld.make_one_shot_iterator([1, -1, -1, -1], -1)
  assert len(list(itr)) is 0
  itr = ld.make_one_shot_iterator([4, 3, 16, 16], 10)
  ret = list(itr)
  assert len(ret) is 10
  assert not ret[0]['hr']
  assert not ret[0]['lr']
  assert not ret[0]['name']


if __name__ == '__main__':
  test_simplest_loader()
  test_complex_loader()
  test_memory_limit()
  test_no_shuffle()
  test_auto_deduce_shape()
  test_load_empty_data()
