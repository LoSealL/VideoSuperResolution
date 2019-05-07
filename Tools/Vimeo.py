#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/6 下午1:39

import argparse
import shutil
from pathlib import Path

import h5py
import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Vimeo dataset splitter")
parser.add_argument("vimeo_dir", help="vimeo data path")
parser.add_argument("--type", help="vimeo data type", default='sep')
parser.add_argument("--hdf", action='store_true', help="export HDF5 dataset")
# To use vimeo HDF file, you need "vimeo" parser.
group_hdf = parser.add_argument_group("HDF5")
group_hdf.add_argument("--compression", type=int, default=None)
group_hdf.add_argument("--data_format",
                       choices=('channels_first', 'channels_last'),
                       default='channels_first',
                       help="data format (default: CHW)")
flags = parser.parse_args()


def split_sep(root: Path):
  testlist = root / 'sep_testlist.txt'
  trainlist = root / 'sep_trainlist.txt'
  if not testlist.exists() or not trainlist.exists():
    raise RuntimeError("[!] Can't find separating text files!")
  # copy destination
  (root / 'train').mkdir(exist_ok=True, parents=True)
  (root / 'test').mkdir(exist_ok=True, parents=True)
  with testlist.open('r') as fd:
    n = fd.readline()[:-1]
    test_cnt = 0
    while n:
      src = root / 'sequences' / n
      dst = root / 'test' / f'{src.parent.stem}_{src.stem}'
      if not dst.exists():
        shutil.copytree(src, dst)
        print(f"{src.stem} --> {dst.stem}")
      n = fd.readline()[:-1]
      test_cnt += 1
  with trainlist.open('r') as fd:
    n = fd.readline()[:-1]
    train_cnt = 0
    while n:
      src = root / 'sequences' / n
      dst = root / 'train' / f'{src.parent.stem}_{src.stem}'
      if not dst.exists():
        shutil.copytree(src, dst)
        print(f"{src.stem} --> {dst.stem}")
      n = fd.readline()[:-1]
      train_cnt += 1
  print(f"Total test videos: {test_cnt}")
  print(f"Total train videos: {train_cnt}")


def make_hdf(root: Path):
  test_dir = root / 'test'
  train_dir = root / 'train'
  test_shape = (7824, 7, 256, 448, 3)
  train_shape = (64612, 7, 256, 448, 3)
  if flags.data_format == 'channels_first':
    test_shape = (7824, 7, 3, 256, 448)
    train_shape = (64612, 7, 3, 256, 448)
  with h5py.File(str(root / 'test.hdf'), 'w') as hdf:
    hdf.attrs['author'] = 'LoSealL'
    hdf.attrs['email'] = 'wenyi.tang@intel.com'
    hdf.attrs['data_format'] = flags.data_format
    data = hdf.create_dataset('seq_test', dtype='uint8', shape=test_shape,
                              compression=flags.compression)
    n = 0
    for v in tqdm.tqdm(sorted(test_dir.iterdir()), desc='Test'):
      frames = sorted(v.glob('*'))
      cube = np.stack([Image.open(f) for f in frames])
      if flags.data_format == 'channels_first':
        cube = cube.transpose([0, 3, 1, 2])
      data[n] = cube
      n += 1
      del cube
  with h5py.File(str(root / 'train.hdf'), 'w') as hdf:
    hdf.attrs['author'] = 'LoSealL'
    hdf.attrs['email'] = 'wenyi.tang@intel.com'
    hdf.attrs['data_format'] = flags.data_format
    data = hdf.create_dataset('seq_train', dtype='uint8', shape=train_shape,
                              compression=flags.compression)
    n = 0
    for v in tqdm.tqdm(sorted(train_dir.iterdir()), desc='Train'):
      frames = sorted(v.glob('*'))
      cube = np.stack([Image.open(f) for f in frames])
      if flags.data_format == 'channels_first':
        cube = cube.transpose([0, 3, 1, 2])
      data[n] = cube
      n += 1
      del cube


def main():
  root = Path(flags.vimeo_dir)
  if not root.exists():
    raise RuntimeError("[!] Vimeo directory doesn't exist.")
  if flags.type == 'sep':
    split_sep(root)
  if flags.hdf:
    make_hdf(root)


if __name__ == '__main__':
  main()
