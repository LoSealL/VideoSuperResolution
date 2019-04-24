#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/22 下午3:54

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from VSR.DataLoader.VirtualFile import ImageFile

parser = argparse.ArgumentParser(description="Sequential Visualizer")
parser.add_argument("input_dir", help="input folder")
parser.add_argument("output", help="output file path")
parser.add_argument("--row", '-r', type=int, default=-1, help="row number")
parser.add_argument("--col", '-c', type=int, default=-1, help="column number")
parser.add_argument("--zoom", nargs=4, type=int, help='zoom coordinate')
parser.add_argument("--compose_id", type=int, default=None)


def main():
  flags = parser.parse_args()
  fp = ImageFile(flags.input_dir)
  frames = np.stack(fp.read_frame(fp.frames))
  if flags.zoom:
    frames = frames[:, flags.zoom[1]: flags.zoom[3],
             flags.zoom[0]: flags.zoom[2]]
  savedir = Path(flags.output)
  if flags.compose_id is not None:
    compose = frames[flags.compose_id]
  else:
    compose = None
  if 0 <= flags.row < frames.shape[1]:
    sliced = frames[:, flags.row]
    if compose is not None:
      sliced = np.concatenate([compose, sliced], axis=0)
    if savedir.is_dir():
      savedir.mkdir(exist_ok=True, parents=True)
      savedir /= f'{fp.name}_slice_row{flags.row}.png'
  elif 0 <= flags.col < frames.shape[2]:
    sliced = frames[:, :, flags.col]
    sliced = np.transpose(sliced, [1, 0, 2])
    if compose:
      sliced = np.concatenate([compose, sliced], axis=1)
    if savedir.is_dir():
      savedir.mkdir(exist_ok=True, parents=True)
      savedir /= f'{fp.name}_slice_col{flags.col}.png'
  Image.fromarray(sliced, 'RGB').save(str(savedir))


if __name__ == '__main__':
  main()
