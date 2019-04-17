#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import argparse
import multiprocessing as mp
import re
from pathlib import Path

import tqdm

from VSR.DataLoader.VirtualFile import RawFile

parser = argparse.ArgumentParser(
  description="Convert a raw video to a folder of images.")
parser.add_argument("input_dir", help="root folder of raw videos.")
parser.add_argument("output_dir", help="root folder of images")
parser.add_argument("--width", type=int, default=0,
                    help="default 0. Auto detect from file name.")
parser.add_argument("--height", type=int, default=0,
                    help="default 0. Auto detect from file name.")
parser.add_argument("--overwrite", action='store_true',
                    help="overwrite existing files with same name.")
FLAGS = parser.parse_args()


def guess_file_size(file):
  name = file.name
  rept = re.compile("\d+[xX]\d+")
  for i in name.split('_'):
    ans = re.findall(rept, i)
    if ans:
      size = ans[0].lower().split('x')
      return int(size[0]), int(size[1])
  return -1, -1


def parse_format(fmt):
  if fmt.upper() in ('YUV', 'YUV420P'):
    return 'YV12'
  return fmt.upper()


def encode(file, save_dir):
  w = FLAGS.width
  h = FLAGS.height
  if w == 0 or h == 0:
    w, h = guess_file_size(file)
  if w <= 0 or h <= 0:
    raise ValueError("No width/height can be retrieved!")
  fmt = file.suffix[1:]
  fmt = parse_format(fmt)
  save_dir /= file.stem
  save_dir.mkdir(exist_ok=FLAGS.overwrite, parents=True)
  fd = RawFile(file, fmt, [w, h])
  frames = fd.read_frame(fd.frames)
  for i, f in enumerate(frames):
    f.convert('RGB').save(f'{str(save_dir)}/{i:05d}.png')
  return file.stem


def main():
  input_dir = Path(FLAGS.input_dir)
  if input_dir.is_dir():
    raw_videos = filter(lambda f: f.is_file(), input_dir.rglob('*'))
  else:
    assert input_dir.is_file()
    raw_videos = [input_dir]
  raw_videos = sorted(raw_videos)
  save_dir = Path(FLAGS.output_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  pool = mp.pool.ThreadPool()
  results = []
  for fp in raw_videos:
    results.append(pool.apply_async(encode, (fp, save_dir)))
  with tqdm.tqdm(results, ascii=True, unit='image') as r:
    for i in r:
      name = i.get()
      r.set_postfix({'name': name})
  pool.close()
  pool.join()


if __name__ == '__main__':
  main()
