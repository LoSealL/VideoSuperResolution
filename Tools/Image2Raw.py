#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/4 下午2:42

import argparse
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

from Tools.Misc import YUVConverter

_YUV_COLOR = ("NV12", "NV21", "YV12", "YV21")
_RGB_COLOR = ("RGBA",)

parser = argparse.ArgumentParser(
  usage=r'''python Image2Raw.py input_dir output_dir [--options]''',
  description=r'''Convert a folder of images to raw video format (FOURCC).''')
parser.add_argument("input_dir", help="root of the input folder, "
                                      "the leaf will be the last child-folder "
                                      "containing individual image files.")
parser.add_argument("output_dir", help="root of the output path.")
parser.add_argument("--color_fmt", choices=_YUV_COLOR + _RGB_COLOR,
                    default='NV12',
                    help="output color format")
FLAGS = parser.parse_args()


def parse_video_clips(path):
  _path = Path(path)
  if not _path.exists():
    raise FileNotFoundError(f"{path} doesn't exist!!")

  files = _path.rglob('*')
  parents = set(f.parent for f in filter(lambda f: f.is_file(), files))
  return parents


def read_video_frames(path):
  _path = Path(path)
  files = sorted(_path.glob('*'))
  images = [Image.open(f) for f in files]
  if FLAGS.color_fmt in _YUV_COLOR:
    images = [m.convert('YCbCr') for m in images]
  mat = np.stack(images).transpose([0, 3, 1, 2])  # [NCHW]
  return {
    'data': mat,
    'length': mat.shape[0],
    'name': path.stem,
    'width': mat.shape[3],
    'height': mat.shape[2],
  }


def main():
  videos = parse_video_clips(FLAGS.input_dir)
  root = Path(FLAGS.output_dir)
  root.mkdir(exist_ok=True, parents=True)
  print(f" [*] Total videos found: {len(videos)}.")
  with tqdm.tqdm(videos, ascii=True, unit=' video') as r:
    for fp in r:
      data = read_video_frames(fp)
      r.set_postfix({"name": data['name']})
      if FLAGS.color_fmt in _YUV_COLOR:
        cvt = YUVConverter(data['data'])
        bytes = cvt.to(FLAGS.color_fmt).getbuffer().tobytes()
      else:
        raise NotImplementedError
      nm = f"{data['name']}_{cvt.width}x{cvt.height}.{FLAGS.color_fmt}"
      save_path = root / nm
      with save_path.open('wb') as fd:
        fd.write(bytes)


if __name__ == '__main__':
  main()
  exit(0)
