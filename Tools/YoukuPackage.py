#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/22 下午1:25

import argparse
import multiprocessing as mp
import os
import subprocess
from pathlib import Path
from functools import partial
import numpy as np
import tqdm
from PIL import Image

from Tools.Misc import YUVConverter

parser = argparse.ArgumentParser(description="Youku VSR Packager")
parser.add_argument("-i", "--input_dir")
parser.add_argument("-o", "--output_dir")
parser.add_argument("--full_percentage", default=0.1)
parser.add_argument("--extract_interval", default=25)
parser.add_argument("-v", action='store_true', help="Show debug information")
parser.add_argument("--async", action='store_true', help="multi process")
FLAGS = parser.parse_args()


def _check_ffmpeg():
  import shutil
  if shutil.which('ffmpeg') is None:
    raise FileNotFoundError("Couldn't find ffmpeg!")


def gen_y4m_full(url):
  _, num, _, size = url.name.split('_')
  output = url.parent / f'Youku_{num}_h_Res.y4m'
  cmd = f'ffmpeg -f rawvideo -pix_fmt yuv420p -s:v {size} '
  cmd += f'-i {str(url)} -vsync 0 {str(output)} -y'
  if FLAGS.v:
    print(cmd)
  subprocess.call(cmd, stderr=subprocess.DEVNULL, shell=True)
  subprocess.call(f'rm {url}', shell=True)


def gen_y4m_part(url):
  _, num, _, size = url.name.split('_')
  output = url.parent / f'Youku_{num}_h_Sub25_Res.y4m'
  cmd = f'ffmpeg -f rawvideo -pix_fmt yuv420p -s:v {size} '
  cmd += f"-i {str(url)} -vf select='not(mod(n\\,{FLAGS.extract_interval}))' "
  cmd += f"-vsync 0 {str(output)} -y"
  if FLAGS.v:
    print(cmd)
  subprocess.call(cmd, stderr=subprocess.DEVNULL, shell=True)
  subprocess.call(f'rm {url}', shell=True)


def parse_video_clips(path):
  _path = Path(path)
  if not _path.exists():
    raise FileNotFoundError(f"{path} doesn't exist!!")

  files = _path.rglob('*')
  parents = set(f.parent for f in filter(lambda f: f.is_file(), files))
  return sorted(parents)


def read_video_frames(path):
  _path = Path(path)
  files = sorted(_path.glob('*'))
  images = [Image.open(f) for f in files]
  images = [m.convert('YCbCr') for m in images]
  mat = np.stack(images).transpose([0, 3, 1, 2])  # [NCHW]
  return {
    'data': mat,
    'length': mat.shape[0],
    'name': path.stem,
    'width': mat.shape[3],
    'height': mat.shape[2],
  }


def zip(url):
  url = Path(url)
  os.chdir(url)
  cmd = 'zip youku_results.zip *.y4m'
  subprocess.call(cmd, shell=True)
  subprocess.call("rm *.y4m", shell=True)


def main():
  _check_ffmpeg()
  videos = parse_video_clips(FLAGS.input_dir)
  root = Path(FLAGS.output_dir)
  root.mkdir(exist_ok=True, parents=True)
  print(f" [*] Total videos found: {len(videos)}.")
  pool = mp.pool.ThreadPool()
  results = []

  def action(index, fp):
    data = read_video_frames(fp)
    cvt = YUVConverter(data['data'])
    bytes = cvt.to('YV12').getbuffer().tobytes()
    nm = f"{data['name']}_{cvt.width}x{cvt.height}"
    save_path = root / nm
    if FLAGS.v:
      print(save_path)
    with save_path.open('wb') as fd:
      fd.write(bytes)
    if index < len(videos) * FLAGS.full_percentage:
      gen_y4m_full(save_path)
    else:
      gen_y4m_part(save_path)
    return data['name']

  for i in enumerate(videos):
    if FLAGS.async:
      results.append(pool.apply_async(action, i))
    else:
      results.append(partial(action, index=i[0], fp=i[1]))

  with tqdm.tqdm(results, ascii=True, unit=' video') as r:
    for i in r:
      if FLAGS.async:
        name = i.get()
      else:
        name = i()
      r.set_postfix({"name": name})
  pool.close()
  pool.join()
  zip(root)


if __name__ == '__main__':
  main()
