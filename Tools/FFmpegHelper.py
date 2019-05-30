#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import argparse
import subprocess
import uuid
from pathlib import Path

import tqdm

parser = argparse.ArgumentParser(
  description="Helper tool to use ffmpeg for transcoding.")
parser.add_argument("input_dir", help="root folder of the raw videos.")
parser.add_argument("output_dir", help="root folder of the targets.")
parser.add_argument("--gop", type=int, default=1,
                    help="[Enc] group of pictures (1)")
parser.add_argument("--bf", type=int, default=0, help="[Enc] # B frames (0).")
parser.add_argument("--codec", default="libx264",
                    help="[Enc] encoder codec (libx264).")
parser.add_argument("--qp", type=int, default=0,
                    help="[Enc] quality index. [0, 51]")
FLAGS = parser.parse_args()


def _check_ffmpeg():
  import shutil
  if shutil.which('ffmpeg') is None:
    raise FileNotFoundError("Couldn't find ffmpeg!")


def parse_inputfile(path):
  filename = Path(path).stem
  suffix = Path(path).suffix
  size = filename.split('_')[-1]
  _size = [int(i) for i in size.split('x')]
  assert len(_size) == 2
  return size, suffix[1:].lower()


def encode(file, work_dir):
  cmd = 'ffmpeg -f rawvideo'
  size, fmt = parse_inputfile(file)
  tmp_name = work_dir / f'{str(uuid.uuid4())}.264'
  cmd += f' -pix_fmt {fmt}'
  cmd += f' -s:v {size}'
  cmd += f' -i {str(file)}'
  cmd += f' -vcodec {FLAGS.codec}'
  cmd += f' -g {FLAGS.gop}'
  cmd += f' -bf {FLAGS.bf}'
  cmd += f' -qp {FLAGS.qp}'
  cmd += f' -f rawvideo {str(tmp_name)}'
  # print(cmd)
  subprocess.call(cmd.split(' '), stderr=subprocess.DEVNULL)
  return tmp_name, fmt


def decode(file, output_dir, name, fmt):
  cmd = f'ffmpeg -i {str(file)}'
  output_name = output_dir / f'{name}_{FLAGS.qp}.{fmt}'
  cmd += f' -f rawvideo -pix_fmt {fmt}'
  cmd += f' {str(output_name)} -y'
  # print(cmd)
  subprocess.call(cmd.split(' '), stderr=subprocess.DEVNULL)


def main():
  _check_ffmpeg()
  raw_videos = filter(lambda f: f.is_file(), Path(FLAGS.input_dir).rglob('*'))
  raw_videos = list(raw_videos)
  if not raw_videos:
    raw_videos = filter(lambda f: f.is_file(), [Path(FLAGS.input_dir)])
  tmp_dir = Path('/tmp/vsr/tools/_ffmpeg')
  tmp_dir.mkdir(exist_ok=True, parents=True)
  save_dir = Path(FLAGS.output_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  with tqdm.tqdm(sorted(raw_videos), ascii=True, unit=' video') as r:
    for fp in r:
      r.set_postfix({'name': fp.name})
      stream, fmt = encode(fp, tmp_dir)
      decode(stream, save_dir, fp.stem, fmt)


if __name__ == '__main__':
  main()
