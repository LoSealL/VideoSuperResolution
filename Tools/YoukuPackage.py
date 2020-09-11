"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-5-22

Utility for Youku SR competition
"""
import argparse
import os
import subprocess
from pathlib import Path

import tqdm

parser = argparse.ArgumentParser(description="Youku VSR Packager")
parser.add_argument("-i", "--input_dir")
parser.add_argument("-o", "--output_dir")
parser.add_argument("--full_percentage", default=0.1)
parser.add_argument("--extract_interval", default=25)
parser.add_argument("-v", action='store_true', help="Show debug information")
FLAGS = parser.parse_args()


def _check_ffmpeg():
    import shutil
    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError("Couldn't find ffmpeg!")


def gen_y4m_full(url, num):
    output = f'{FLAGS.output_dir}/Youku_{num}_h_Res.y4m'
    cmd = f'ffmpeg -i {str(url)} -pix_fmt yuv420p -vsync 0 {str(output)} -y'
    if FLAGS.v:
        print(cmd)
    subprocess.call(cmd, stderr=subprocess.DEVNULL, shell=True)


def gen_y4m_part(url, num):
    output = f'{FLAGS.output_dir}/Youku_{num}_h_Sub25_Res.y4m'
    cmd = f'ffmpeg -i {str(url)} -pix_fmt yuv420p '
    cmd += f"-vf select='not(mod(n\\,{FLAGS.extract_interval}))' "
    cmd += f"-vsync 0 {str(output)} -y"
    if FLAGS.v:
        print(cmd)
    subprocess.call(cmd, stderr=subprocess.DEVNULL, shell=True)


def parse_video_clips(path):
    _path = Path(path)
    if not _path.exists():
        raise FileNotFoundError(f"{path} doesn't exist!!")

    files = _path.rglob('*')
    parents = set(f.parent for f in filter(lambda f: f.is_file(), files))
    return sorted(parents)


def parse_url(path):
    _path = Path(path)
    if not _path.exists():
        raise FileNotFoundError(f"{path} doesn't exist!!")
    files = filter(lambda f: f.is_file(), _path.glob('*'))
    files = sorted(files)
    for i, fp in enumerate(files):
        target = _path / f'frames_{i:04d}{fp.suffix}'
        fp.rename(target)
        assert target.exists()
    return _path / f'frames_%04d{fp.suffix}'


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
    with tqdm.tqdm(videos, ascii=True, unit=' video') as r:
        for i, fp in enumerate(r):
            _, num, _ = fp.name.split('_')
            url = parse_url(fp)
            if i < FLAGS.full_percentage * len(videos):
                gen_y4m_full(url, num)
            else:
                gen_y4m_part(url, num)
            r.set_postfix({"name": fp.stem})
    zip(root)


if __name__ == '__main__':
    main()
