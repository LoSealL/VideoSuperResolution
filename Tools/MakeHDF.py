"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-3

Making HDF5 dataset (experimental)
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import tqdm
from PIL import Image

__all__ = ["gather_videos_vqp", "gather_videos", "print_dataset"]

parser = argparse.ArgumentParser(description="Make HDF5 datasets")
parser.add_argument("input_dir", help="path of the input root folder.")
parser.add_argument("-o", "--output", help="output hdf file path.")
parser.add_argument("-a", "--append", action='store_true')
parser.add_argument("-t", "--task_name", choices=__all__, help="task name")
parser.add_argument("--compression", type=int, default=None)
parser.add_argument("--glob", help="glob pattern to gather files inside input."
                                   "For recursively glob, use **/*.")
parser.add_argument("--data_format",
                    choices=('channels_first', 'channels_last'),
                    default='channels_first', help="data format (default: CHW)")
FLAGS, args = parser.parse_known_args()


def make_hdf_header():
    if FLAGS.output:
        if FLAGS.append:
            fd = h5py.File(FLAGS.output, 'a')
        else:
            fd = h5py.File(FLAGS.output, 'w')
        fd.attrs['author'] = 'LoSealL'
        fd.attrs['email'] = 'wenyi.tang@intel.com'
        fd.attrs['date'] = time.strftime("%Y-%m-%d")
        fd.attrs['data_format'] = FLAGS.data_format

        return fd


def flush_hdf(fd: h5py.File):
    if isinstance(fd, h5py.File):
        fd.close()


def gather_videos_vqp(fd: h5py.File):
    """Specified for VQP"""
    root = Path(FLAGS.input_dir)
    glob = FLAGS.glob or '*'
    inputs = sorted(root.glob(glob))
    candidates = set(i.parent for i in filter(lambda f: f.is_file(), inputs))
    frames_info = {}
    for p in tqdm.tqdm(candidates):
        seq = [Image.open(f) for f in
               filter(lambda f: f.is_file(), sorted(p.rglob('*')))]
        cube = np.stack(seq)
        if FLAGS.data_format == 'channels_first':
            cube = cube.transpose([0, 3, 1, 2])
        cube = np.expand_dims(cube, 0)
        path = p.relative_to(root)
        # ugly
        path = path.parent / path.stem.split('_')[0]
        key = str(path.as_posix())
        if not key in fd:
            fd.create_dataset(key, data=cube,
                              maxshape=(52,) + cube.shape[1:],
                              compression=FLAGS.compression)
            frames_info[key] = len(seq)
        else:
            d = fd[key]
            cnt = d.shape[0] + 1
            d.resize(cnt, 0)
            d[-1] = cube
        del cube


def gather_videos(fd: h5py.File):
    """Gather videos. Video is defined in a folder containing sequential images."""
    root = Path(FLAGS.input_dir)
    glob = FLAGS.glob or '*'
    inputs = sorted(root.glob(glob))
    candidates = set(i.parent for i in filter(lambda f: f.is_file(), inputs))
    frames_info = {}
    for p in tqdm.tqdm(candidates):
        seq = [Image.open(f) for f in
               filter(lambda f: f.is_file(), sorted(p.rglob('*')))]
        cube = np.stack(seq)
        if FLAGS.data_format == 'channels_first':
            cube = cube.transpose([0, 3, 1, 2])
        path = p.relative_to(root)
        key = str(path.as_posix())
        fd.create_dataset(key, data=cube, compression=FLAGS.compression)
        frames_info[key] = len(seq)
        del cube
    fd.attrs['frames_info'] = list(frames_info.items())


def print_dataset(*args):
    def _print(name, obj):
        print(f"key: [{name}], shape: {obj.shape}")

    fd = Path(FLAGS.input_dir)
    if fd.exists():
        with h5py.File(str(fd), 'r') as fd:
            fd.visititems(_print)


def main():
    fd = make_hdf_header()
    globals()[FLAGS.task_name](fd)
    flush_hdf(fd)


if __name__ == '__main__':
    main()
