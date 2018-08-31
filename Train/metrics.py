"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Aug 17th 2018

Calculate metrics for outputs and labels:
- PSNR
- SSIM
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

from VSR.DataLoader.Dataset import load_datasets, Dataset
from VSR.DataLoader.Loader import QuickLoader

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

tf.flags.DEFINE_string("input_dir", None, "images to test")
tf.flags.DEFINE_string("dataset", None, "dataset to compare")
tf.flags.DEFINE_bool("no_ssim", False, "disable ssim metric")
tf.flags.DEFINE_bool("no_psnr", False, "disable psnr metric")
tf.flags.DEFINE_bool("l_only", False, "compute luminance only")
tf.flags.DEFINE_integer("shave", 0, "shave border pixels")
tf.flags.DEFINE_integer("clip", -1, "depth of a clip, -1 includes all frames")

opt = tf.flags.FLAGS


def load_folder(path):
    """loading `path` into a Dataset"""
    if not Path(path).exists():
        raise ValueError("--input_dir can't be found")

    images = list(Path(path).glob('*'))
    images.sort()
    if not images:
        images = list(Path(path).iterdir())
    D = Dataset(test=images)
    return D


def shave(img, border):
    """shave away border pixels"""
    return img[..., border:-border, border:-border, :]


def main(*args):
    if not opt.input_dir:
        raise ValueError("--input_dir is required")
    if not opt.dataset.upper() in DATASETS.keys():
        raise ValueError("--dataset is missing, or can't be found")
    data_ref = DATASETS.get(opt.dataset.upper())
    data_ref.setattr(depth=opt.clip)
    data = load_folder(opt.input_dir)
    data.setattr(depth=opt.clip)
    loader = QuickLoader(1, data, 'test', 1, convert_to='RGB', no_patch=True)
    ref_loader = QuickLoader(1, data_ref, 'test', 1, convert_to='RGB', no_patch=True)
    # make sure len(ref_loader) == len(loader)
    loader_iter = loader.make_one_shot_iterator()
    ref_iter = ref_loader.make_one_shot_iterator()
    for ref, _, name in ref_iter:
        name = str(name)
        img, _, _ = next(loader_iter)
        # reduce the batch dimension for video clips
        if img.ndim == 5: img = img[0]
        if ref.ndim == 5: ref = ref[0]
        img = tf.constant(img.astype(np.float32))
        ref = tf.constant(ref.astype(np.float32))
        if opt.shave:
            img = shave(img, opt.shave)
            ref = shave(ref, opt.shave)
        if opt.l_only:
            img = tf.image.rgb_to_grayscale(img)
            ref = tf.image.rgb_to_grayscale(ref)
        psnr = tf.reduce_mean(tf.image.psnr(ref, img, 255)).eval() if not opt.no_psnr else 0
        ssim = tf.reduce_mean(tf.image.ssim(ref, img, 255)).eval() if not opt.no_ssim else 0
        tf.logging.info(f'[{name}] PSNR = {psnr}, SSIM = {ssim}')
        tf.add_to_collection('PSNR', psnr)
        tf.add_to_collection('SSIM', ssim)
    for key in ('PSNR', 'SSIM'):
        mp = np.mean(tf.get_collection(key))
        tf.logging.info(f'Mean {key}: {mp}')


if __name__ == '__main__':
    with tf.Session():
        # show log
        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.app.run(main)
