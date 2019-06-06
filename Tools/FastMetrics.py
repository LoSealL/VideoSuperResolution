#   Copyright (c): Wenyi Tang 2017-2019.
#   Author: Wenyi Tang
#   Email: wenyi.tang@intel.com
#   Update Date: 6/6/19, 10:35 AM

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from skimage.measure import compare_ssim

from VSR.Util.ImageProcess import rgb_to_yuv

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="target folder")
parser.add_argument("reference_dir", help="reference folder")
parser.add_argument("--ssim", action='store_true')
parser.add_argument("--l_only", action='store_true', help="luminance only")
parser.add_argument("--shave", type=int, default=0, help="shave border pixels")
parser.add_argument("--multithread", action='store_true')
FLAGS = parser.parse_args()


def gen():
  d1 = Path(FLAGS.input_dir)
  d2 = Path(FLAGS.reference_dir)

  assert d1.exists() and d2.exists(), "Path not found!"
  assert len(list(d1.iterdir())) == len(list(d2.iterdir())), f"{d1} v {d2}"

  for x, y in zip(sorted(d1.iterdir()), sorted(d2.iterdir())):
    if x.is_dir() and y.is_dir():
      assert len(list(x.iterdir())) == len(list(y.iterdir())), f"{x} v {y}"
      for i, j in zip(sorted(x.iterdir()), sorted(y.iterdir())):
        if i.is_file() and j.is_file():
          yield i, j
        else:
          print(f" [!] Found {i} v.s. {j} not file.")
    elif x.is_file() and y.is_file():
      yield x, y
    else:
      print(f" [!] Found {x} v.s. {y} mismatch.")


def main():
  mmse = 0
  apsnr = 0
  assim = 0
  count = 0

  def action(x, y):
    xname = f'{x.parent.name}/{x.stem}'
    yname = f'{y.parent.name}/{y.stem}'
    x = Image.open(x)
    y = Image.open(y)
    assert x.width == y.width and x.height == y.height, "Image size mismatch!"
    xx = np.asarray(x, dtype=np.float) / 255.0
    yy = np.asarray(y, dtype=np.float) / 255.0
    if FLAGS.l_only:
      xx = rgb_to_yuv(xx, standard='matlab')[..., :1]
      yy = rgb_to_yuv(yy, standard='matlab')[..., :1]
    if FLAGS.shave:
      xx = xx[..., FLAGS.shave:-FLAGS.shave, FLAGS.shave:-FLAGS.shave, :]
      yy = yy[..., FLAGS.shave:-FLAGS.shave, FLAGS.shave:-FLAGS.shave, :]
    mse = np.mean((xx - yy) ** 2)
    psnr = np.log10(1.0 / mse) * 10.0
    info = {"x": xname, "y": yname}
    if FLAGS.ssim:
      ssim = compare_ssim(xx, yy, multichannel=True)
      info.update(SSIM=ssim)
    info.update(PSNR=psnr)
    info.update(MSE=mse)
    return info

  if FLAGS.multithread:
    pool = mp.pool.ThreadPool()
    results = [pool.apply_async(action, (i, j)) for i, j in gen()]
    with tqdm.tqdm(results) as r:
      for info in r:
        info = info.get()
        if FLAGS.ssim:
          assim += info['SSIM']
        mmse += info['MSE']
        apsnr += info['PSNR']
        count += 1
        r.set_postfix(info)
  else:
    with tqdm.tqdm(gen()) as r:
      for x, y in r:
        info = action(x, y)
        if FLAGS.ssim:
          assim += info['SSIM']
        mmse += info['MSE']
        apsnr += info['PSNR']
        count += 1
        r.set_postfix(info)
  mmse /= count
  apsnr /= count
  assim /= count
  mpsnr = np.log10(1.0 / mmse) * 10.0
  print(f"[*] Mean PSNR(MMSE): {mpsnr:.2f}")
  print(f"[*] Avg PSNR: {apsnr:.2f}")
  print(f"[*] Avg SSIM: {assim:.4f}")


if __name__ == '__main__':
  main()
