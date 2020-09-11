"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-15

Calc PSNR/SSIM using thread pool
"""
import argparse
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from skimage.metrics import structural_similarity

from VSR.Util.ImageProcess import rgb_to_yuv

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="target folder")
parser.add_argument("reference_dir", help="reference folder")
parser.add_argument("--ssim", action='store_true')
parser.add_argument("--l_only", action='store_true', help="luminance only")
parser.add_argument("--shave", type=int, default=0, help="shave border pixels")
parser.add_argument("--multithread", action='store_true')
FLAGS = parser.parse_args()


def split_path_filter(x: Path):
    try:
        x = x.resolve()
        # path, glob pattern, recursive
        return x, '*', False
    except OSError:
        print(str(x.as_posix()))
        pattern = x.name
        rec = False
        x = x.parent
        if '*' in x.name:
            x = x.parent
            rec = True
        print(x, pattern, rec)
        return x, pattern, rec


def gen():
    d1 = Path(FLAGS.input_dir)
    d2 = Path(FLAGS.reference_dir)
    d1, d1_filter, d1_rec = split_path_filter(d1)
    d2, d2_filter, d2_rec = split_path_filter(d2)

    assert d1.exists() and d2.exists(), "Path not found!"
    d1 = sorted(d1.rglob(d1_filter)) if d1_rec else sorted(d1.glob(d1_filter))
    d2 = sorted(d2.rglob(d2_filter)) if d2_rec else sorted(d2.glob(d2_filter))
    assert len(d1) == len(d2), f"{len(d1)} v {len(d2)}"

    for x, y in zip(d1, d2):
        if x.is_file() and y.is_file():
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
        x = Image.open(x).convert('RGB')
        y = Image.open(y).convert('RGB')
        if x.width != y.width or x.height != y.height:
            min_w = min(x.width, y.width)
            min_h = min(x.height, y.height)
            x = x.crop([0, 0, min_w, min_h])
            y = y.crop([0, 0, min_w, min_h])
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
            ssim = structural_similarity(xx, yy, multichannel=True)
            info.update(SSIM=ssim)
        info.update(PSNR=psnr)
        info.update(MSE=mse)
        return info

    if FLAGS.multithread:
        pool = ThreadPool()
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
