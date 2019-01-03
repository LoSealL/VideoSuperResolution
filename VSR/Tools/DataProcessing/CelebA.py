"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Jan 3rd, 2019

Pre-processing CelebA dataset:
- Crop and resize to WxH
- Randomly split into (train, test)
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("celeba", help="CelebA root folder.")
parser.add_argument("-W", type=int, default=64, help="width")
parser.add_argument("-H", type=int, default=64, help="height")
parser.add_argument("--n_test", type=int, default=10000, help="test samples")
args = parser.parse_args()


def main():
    root = Path(args.celeba)
    if not root.exists():
        raise FileNotFoundError("Root of CelebA does not exist!")

    images = list(root.rglob('*.jpg'))
    resize_dir = root / 'resize{}'.format(args.W)
    test_dir = root / 'test{}'.format(args.W)
    resize_dir.mkdir(parents=True, exist_ok=False)
    test_dir.mkdir(parents=True, exist_ok=False)

    np.random.shuffle(images)
    for img in images[:args.n_test]:
        x = Image.open(img)
        dw = (x.width - args.W) // 2
        dh = (x.height - args.H) // 2
        box = [dw, dh, x.width - dw, x.height - dh]
        x.crop(box).save(str(test_dir) + '/{}.png'.format(img.stem))

    for img in images[args.n_test:]:
        x = Image.open(img)
        dw = (x.width - args.W) // 2
        dh = (x.height - args.H) // 2
        box = [dw, dh, x.width - dw, x.height - dh]
        x.crop(box).save(str(resize_dir) + '/{}.png'.format(img.stem))


if __name__ == '__main__':
    main()
