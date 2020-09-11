"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2019-4-3

NTIRE Data helpers for NTIRE 2019
Containing tasks for:
- [Real Image Super-Resolution](https://competitions.codalab.org/competitions/21439)
- [Real Image Denoising (sRGB)](https://competitions.codalab.org/competitions/21266)
- [Image Dehazing](https://competitions.codalab.org/competitions/21163)

For reports of my related works, please refer to [README_NTIRE19.md](../Docs/README_NTIRE19.md)
"""

import argparse
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from scipy.io import loadmat, savemat

from VSR.Util.ImageProcess import array_to_img, img_to_array

__all__ = ["parse_mat", "group_mat", "divide", "combine"]


def main():
    parser = argparse.ArgumentParser(
        description=r"""NTIRE Data helpers for NTIRE 2019
    Containing tasks for:
    - [Real Image Super-Resolution](https://competitions.codalab.org/competitions/21439)
    - [Real Image Denoising (sRGB)](https://competitions.codalab.org/competitions/21266)
    - [Image Dehazing](https://competitions.codalab.org/competitions/21163)
    
    For reports of my related works, please refer to [README_NTIRE19.md](../Docs/README_NTIRE19.md)
    """)
    parser.add_argument("input_dir")
    parser.add_argument("save_dir")
    parser.add_argument("--task", choices=__all__, default=None)
    group0 = parser.add_argument_group("divide")
    group0.add_argument("--patch", type=int, default=0,
                        help="Patch size for dividing/combining sub-images")
    group0.add_argument("--stride", type=int, default=0,
                        help="Stride for dividing/combining sub-images")
    group1 = parser.add_argument_group("denoise")
    group1.add_argument(
        "--metadata", help="Path to denoising metadata directory")
    group2 = parser.add_argument_group("combine")
    group2.add_argument("--ref", help="Path to referenced original images")
    flags = parser.parse_args()

    if flags.task:
        functor = globals()[flags.task]
    else:
        # guess
        inputs = Path(flags.input_dir)
        if inputs.suffix == '.mat':
            functor = parse_mat
        elif flags.patch > 0 and flags.stride > 0:
            functor = divide
        elif flags.stride > 0:
            functor = combine
        else:
            raise RuntimeError("Should specify a running task!")
    return functor(flags)


def parse_mat(flags):
    """Parse MAT and its corresponding metadata file. Extract and save into png
      files, named using meta-data elements

    Challenge: Real Image Denoising

    Args:
      input_dir: path of the .MAT file
      metadata: folder that containing .mat format meta-data
    """

    save_dir = Path(flags.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    mat_dir = Path(flags.input_dir)
    if not mat_dir.exists():
        raise ValueError(f"Can't find {mat_dir.name}!")
    metadata = flags.metadata
    if metadata:
        metadata = sorted(Path(flags.metadata).rglob('*.MAT'))
        metadata = [loadmat(str(m))['metadata'] for m in metadata]
        metadata = [m[0, 0][0][0] for m in metadata]
        metadata = [Path(m).parent.parent.stem for m in metadata]
        # wrong name in original mat file
        metadata[33] = "0158_007_GP_03200_03200_5500_N"
        metadata = np.asarray([m.split('_') for m in metadata])
        assert metadata.shape[1] == 7, "Probably a wrong metadata folder."
    mat = loadmat(str(mat_dir))
    key = list(mat.keys())[-1]
    print(f"find key: [{key}]")
    val_mat = mat[key]
    assert val_mat.shape == (40, 32, 256, 256, 3), "Probably a wrong mat file."
    assert val_mat.dtype == 'uint8'
    g = enumerate(val_mat.reshape([-1, 256, 256, 3]))
    for i, img in tqdm.tqdm(g, total=40 * 32, ascii=True):
        img = Image.fromarray(img, 'RGB')
        if metadata is not None:
            suffix = "{}_{}_{}_{}_{}_{}".format(*metadata[i // 32][1:])
            img.save("{}/{:04d}_{}.png".format(save_dir, i, suffix))
        else:
            img.save("{}/{:04d}.png".format(save_dir, i))


def group_mat(flags):
    """Group denoised images into required mat format.

    Challenge: Real Image Denoising

    Args:
      input_dir: path of the result images.
      save_dir: saving folder or file name of .mat file.
    """

    save_dir = Path(flags.save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    if save_dir.is_dir():
        save_dir /= 'results'
    results = []
    g = sorted(Path(flags.input_dir).glob('*.png'))
    assert len(g) == 40 * 32, "Not enough image files!"
    print(" [*] Appending results...")
    for img in tqdm.tqdm(g, ascii=True):
        img = Image.open(img)
        if img.width != 256 or img.height != 256:
            img = img.resize([256, 256], Image.BICUBIC)
        results.append(img_to_array(img))
    results = np.stack(results).reshape([40, 32, 256, 256, 3])
    savemat(str(save_dir), {"results": results})
    print(" [*] Saved to {}.mat".format(save_dir))


def divide(flags):
    """Divide given images to small patches.

    Challenge: can be used to all challenges.

    Args:
      input_dir: path of images to be divided
      patch: dividing patch size
      stride: dividing stride (usually smaller than `patch`)
    """

    def _divide(img: Image, stride: int, size: int) -> list:
        w = img.width
        h = img.height
        img = img_to_array(img, data_format='channels_last')
        patches = []
        img = np.pad(img, [[0, size - h % stride or stride],
                           [0, size - w % stride or stride], [0, 0]],
                     mode='reflect')
        size - w % stride
        for i in np.arange(0, h, stride):
            for j in np.arange(0, w, stride):
                patches.append(img[i:i + size, j:j + size])
        return patches

    save_dir = Path(flags.save_dir).resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    files = sorted(Path(flags.input_dir).glob("*.png"))
    print(" [*] Dividing...\n")
    for f in tqdm.tqdm(files, ascii=True):
        pf = _divide(Image.open(f), flags.stride, flags.patch)
        for i, p in enumerate(pf):
            array_to_img(p, 'RGB', data_format='channels_last').save(
                f"{save_dir}/{f.stem}_{i:04d}.png")


def combine(flags):
    """Combining the divided small patches into entire big image.
      Used as a pair to `divide`.

    Args:
      input_dir: path of processed fragile images
      ref: path to referenced original images (target name, width/height)
      stride: stride used to divide them
    """

    def _combine(ref: Image, sub: list, stride) -> Image:
        w = ref.width
        h = ref.height
        blank = np.zeros([h, w, 3], 'float32')
        count = np.zeros([h, w, 1])
        k = 0
        for i in np.arange(0, h, stride):
            for j in np.arange(0, w, stride):
                p = sub[k]
                k += 1
                try:
                    blank[i:i + p.height, j:j + p.width] += img_to_array(
                        p, 'channels_last')
                except ValueError:
                    blank[i:i + p.height, j:j + p.width] += img_to_array(
                        p, 'channels_last')[:h - i, :w - j]
                count[i:i + p.height, j:j + p.width] += 1
        blank /= count
        return array_to_img(blank, 'RGB', 'channels_last')

    save_dir = Path(flags.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    files = sorted(Path(flags.ref).glob("*.png"))
    print(" [!] Combining...\n")
    results = Path(flags.input_dir)
    for f in tqdm.tqdm(files, ascii=True):
        sub = list(results.glob("{}_*.png".format(f.stem)))
        sub.sort(key=lambda x: int(x.stem[-4:]))
        sub = [Image.open(s) for s in sub]
        img = _combine(Image.open(f), sub, flags.stride)
        img.save("{}/{}.png".format(save_dir, f.stem))


if __name__ == '__main__':
    main()
