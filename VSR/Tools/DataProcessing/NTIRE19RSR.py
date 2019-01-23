#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 1 - 22


from pathlib import Path
import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image

from VSR.Util.ImageProcess import array_to_img, img_to_array

tf.flags.DEFINE_string("trainlr", None, "Path to train LR Data.")
tf.flags.DEFINE_string("trainhr", None, "Path to train HR Data.")
tf.flags.DEFINE_string("validation", None, "Path to validation LR.")
tf.flags.DEFINE_string("results", None, "Path to results' png.")
tf.flags.DEFINE_string("save_dir", None, "Output directory.")
tf.flags.DEFINE_integer("patch_size", 128, "Cropped patch size.")
tf.flags.DEFINE_integer("stride", 128, "Cropped patch stride.")
tf.flags.DEFINE_integer("scale", 1, "Resize lr images.")

FLAGS = tf.flags.FLAGS


def divide(img: Image, stride: int, size: int) -> list:
  w = img.width
  h = img.height
  img = img_to_array(img)
  patches = []
  img = np.pad(img, [[0, size - h % stride or stride],
                     [0, size - w % stride or stride], [0, 0]],
               mode='reflect')
  size - w % stride
  for i in np.arange(0, h, stride):
    for j in np.arange(0, w, stride):
      patches.append(img[i:i + size, j:j + size])
  return patches


def combine(ref: Image, sub: list, stride) -> Image:
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
        blank[i:i + p.height, j:j + p.width] += img_to_array(p)
      except ValueError:
        blank[i:i + p.height, j:j + p.width] += img_to_array(p)[:h - i, :w - j]
      count[i:i + p.height, j:j + p.width] += 1
  blank /= count
  return array_to_img(blank, 'RGB')


def rsr():
  save_dir = Path(FLAGS.save_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  if FLAGS.trainlr:
    files = sorted(Path(FLAGS.trainlr).glob("*.png"))
    if FLAGS.results:
      print(" [!] Combining...\n")
      results = Path(FLAGS.results)
      for f in tqdm.tqdm(files, ascii=True):
        sub = list(results.glob("{}_????.png".format(f.stem)))
        sub.sort(key=lambda x: int(x.stem[-4:]))
        sub = [Image.open(s) for s in sub]
        img = combine(Image.open(f), sub, FLAGS.stride)
        img.save("{}/{}_sr.png".format(save_dir, f.stem))
    else:
      print(" [!] Dividing...\n")
      for f in tqdm.tqdm(files, ascii=True):
        pf = divide(Image.open(f), FLAGS.stride, FLAGS.patch_size)
        for i, p in enumerate(pf):
          array_to_img(p, 'RGB').save(
            "{}/{}_{:04d}.png".format(save_dir, f.stem, i))
  if FLAGS.validation:
    files = sorted(Path(FLAGS.validation).glob("*.png"))
    if FLAGS.results:
      results = Path(FLAGS.results)
      print(" [!] Combining...\n")
      for f in tqdm.tqdm(files, ascii=True):
        sub = list(results.glob("{}_????.png".format(f.stem)))
        sub.sort(key=lambda x: int(x.stem[-4:]))
        sub = [Image.open(s) for s in sub]
        img = combine(Image.open(f), sub, FLAGS.stride)
        img.save("{}/{}_sr.png".format(save_dir, f.stem))
    else:
      print(" [!] Dividing...\n")
      for f in tqdm.tqdm(files, ascii=True):
        pf = divide(Image.open(f), FLAGS.stride, FLAGS.patch_size)
        for i, p in enumerate(pf):
          array_to_img(p, 'RGB').save(
            "{}/{}_{:04d}.png".format(save_dir, f.stem, i))


def main(*args, **kwargs):
  rsr()


if __name__ == '__main__':
  tf.app.run(main)
