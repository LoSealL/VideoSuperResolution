#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 1 - 15

from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from scipy.io import loadmat, savemat

from VSR.Util.ImageProcess import img_to_array, array_to_img

tf.flags.DEFINE_string("train_dir", None, "Path to train Data.")
tf.flags.DEFINE_string("validation", None,
                       "Path to validation mat, save in individual png.")
tf.flags.DEFINE_string("metadata", None, "metadata")
tf.flags.DEFINE_string("results", None, "Path to results' png, save in mat.")
tf.flags.DEFINE_string("save_dir", None, "Output directory.")
tf.flags.DEFINE_integer("patch_size", 128, "Cropped patch size.")
tf.flags.DEFINE_integer("stride", 128, "Cropped patch stride.")
tf.flags.DEFINE_integer("scale", 1, "Resize lr images.")
tf.flags.DEFINE_integer("shave", 4, "Shave boarder pixels.")
tf.flags.DEFINE_bool("augment", False, "Augment data.")
tf.flags.DEFINE_integer("num", 100000, "Number of patches.")
FLAGS = tf.flags.FLAGS


def _augment(image, op):
  """Image augmentation"""
  if op[0]:
    image = np.rot90(image, 1)
  if op[1]:
    image = np.fliplr(image)
  if op[2]:
    image = np.flipud(image)
  return image


def denoise():
  save_dir = Path(FLAGS.save_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  if FLAGS.train_dir:
    # pre-processing training data
    train_dir = Path(FLAGS.train_dir)
    train_gt = sorted(train_dir.rglob('*GT_SRGB_010.PNG'))
    train_noisy = sorted(train_dir.rglob('*NOISY_SRGB_010.PNG'))
    assert len(train_gt) == len(train_noisy)
    # loading images
    train_gt_img = [Image.open(i) for i in train_gt]
    train_noisy_img = [Image.open(i) for i in train_noisy]
    # divide into patches and shave out boarders
    for name, img in tqdm.tqdm(zip(train_gt, train_gt_img), ascii=True,
                               total=len(train_gt)):
      folder = save_dir / 'train' / 'gt_patches'
      folder /= name.stem
      folder.mkdir(exist_ok=True, parents=True)
      box = [FLAGS.shave, FLAGS.shave,
             img.width - FLAGS.shave, img.height - FLAGS.shave]
      img = img.crop(box)
      w = img.width
      h = img.height
      img = img_to_array(img)
      patches = []
      size = FLAGS.patch_size
      stride = FLAGS.stride
      img = np.pad(img, [[0, size - h % stride or stride],
                         [0, size - w % stride or stride], [0, 0]],
                   mode='reflect')
      for i in np.arange(0, h, stride):
        for j in np.arange(0, w, stride):
          patches.append(img[i:i + size, j:j + size])
      for i, p in enumerate(patches):
        array_to_img(p, 'RGB').save(
          "{}/{}_{:04d}.png".format(str(folder), name.stem, i))
    for name, img in tqdm.tqdm(zip(train_noisy, train_noisy_img), ascii=True,
                               total=len(train_noisy)):
      folder = save_dir / 'train' / 'noisy_patches'
      folder /= name.stem
      folder.mkdir(exist_ok=True, parents=True)
      box = [FLAGS.shave, FLAGS.shave,
             img.width - FLAGS.shave, img.height - FLAGS.shave]
      img = img.crop(box)
      w = img.width
      h = img.height
      img = img_to_array(img)
      patches = []
      size = FLAGS.patch_size
      stride = FLAGS.stride
      img = np.pad(img, [[0, size - h % stride or stride],
                         [0, size - w % stride or stride], [0, 0]],
                   mode='reflect')
      for i in np.arange(0, h, stride):
        for j in np.arange(0, w, stride):
          patches.append(img[i:i + size, j:j + size])
      for i, p in enumerate(patches):
        array_to_img(p, 'RGB').save(
          "{}/{}_{:04d}.png".format(str(folder), name.stem, i))

  val_mat = FLAGS.validation
  metadata = FLAGS.metadata
  if metadata:
    metadata = sorted(Path(FLAGS.metadata).rglob('*.MAT'))
    metadata = [loadmat(str(m))['metadata'] for m in metadata]
    metadata = [m[0, 0][0][0] for m in metadata]
    metadata = [Path(m).parent.parent.stem for m in metadata]
    metadata[33] = "0158_007_GP_03200_03200_5500_N"
    metadata = np.asarray([m.split('_') for m in metadata])
    assert metadata.shape[1] == 7
  if val_mat:
    val_mat = loadmat(val_mat)['ValidationNoisyBlocksSrgb']
    assert val_mat.shape == (40, 32, 256, 256, 3)
    assert val_mat.dtype == 'uint8'
    g = enumerate(val_mat.reshape([-1, 256, 256, 3]))
    for i, img in tqdm.tqdm(g, total=40 * 32, ascii=True):
      img = Image.fromarray(img, 'RGB')
      if metadata is not None:
        suffix = "{}_{}_{}_{}_{}_{}".format(*metadata[i // 32][1:])
        img.save("{}/{:04d}_{}.png".format(save_dir, i, suffix))
  if FLAGS.results:
    results = []
    g = sorted(Path(FLAGS.results).glob('*.png'))
    assert len(g) == 40 * 32
    print(" [*] Appending results...")
    for img in tqdm.tqdm(g, ascii=True):
      img = Image.open(img)
      if img.width != 256 or img.height != 256:
        img = img.resize([256, 256], Image.BICUBIC)
      results.append(img_to_array(img))
    results = np.stack(results).reshape([40, 32, 256, 256, 3])
    savemat("{}/results".format(save_dir), {"results": results})
    print(" [*] Saved to {}/results.mat".format(save_dir))


def main(*args, **kwargs):
  denoise()


if __name__ == '__main__':
  tf.app.run(main)
