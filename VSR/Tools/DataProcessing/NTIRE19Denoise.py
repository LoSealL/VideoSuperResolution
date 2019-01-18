#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 1 - 15

import io
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from scipy.io import loadmat, savemat
from VSR.Util.ImageProcess import img_to_array, array_to_img
from VSR.Tools.DataProcessing.Util import make_tensor_label_records

tf.flags.DEFINE_string("train_dir", None, "Path to train Data.")
tf.flags.DEFINE_string("validation", None,
                       "Path to validation mat, save in individual png.")
tf.flags.DEFINE_string("metadata", None, "metadata")
tf.flags.DEFINE_string("results", None, "Path to results' png, save in mat.")
tf.flags.DEFINE_string("save_dir", None, "Output directory.")
tf.flags.DEFINE_integer("patch_size", 128, "Cropped patch size.")
tf.flags.DEFINE_integer("scale", 1, "Resize lr images.")
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
    train_dir = Path(FLAGS.train_dir)
    train_gt = sorted(train_dir.rglob('*GT_SRGB_010.PNG'))
    train_noisy = sorted(train_dir.rglob('*NOISY_SRGB_010.PNG'))
    assert len(train_gt) == len(train_noisy)
    writer = tf.io.TFRecordWriter(
      "{}/ntire_denoise_x{}-train.tfrecords".format(FLAGS.save_dir,
                                                    FLAGS.scale))
    num_each = FLAGS.num // len(train_gt)
    for gt, noisy in zip(train_gt, train_noisy):
      print(gt.stem, noisy.stem)
      name = gt.stem[:4]
      image_gt = Image.open(gt)
      image_noisy = Image.open(noisy)
      _w, _h = image_gt.width, image_gt.height
      _pw = _ph = FLAGS.patch_size
      x = np.random.randint(0, _w - _pw + 1, size=num_each)
      y = np.random.randint(0, _h - _ph + 1, size=num_each)
      box = [(_x, _y, _x + _pw, _y + _ph) for _x, _y in zip(x, y)]
      patches_gt = [np.asarray(image_gt.crop(b)) for b in box]
      patches_noisy = [np.asarray(image_noisy.crop(b)) for b in box]
      if FLAGS.augment:
        ops = np.random.randint(0, 2, size=[num_each, 3])
        patches_gt = [_augment(p, op) for p, op in zip(patches_gt, ops)]
        patches_noisy = [_augment(p, op) for p, op in zip(patches_noisy, ops)]
      for i, patches in tqdm.tqdm(enumerate(zip(patches_gt, patches_noisy)),
                                  total=num_each, ascii=True):
        hr, noise = patches
        label = "{}_{}".format(name, i).encode()
        with io.BytesIO() as fp:
          hr = array_to_img(hr, 'RGB')
          hr.save(fp, format='png')
          fp.seek(0)
          hr_png = fp.read()
        with io.BytesIO() as fp:
          lr = hr.resize([hr.width // FLAGS.scale, hr.height // FLAGS.scale],
                         Image.BICUBIC)
          lr.save(fp, format='png')
          fp.seek(0)
          lr_png = fp.read()
        with io.BytesIO() as fp:
          array_to_img(noise, 'RGB').save(fp, format='png')
          fp.seek(0)
          noisy_png = fp.read()
        make_tensor_label_records(
          [hr_png, lr_png, label, noisy_png],
          ["image/hr", "image/lr", "name", "image/post"],
          writer)

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
    root = Path(FLAGS.validation).parent / 'validation'
    root.mkdir(exist_ok=True, parents=True)
    g = enumerate(val_mat.reshape([-1, 256, 256, 3]))
    for i, img in tqdm.tqdm(g, total=40 * 32, ascii=True):
      img = Image.fromarray(img, 'RGB')
      if metadata is not None:
        suffix = "{}_{}_{}_{}_{}_{}".format(*metadata[i // 32][1:])
        img.save("{}/{:04d}_{}.png".format(root, i, suffix))
  if FLAGS.results:
    results = []
    g = sorted(Path(FLAGS.results).glob('*.png'))
    assert len(g) == 40 * 32
    print("Appending results...")
    for img in tqdm.tqdm(g, ascii=True):
      img = Image.open(img)
      if img.width != 256 or img.height != 256:
        img = img.resize([256, 256], Image.BICUBIC)
      results.append(img_to_array(img))
    results = np.stack(results).reshape([40, 32, 256, 256, 3])
    savemat("{}/results.MAT".format(save_dir), {"results": results})
    print("Saved to {}/results.MAT".format(save_dir))


def main(*args, **kwargs):
  denoise()


if __name__ == '__main__':
  tf.app.run(main)
