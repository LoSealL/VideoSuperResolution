#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 1 - 12

import io
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

from VSR.Tools.DataProcessing.Util import make_tensor_label_records

tf.flags.DEFINE_string("train_gt", None, "Path to TrainGT.")
tf.flags.DEFINE_string("train_hazy", None, "Path to TrainHazy.")
tf.flags.DEFINE_string("validation", None,
                       "Path to Validations. For cropping into pieces.")
tf.flags.DEFINE_string("save_dir", None, "Output directory.")
tf.flags.DEFINE_integer("patch_size", 128, "Cropped patch size.")
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


def dehaze():
  train_gt = Path(FLAGS.train_gt).glob('*.png')
  train_hazy = Path(FLAGS.train_hazy).glob('*.png')
  train_gt = sorted(train_gt)
  train_hazy = sorted(train_hazy)
  Path(FLAGS.save_dir).mkdir(exist_ok=True, parents=True)
  writer = tf.io.TFRecordWriter(
    FLAGS.save_dir + "/ntire_dehaze-train.tfrecords")
  writer2 = tf.io.TFRecordWriter(
    FLAGS.save_dir + "/ntire_dehaze-test.tfrecords")
  num_each = FLAGS.num // len(train_gt)
  for gt, hazy in zip(train_gt, train_hazy):
    assert gt.stem == hazy.stem
    name = gt.stem
    print(name)
    image_gt = Image.open(gt)
    image_hazy = Image.open(hazy)
    _w, _h = image_gt.width, image_gt.height
    _pw = _ph = FLAGS.patch_size
    x = np.random.randint(0, _w - _pw + 1, size=num_each)
    y = np.random.randint(0, _h - _ph + 1, size=num_each)
    box = [(_x, _y, _x + _pw, _y + _ph) for _x, _y in zip(x, y)]
    patches_gt = [np.asarray(image_gt.crop(b)) for b in box]
    patches_hazy = [np.asarray(image_hazy.crop(b)) for b in box]
    if FLAGS.augment:
      ops = np.random.randint(0, 2, size=[num_each, 3])
      patches_gt = [_augment(p, op) for p, op in zip(patches_gt, ops)]
      patches_hazy = [_augment(p, op) for p, op in zip(patches_hazy, ops)]
    for i, patches in tqdm.tqdm(enumerate(zip(patches_gt, patches_hazy)),
                                total=num_each, ascii=True):
      hr, lr = patches
      label = "{}_{}".format(name, i).encode()
      with io.BytesIO() as fp:
        Image.fromarray(hr, 'RGB').save(fp, format='png')
        fp.seek(0)
        hr_png = fp.read()
      with io.BytesIO() as fp:
        Image.fromarray(lr, 'RGB').save(fp, format='png')
        fp.seek(0)
        lr_png = fp.read()
      make_tensor_label_records([hr_png, lr_png, label, lr_png],
                                ["image/hr", "image/lr", "name", "image/post"],
                                writer)
      if i == 0:
        make_tensor_label_records(
          [hr_png, lr_png, label, lr_png],
          ["image/hr", "image/lr", "name", "image/post"],
          writer2)
  if FLAGS.validation:
    validation_hazy = Path(FLAGS.validation).glob('*.png')
    for val in tqdm.tqdm(validation_hazy, ascii=True):
      name = val.stem
      image_val = Image.open(val)
      _w, _h = image_val.width, image_val.height
      _pw = _ph = FLAGS.patch_size
      image_val = np.array(image_val)
      n_rows = _h // _ph
      n_cols = _w // _pw
      for i in range(n_rows):
        for j in range(n_cols):
          p = image_val[i * _ph: i * _ph + _ph, j * _pw:j * _pw + _pw, :]


def main(*args, **kwargs):
  dehaze()


if __name__ == '__main__':
  tf.app.run(main)
