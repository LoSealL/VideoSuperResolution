"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Specifies a model and evaluate its corresponded checkpoint.
"""

from pathlib import Path

import tensorflow as tf

from .Eval import Eval
from .Run import Config, fetch_datasets
from ..DataLoader.Loader import BasicLoader
from ..DataLoader.Dataset import Dataset

tf.flags.DEFINE_string("input_dir", None, "images to test")
tf.flags.DEFINE_string("reference_dir", None,
                       "GT images to refer, ignored if --dataset is not none.")
FLAGS = tf.flags.FLAGS


def load_folder(path1, path2=None):
  """loading `path` into a Dataset"""
  if not Path(path1).exists():
    raise FileNotFoundError("path can't be found.")

  images = sorted(Path(path1).glob('*'))
  if not images:
    images = sorted(Path(path1).iterdir())
  images2 = None
  if path2 is not None:
    images2 = sorted(Path(path2).glob('*'))
    if not images2:
      images2 = sorted(Path(path2).iterdir())
  return Dataset(test=images, pair=images2)


def evaluate(*args):
  if not FLAGS.input_dir:
    raise ValueError("--input_dir is required.")
  data_config_file = Path(FLAGS.data_config)
  if not data_config_file.exists():
    raise RuntimeError("dataset config file doesn't exist!")
  _, ref_data, _ = fetch_datasets(data_config_file, FLAGS)
  if FLAGS.reference_dir:
    input_data = load_folder(FLAGS.input_dir, FLAGS.reference_dir)
  else:
    input_data = load_folder(FLAGS.input_dir, ref_data.test)
  metric_config = Config(batch=1, scale=1, modcrop=False, crop=None)
  input_loader = BasicLoader(input_data, 'test', metric_config)
  input_images = [x[0] for x in input_loader.make_one_shot_iterator()]
  label_images = [x[1] for x in input_loader.make_one_shot_iterator()]
  Eval.evaluate(label_images, input_images)
