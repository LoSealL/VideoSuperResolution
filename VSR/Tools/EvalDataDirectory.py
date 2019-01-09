"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Specifies a model and evaluate its corresponded checkpoint.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

from ..DataLoader.Dataset import Dataset
from .Run import fetch_datasets, Config, QuickLoader
from .Eval import Eval

tf.flags.DEFINE_string("input_dir", None, "images to test")
tf.flags.DEFINE_string("reference_dir", None, "GT images to refer, ignored if --dataset is not none.")
FLAGS = tf.flags.FLAGS


def load_folder(path):
    """loading `path` into a Dataset"""
    if not Path(path).exists():
        raise FileNotFoundError("path can't be found.")

    images = list(Path(path).glob('*'))
    images.sort()
    if not images:
        images = list(Path(path).iterdir())
    return Dataset(test=images)


def evaluate():
    if not FLAGS.input_dir:
        raise ValueError("--input_dir is required.")
    data_config_file = Path(FLAGS.data_config)
    if not data_config_file.exists():
        raise RuntimeError("dataset config file doesn't exist!")
    _, ref_data, _ = fetch_datasets(data_config_file, FLAGS)
    input_data = load_folder(FLAGS.input_dir)
    metric_config = Config(batch=1, scale=1, modcrop=False, crop=None)
    ref_loader = QuickLoader(ref_data, 'test', metric_config)
    input_loader = QuickLoader(input_data, 'test', metric_config)
    label_images = [x[0] for x in ref_loader.make_one_shot_iterator()]
    if not label_images:
        backup_data = load_folder(FLAGS.reference_dir)
        backup_loader = QuickLoader(backup_data, 'test', metric_config)
        label_images = [x[0] for x in backup_loader.make_one_shot_iterator()]
    input_images = [x[0] for x in input_loader.make_one_shot_iterator()]
    Eval.evaluate(label_images, input_images)
