#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

from .. import LOG

import tensorflow as tf

ver_major, ver_minor, _ = [int(s) for s in tf.__version__.split('.')]
if ver_major < 2:
  LOG.warning("legacy tensorflow 1.x is not verified in keras backend")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
