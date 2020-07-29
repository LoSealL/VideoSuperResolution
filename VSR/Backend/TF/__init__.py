#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

import tensorflow as tf

ver_major, ver_minor, _ = [int(s) for s in tf.__version__.split('.')]
if ver_major >= 2:
  import tensorflow.compat.v1 as tf

  tf.disable_v2_behavior()
else:
  tfc = tf.contrib
  if ver_minor >= 15:
    import tensorflow.compat.v1 as tf
