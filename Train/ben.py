"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Mar. 15 2019

help to benchmark models
Type --helpfull to get full doc.
"""

# Import models in development
try:
  from Exp import *
except ImportError as ex:
  pass

from importlib import import_module

import tensorflow as tf

from VSR.Tools import EvalDataDirectory, EvalModelCheckpoint

FLAGS = tf.flags.FLAGS


def main(*args, **kwargs):
  additional_functions = {}
  callbacks = []
  callbacks += FLAGS.f or []
  callbacks += FLAGS.f2 or []
  callbacks += FLAGS.f3 or []
  if callbacks:
    m = import_module('custom_api')
    for fn_name in callbacks:
      try:
        if '#' in fn_name:
          fn_name = fn_name.split('#')[0]
        additional_functions[fn_name] = m.__dict__[fn_name]
      except KeyError:
        raise KeyError(
          "Function [{}] couldn't be found in 'custom_api.py'".format(fn_name))

  if FLAGS.checkpoint_dir:
    return EvalModelCheckpoint.evaluate(*args[0][1:])
  elif FLAGS.input_dir:
    return EvalDataDirectory.evaluate(*args[0][1:])
  print(("In mode 'eval', parse either '--checkpoint_dir' with '--model'"
         " or '--input_dir' to evaluate models, see details --helpfull"))


if __name__ == '__main__':
  tf.app.run(main)
