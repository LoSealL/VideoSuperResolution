"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Improved train/benchmark/infer script
Type --helpfull to get full doc.
"""

# Import models in development
try:
    from Exp import *
except ImportError as ex:
    pass

import tensorflow as tf
from importlib import import_module
from VSR.Tools import Run
from VSR.Tools import EvalModelCheckpoint, EvalDataDirectory

tf.flags.DEFINE_enum("mode", 'run', ('run', 'eval'), "tools to use.")
FLAGS = tf.flags.FLAGS


def main(*args, **kwargs):
    additional_functions = {}
    if FLAGS.add_custom_callbacks:
        m = import_module('custom_api')
        for fn_name in FLAGS.add_custom_callbacks:
            try:
                additional_functions[fn_name] = m.__dict__[fn_name]
            except KeyError:
                raise KeyError(f"Function [{fn_name}] couldn't be found in 'custom_api.py'")
    if FLAGS.mode == 'run':
        return Run.run(**additional_functions)
    if FLAGS.mode == 'eval':
        if FLAGS.checkpoint_dir:
            return EvalModelCheckpoint.evaluate()
        elif FLAGS.input_dir:
            return EvalDataDirectory.evaluate()
        print(("In mode 'eval', parse either '--checkpoint_dir' with '--model'"
               " or '--input_dir' to evaluate models, see details --helpfull"))


if __name__ == '__main__':
    tf.app.run(main)
