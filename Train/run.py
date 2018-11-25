"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Improved train/benchmark/infer script
"""

import tensorflow as tf
from importlib import import_module
# Import models in development
try:
    from Exp import *
except ImportError as ex:
    pass

from VSR.Tools import Run
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
    return Run.run(**additional_functions)


if __name__ == '__main__':
    tf.app.run(main)
