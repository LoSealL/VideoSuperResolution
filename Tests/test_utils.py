"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

Test utility
"""
import os
import unittest

from VSR.Backend import BACKEND

if not os.getcwd().endswith('Tests'):
    os.chdir('Tests')

try:
    if BACKEND == 'pytorch':
        # pylint: disable=wildcard-import
        from pt_backend import *
    elif BACKEND == 'tensorflow' or BACKEND == 'keras':
        # pylint: disable=wildcard-import
        from tf_backend import *
finally:
    # pylint: disable=wildcard-import
    from vsr_framework import *


if __name__ == "__main__":
    unittest.main()
