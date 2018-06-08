"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 8th 2018
Updated Date: June 8th 2018

Learning a Single Convolutional Super-Resolution Network for Multiple Degradations
See https://arxiv.org/abs/1712.06116
"""

from ..Framework.SuperResolution import SuperResolution
from ..Util.Utility import *

import tensorflow as tf
import numpy as np


class SRMD(SuperResolution):

    def __init__(self, name='srmd', **kwargs):
        self.name = name
        super(SRMD, self).__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name):
            super(SRMD, self).build_graph()
