"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: June 1st 2018
Updated Date: June1st 2018

Implementing super-resolution network for multiple degradations (SRMD)
See https://arxiv.org/abs/1712.06116
**Learning a Single Convolutional Super-Resolution Network for Multiple Degradations**
"""
from VSR.Framework.SuperResolution import SuperResolution

import tensorflow as tf


class SRMD(SuperResolution):

    def __init__(self, layers=8, name='srmd', **kwargs):
        self.layers = layers
        self.name = name
        super(SRMD, self).__init__(**kwargs)

    def build_graph(self):
        with tf.name_scope(self.name):
            super(SRMD, self).build_graph()
            # degradation model
            self.inputs.append(tf.placeholder(tf.float32, [None, None, None]))
