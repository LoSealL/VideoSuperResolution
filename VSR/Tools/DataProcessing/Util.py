"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Jan 7th, 2019

Misc utility tools
- make TFRecords files
"""

import numpy as np
import tensorflow as tf


def make_tensor_label_records(tensors, labels, writer):
    assert isinstance(tensors, (list, tuple))
    assert isinstance(labels, (list, tuple))
    assert len(tensors) == len(labels)

    example = tf.train.Example(features=tf.train.Features())
    for _t, _l in zip(tensors, labels):
        assert isinstance(_t, bytes)
        assert isinstance(_l, str)

        bl = tf.train.BytesList(value=[_t])
        ff = example.features.feature.get_or_create(_l)
        ff.MergeFrom(tf.train.Feature(bytes_list=bl))
    writer.write(example.SerializeToString())
