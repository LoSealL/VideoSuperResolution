"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 9th 2018
Updated Date: May 11th 2018

Framework for network model (tensorflow)
"""
import tensorflow as tf
import numpy as np
from pathlib import Path

from ..Util.Utility import to_list


class SuperResolution:

    def __init__(self, scale, rgb_input=False, **kwargs):
        self.scale = to_list(scale, repeat=2)
        self.rgba = rgb_input

        self.trainable_weights = []
        self.bias = []
        self.inputs = []
        self.inputs_preproc = []
        self.label = []
        self.outputs = []
        self.loss = []
        self.metrics = {}
        self.summary_op = None
        self.summary_writer = None
        self.unknown_args = kwargs
        self.sess = self._init_session()

    def __getattr__(self, item):
        return self.unknown_args.get(item)

    def _init_session(self):
        self.training_phase = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        return tf.InteractiveSession()

    def compile(self):
        self.build_graph()
        self.build_loss()
        self.build_summary()
        self.summary_op = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        return self

    def summary(self):
        pass

    def build_graph(self):
        if not self.rgba:
            self.inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None, 1], name='input/lr/gray'))
            self.inputs_preproc.append(tf.cast(self.inputs[-1], tf.float32, name='cast/lr/gray_float'))
        else:
            self.inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None, 4], name='input/lr/rgba'))
            yuv = tf.cast(self.inputs[-1], tf.float32) / 255.0
            yuv = tf.image.rgb_to_yuv(yuv[..., :-1])  # discard alpha channel
            self.inputs_preproc.append(yuv[..., 1:])  # unscaled UV channel
            self.inputs_preproc.append(yuv[..., 0:1] * 255)  # scaled Y channel

    def build_loss(self):
        raise NotImplementedError('DO NOT use base SuperResolution directly! Use inheritive models instead.')

    def build_summary(self):
        raise NotImplementedError('DO NOT use base SuperResolution directly! Use inheritive models instead.')

    def train_batch(self, feature, label, **kwargs):
        feature = to_list(feature)
        label = to_list(label)
        lr = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1e-4
        feed_dict = {self.training_phase: True, self.learning_rate: lr}
        for i in range(len(self.inputs)):
            feed_dict[self.inputs[i]] = feature[i]
        for i in range(len(self.label)):
            feed_dict[self.label[i]] = label[i]
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def validate_batch(self, feature, label, **kwargs):
        feature = to_list(feature)
        label = to_list(label)
        feed_dict = {self.training_phase: False}
        for i in range(len(self.inputs)):
            feed_dict[self.inputs[i]] = feature[i]
        for i in range(len(self.label)):
            feed_dict[self.label[i]] = label[i]
        metrics = self.sess.run(list(self.metrics.values()) + [self.summary_op], feed_dict=feed_dict)
        ret = {}
        for k, v in zip(self.metrics, metrics[:-1]):
            ret[k] = v
        return ret, metrics[-1]

    def test_batch(self, inputs, label=None, **kwargs):
        feature = to_list(inputs)
        label = to_list(label)
        feed_dict = {self.training_phase: False}
        for i in range(len(self.inputs)):
            feed_dict[self.inputs[i]] = feature[i]
        if label:
            for i in range(len(self.label)):
                feed_dict[self.label[i]] = label[i]
            return self.sess.run(self.outputs + list(self.metrics.values()), feed_dict=feed_dict)
        else:
            return self.sess.run(self.outputs, feed_dict=feed_dict)

    def export_model_pb(self, export_dir='.', export_name='model.pb', **kwargs):

        graph = self.sess.graph.as_graph_def()
        graph = tf.graph_util.remove_training_nodes(graph)
        graph = tf.graph_util.convert_variables_to_constants(
            self.sess, graph, [outp.name.split(':')[0] for outp in self.outputs])
        tf.train.write_graph(graph, export_dir, export_name, as_text=False)
        print(f"Model exported to [ {Path(export_dir).resolve() / export_name} ].")
