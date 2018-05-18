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

    def __init__(self, scale, **kwargs):
        self.scale = to_list(scale, repeat=2)

        self.trainable_weights = []
        self.bias = []
        self.inputs = []
        self.label = []
        self.outputs = []
        self.loss = []
        self.metrics = {}
        self.summary_op = None
        self.summary_writer = None
        self.sess = self._init_session()

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
        raise NotImplementedError('DO NOT use base SuperResolution directly! Use inheritive models instead.')

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
