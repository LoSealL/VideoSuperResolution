"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 9th 2018
Updated Date: June 15th 2018

Framework for network model (tensorflow)
"""
import tensorflow as tf
from pathlib import Path

from ..Util.Utility import to_list
from .LayersHelper import Layers


class SuperResolution(Layers):
    r"""A utility class helps for building SR architectures easily

    Usage:
        Inherit from `SuperResolution` and implement:
          >>> build_graph()
          >>> build_loss()
          >>> build_summary()
        If you want to export gragh as a protobuf (say model.pb), implement:
          >>> export_model_pb()
        and call its super method at the end
    """

    def __init__(self, scale, channel=1, weight_decay=1e-4, rgb_input=False, **kwargs):
        r"""Common initialize parameters

        Args:
            scale: the scale factor, can be a list of 2 integer to specify different stretch in width and height
            channel: input color channel
            weight_decay: decay of L2 regularization on trainable weights
            rgb_input: if True, specify inputs as RGBA with 4 channels, otherwise the input is grayscale image1
        """

        self.scale = to_list(scale, repeat=2)
        self.channel = channel
        self.weight_decay = weight_decay
        self.rgba = rgb_input

        self.trainable_weights = []
        self.bias = []
        self.inputs = []
        self.inputs_preproc = []
        self.label = []
        self.outputs = []
        self.loss = []
        self.train_metric = {}
        self.metrics = {}
        self.feed_dict = {}
        self.global_steps = None
        self.summary_op = None
        self.summary_writer = None
        self.compiled = False
        self.unknown_args = kwargs

        self.global_steps = tf.Variable(0, trainable=False, name='global_step')
        self.training_phase = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def __getattr__(self, item):
        """return extra initialized parameters"""

        return self.unknown_args.get(item)

    def compile(self):
        """build entire graph and training ops"""

        self.build_graph()
        self.build_loss()
        self.build_summary()
        self.summary_op = tf.summary.merge_all()
        self.build_saver()
        self.compiled = True
        return self

    def summary(self):
        """print model information"""

        pass

    def build_saver(self):
        """Build variable savers.
        By default, I build a saver to save all variables. In case you need to recover a part of variables,
        you can inherit this method and create multiple savers for different variables. All savers should
        arrange in a dict which maps saver and its saving name
        """

        default_saver = tf.train.Saver(max_to_keep=10, allow_empty=True)
        self.savers = {self.name: default_saver}

    def build_graph(self):
        """this super method create 2 kinds of input placeholder:
            - For grayscale image1, input type is uint8, cast to float32 in self.inputs_preproc
            - For colored image1, input type is uint8, mode is RGBA, discard alpha channel and convert to YUV,
              self.inputs_preproc[0] is normalized UV with value ranged [-0.5, 0.5]
              self.inputs_preproc[1] is casted Y channel ranged [0, 255.0]

        Note
            You can also suppress this method and create your own inputs from scratch
        """
        if not self.rgba:
            self.inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, None], name='input/lr/gray'))
            self.inputs_preproc.append(self.inputs[-1][..., self.channel:])
            self.inputs_preproc.append(self.inputs[-1][..., :self.channel])
            self.inputs_preproc[-1].set_shape([None, None, None, self.channel])
        else:
            self.inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None, None], name='input/lr/rgba'))
            yuv = tf.cast(self.inputs[-1], tf.float32) / 255.0
            yuv = tf.image.rgb_to_yuv(yuv[..., 0:3])  # discard alpha channel
            self.inputs_preproc.append(yuv[..., 1:3])  # unscaled UV channel
            self.inputs_preproc.append(yuv[..., 0:1] * 255)  # scaled Y channel
        self.label.append(tf.placeholder(tf.float32, shape=[None, None, None, self.channel], name='label/hr'))

    def build_loss(self):
        """help to build mse loss via self.label[-1] and self.outputs[-1] for simplicity

        >>> loss = tf.losses.mean_squared_error(self.label[-1], self.outputs[-1])

        Note
            You can also suppress this method and build your own loss function from scratch
        """

        opt = tf.train.AdamOptimizer(self.learning_rate)
        mse = tf.losses.mean_squared_error(self.label[-1], self.outputs[-1])
        loss = tf.losses.get_total_loss()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.loss.append(opt.minimize(loss, self.global_steps))

        return mse, loss

    def build_summary(self):
        # the pure abstract method
        raise NotImplementedError('DO NOT use base SuperResolution directly! Use inheritive models instead.')

    def train_batch(self, feature, label, learning_rate=1e-4, **kwargs):
        r"""training one batch one step

        Args:
            feature: input tensors, LR image1 for SR use case
            label: labels, HR image1 for SR use case
            learning_rate: update step size in current calculation
            kwargs: for future use

        Return:
            the results of ops in `self.loss`
        """

        feature = to_list(feature)
        label = to_list(label)
        self.feed_dict.update({self.training_phase: True, self.learning_rate: learning_rate})
        for i in range(len(self.inputs)):
            self.feed_dict[self.inputs[i]] = feature[i]
        for i in range(len(self.label)):
            self.feed_dict[self.label[i]] = label[i]
        loss = kwargs.get('loss') or self.loss
        loss = to_list(loss)
        loss = tf.get_default_session().run(list(self.train_metric.values()) + loss, feed_dict=self.feed_dict)
        ret = {}
        for k, v in zip(self.train_metric, loss):
            ret[k] = v
        return ret

    def validate_batch(self, feature, label, **kwargs):
        r"""validate one batch for one step

        Args:
            feature: input tensors, LR image1 for SR use case
            label: labels, HR image1 for SR use case
            kwargs: for future use

        Return:
            a dict of metrics defined in model, the summary op
        """

        feature = to_list(feature)
        label = to_list(label)
        self.feed_dict.update({self.training_phase: False})
        for i in range(len(self.inputs)):
            self.feed_dict[self.inputs[i]] = feature[i]
        for i in range(len(self.label)):
            self.feed_dict[self.label[i]] = label[i]
        metrics = tf.get_default_session().run(list(self.metrics.values()) + [self.summary_op],
                                               feed_dict=self.feed_dict)
        ret = {}
        for k, v in zip(self.metrics, metrics[:-1]):
            ret[k] = v
        return ret, metrics[-1]

    def test_batch(self, inputs, label=None, **kwargs):
        r"""test one batch

        Args:
            inputs: LR images
            label: if None, return only predicted outputs; else return outputs along with metrics
            kwargs: for future use

        Return:
            predicted outputs, metrics if `label` is not None
        """

        feature = to_list(inputs)
        label = to_list(label)
        self.feed_dict.update({self.training_phase: False})
        for i in range(len(self.inputs)):
            self.feed_dict[self.inputs[i]] = feature[i]
        if label:
            for i in range(len(self.label)):
                self.feed_dict[self.label[i]] = label[i]
            return tf.get_default_session().run(self.outputs + list(self.metrics.values()), feed_dict=self.feed_dict)
        else:
            return tf.get_default_session().run(self.outputs, feed_dict=self.feed_dict)

    def export_model_pb(self, export_dir='.', export_name='model.pb', **kwargs):
        r"""export model as protobuf

        Args:
            export_dir: directory to save the exported model
            export_name: model name
        """
        sess = tf.get_default_session()
        graph = sess.graph.as_graph_def()
        graph = tf.graph_util.remove_training_nodes(graph)
        graph = tf.graph_util.convert_variables_to_constants(
            sess, graph, [outp.name.split(':')[0] for outp in self.outputs])
        tf.train.write_graph(graph, export_dir, export_name, as_text=False)
        print(f"Model exported to [ {Path(export_dir).resolve() / export_name} ].")
