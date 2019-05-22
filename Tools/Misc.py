"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Jan 7th, 2019

Misc utility tools
- make TFRecords files
"""

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import io

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


class YUVConverter:
  def __init__(self, frame):
    self.data = frame
    self.length = frame.shape[0]
    self.height = frame.shape[2]
    self.width = frame.shape[3]

  def to_nv12(self):
    # YUV -> NV12
    h_tail = self.height % 2
    w_tail = self.width % 2
    y = np.pad(self.data[:, 0], [[0, 0], [0, h_tail], [0, w_tail]],
               mode='reflect')
    u = self.data[:, 1, ::2, ::2]
    v = self.data[:, 2, ::2, ::2]
    buffer = io.BytesIO()
    for i in range(self.length):
      plain = y[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = np.stack([u[i], v[i]], -1).flatten().astype('uint8').tobytes()
      buffer.write(plain)
    if buffer.tell() != np.prod(self.data.shape) // 2:
      print(" [$] warning: even frame size, crop 1 pixel out")
      assert buffer.tell() == np.prod(y.shape) + np.prod(u.shape) + np.prod(
        v.shape)
      self.width = y.shape[2]
      self.height = y.shape[1]
    return buffer

  def to_nv21(self):
    # YUV -> NV21
    h_tail = self.height % 2
    w_tail = self.width % 2
    y = np.pad(self.data[:, 0], [[0, 0], [0, h_tail], [0, w_tail]],
               mode='reflect')
    u = self.data[:, 2, ::2, ::2]
    v = self.data[:, 1, ::2, ::2]
    buffer = io.BytesIO()
    for i in range(self.length):
      plain = y[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = np.stack([u[i], v[i]], -1).flatten().astype('uint8').tobytes()
      buffer.write(plain)
    if buffer.tell() != np.prod(self.data.shape) // 2:
      print(" [$] warning: even frame size, crop 1 pixel out")
      assert buffer.tell() == np.prod(y.shape) + np.prod(u.shape) + np.prod(
        v.shape)
      self.width = y.shape[2]
      self.height = y.shape[1]
    return buffer

  def to_yv12(self):
    h_tail = self.height % 2
    w_tail = self.width % 2
    y = np.pad(self.data[:, 0], [[0, 0], [0, h_tail], [0, w_tail]],
               mode='reflect')
    u = self.data[:, 1, ::2, ::2]
    v = self.data[:, 2, ::2, ::2]
    buffer = io.BytesIO()
    for i in range(self.length):
      plain = y[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = u[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = v[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
    if buffer.tell() != np.prod(self.data.shape) // 2:
      print(" [$] warning: even frame size, crop 1 pixel out")
      assert buffer.tell() == np.prod(y.shape) + np.prod(u.shape) + np.prod(
        v.shape)
      self.width = y.shape[2]
      self.height = y.shape[1]
    return buffer

  def to_yv21(self):
    h_tail = self.height % 2
    w_tail = self.width % 2
    y = np.pad(self.data[:, 0], [[0, 0], [0, h_tail], [0, w_tail]],
               mode='reflect')
    u = self.data[:, 1, ::2, ::2]
    v = self.data[:, 2, ::2, ::2]
    buffer = io.BytesIO()
    for i in range(self.length):
      plain = y[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = v[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = u[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
    if buffer.tell() != np.prod(self.data.shape) // 2:
      print(" [$] warning: even frame size, crop 1 pixel out")
      assert buffer.tell() == np.prod(y.shape) + np.prod(u.shape) + np.prod(
        v.shape)
      self.width = y.shape[2]
      self.height = y.shape[1]
    return buffer

  def to_yv12(self):
    h_tail = -1 if self.height % 2 else None
    w_tail = -1 if self.width % 2 else None
    y = self.data[:, 0, :h_tail, :w_tail]
    u = self.data[:, 1, ::2, ::2]
    v = self.data[:, 2, ::2, ::2]
    buffer = io.BytesIO()
    for i in range(self.length):
      plain = y[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = u[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = v[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
    if buffer.tell() != np.prod(self.data.shape) // 2:
      print(" [$] warning: even frame size, crop 1 pixel out")
      assert buffer.tell() == np.prod(y.shape) + np.prod(u.shape) + np.prod(
        v.shape)
    return buffer

  def to_yv21(self):
    h_tail = -1 if self.height % 2 else None
    w_tail = -1 if self.width % 2 else None
    y = self.data[:, 0, :h_tail, :w_tail]
    u = self.data[:, 1, ::2, ::2]
    v = self.data[:, 2, ::2, ::2]
    buffer = io.BytesIO()
    for i in range(self.length):
      plain = y[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = v[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
      plain = u[i].flatten().astype('uint8').tobytes()
      buffer.write(plain)
    if buffer.tell() != np.prod(self.data.shape) // 2:
      print(" [$] warning: even frame size, crop 1 pixel out")
      assert buffer.tell() == np.prod(y.shape) + np.prod(u.shape) + np.prod(
        v.shape)
    return buffer

  def to(self, fmt):
    func_name = 'to_' + fmt.lower()
    if hasattr(self, func_name):
      return getattr(self, func_name)()
    raise NotImplementedError(f"Unsupported color format: {fmt}!")
