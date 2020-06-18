#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

import logging

import tensorflow as tf

from .. import LOG
from ..Framework.Trainer import SRTrainer


class BasicModel:
  """Trainable model wrapper for keras.Model objects

  There are 2 built-in attributes:
    - modules: contains a K-V pair of `str: Model`. It will be automatically
      appended if a derived object assign any attribute with `Model` object.
    - opts: contains a K-V pair of `str: Optimizer`. Will be automatically
      appended if a derived object assign any attribute with `Optimizer`.
  """

  def __init__(self, **kwargs):
    self.modules = {}
    self.opts = {}
    self.name = kwargs.get('name', 'model')
    self._trainer = None

  def __setattr__(self, key, value):
    if key in ('modules', 'opts',):
      if hasattr(self, key):
        raise ValueError(f"Can't overwrite built-in '{key}' of BasicModel")
    if isinstance(value, tf.keras.Model):
      if key in self.modules:
        if self.modules[key] is value:
          return
        else:
          # TODO: why assign twice??
          raise NotImplementedError
      else:
        self.modules[key] = value
    if isinstance(value, tf.keras.optimizers.Optimizer):
      if key in self.opts:
        if self.opts[key] is value:
          return
        else:
          raise NotImplementedError
      else:
        self.opts[key] = value

    return super(BasicModel, self).__setattr__(key, value)

  def trainable_variables(self, name=None):
    """Return variables who require gradients.

    Args:
      name: module name. Will return all trainable variables if no name given.
    """

    _m = [self.modules.get(name)] if name else self.modules.values()
    _var = []
    for i in _m:
      _var += i.trainable_variables
    return _var

  def to_train(self):
    """Change modules to train mode."""
    pass

  def train(self, *args, **kwargs):
    """Forward and backward data path.
      The trainer knows data pipeline through this callback."""
    raise NotImplementedError

  def to_eval(self):
    """Change modules to evaluate mode."""
    pass

  def eval(self, *args, **kwargs):
    """Forward data path. No backward needed for this is only for testing."""
    raise NotImplementedError

  def display(self):
    """Show model info"""
    num_params = 0
    for m in self.modules.values():
      for p in m.variables:
        num_params += p.get_shape().num_elements()
    LOG.info(f"Total params: {num_params}")
    if LOG.isEnabledFor(logging.DEBUG):
      [v.summary() for v in self.modules.values()]

  def cuda(self):
    """Move model to cuda device."""
    pass

  def distributed(self):
    pass

  def export(self, export_dir):
    """export keras model.

    Args:
      export_dir: path to save pb files.
    """

    raise NotImplementedError("Should implement in specific model!")

  @property
  def executor(self):
    """Return the trainer class type for this model."""
    return self.get_executor(None)

  def get_executor(self, root):
    if issubclass(self._trainer.__class__, type):
      self._trainer = self._trainer(self, root)
      return self._trainer
    else:
      return self._trainer

  def load(self, ckpt):
    for key, model in self.modules.items():
      if not isinstance(ckpt, dict):
        model.load_weights(str(ckpt))
        break
      model.load_weights(str(ckpt[key]))


class SuperResolution(BasicModel):
  """A default model for (video) super-resolution"""

  def __init__(self, scale, channel, **kwargs):
    super(SuperResolution, self).__init__(**kwargs)
    self.scale = scale
    self.channel = channel
    # Default SR trainer
    self._trainer = SRTrainer
