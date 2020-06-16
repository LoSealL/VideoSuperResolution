#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import logging
from collections import OrderedDict

import torch

from ..Framework.Trainer import SRTrainer


class BasicModel(object):
  """Trainable model wrapper for PyTorch nn.Module objects

  There are 2 built-in attributes:
    - modules: contains a K-V pair of `str: nn.Module`. It will be automatically
      appended if a derived object assign any attribute with `nn.Module` object.
    - opts: contains a K-V pair of `str: optim.Optimizer`. Will be automatically
      appended if a derived object assign any attribute with `optim.Optimizer`.
  """
  modules = OrderedDict()
  opts = OrderedDict()
  name = ''
  loaded = None
  _trainer = None

  def __setattr__(self, key, value):
    if key in ('modules', 'opts',):
      if hasattr(self, key):
        raise ValueError(f"Can't overwrite built-in '{key}' of BasicModel")
    if isinstance(value, torch.nn.Module):
      if key in self.modules:
        if self.modules[key] is value:
          return
        else:
          # TODO: why assign twice??
          raise NotImplementedError
      elif len(list(value.parameters())):
        self.modules[key] = value
        self.name += f'[{key}]'
    if isinstance(value, torch.optim.Optimizer):
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
      _var += filter(lambda p: p.requires_grad, i.parameters())
    return _var

  def to_train(self):
    """Change modules to train mode."""
    for _m in self.modules.values():
      _m.train()

  def train(self, *args, **kwargs):
    """Forward and backward data path.
      The trainer knows data pipeline through this callback."""
    raise NotImplementedError

  def to_eval(self):
    """Change modules to evaluate mode."""
    for _m in self.modules.values():
      _m.eval()

  def eval(self, *args, **kwargs):
    """Forward data path. No backward needed for this is only for testing."""
    raise NotImplementedError

  def display(self):
    """Show model info"""
    num_params = 0
    for m in self.modules.values():
      for p in m.parameters():
        num_params += p.nelement()
    logging.getLogger('VSR').info(f"Total params: {num_params}")

  def cuda(self):
    """Move model to cuda device."""
    for i in self.modules:
      if torch.cuda.is_available():
        self.modules[i] = self.modules[i].cuda()

  def export(self, export_dir):
    """export ONNX model.

    Args:
      export_dir: path to save onnx files.
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

  def load(self, pth, map_location=None):
    for key, model in self.modules.items():
      if not isinstance(pth, dict):
        self.sequential_load(model, str(pth), map_location)
        break
      self.sequential_load(model, str(pth[key]), map_location)
    self.loaded = True
    for key, opt in self.opts.items():
      if isinstance(pth, dict):
        opt.load_state_dict(
            torch.load(str(pth[key]), map_location=map_location))

  @staticmethod
  def sequential_load(module, pth, map_location=None):
    state_dict = torch.load(pth, map_location=map_location)
    p = module.state_dict()
    while len(state_dict) and len(p):
      saved_name, saved_data = state_dict.popitem()
      name, buffer = p.popitem()
      if saved_name != name:
        logging.getLogger('VSR').warning(
            f"unmatched name: expected {name}, got {saved_name}.")
      if buffer.shape == saved_data.shape:
        buffer.data.copy_(saved_data)
      else:
        logging.getLogger('VSR').error(
            f"Checkpoint shape mismatch for {name}, "
            f"expected {buffer.shape}, but got {saved_data.shape}")
        raise ValueError
    while len(state_dict):
      saved_name, _ = state_dict.popitem()
      logging.getLogger('VSR').warning(f"Unexpected keys: {saved_name}")
    while len(p):
      name, _ = p.popitem()
      logging.getLogger('VSR').warning(f"Missing keys: {saved_name}")


class SuperResolution(BasicModel):
  """A default model for (video) super-resolution"""

  def __init__(self, scale, channel, **kwargs):
    super(SuperResolution, self).__init__(**kwargs)
    self.scale = scale
    self.channel = channel
    # Default SR trainer
    self._trainer = SRTrainer
