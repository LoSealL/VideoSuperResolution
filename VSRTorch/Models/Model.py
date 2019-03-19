#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 13

import torch
import logging

from ..Framework.Trainer import SRTrainer


class BasicModel:

  def __init__(self, **kwargs):
    self.modules = {}
    self.name = ''
    self.writer = None
    self._trainer = None

  def __setattr__(self, key, value):
    if isinstance(value, torch.nn.Module):
      if key in self.modules:
        if self.modules[key] is value:
          return
        else:
          raise NotImplemented
      else:
        self.modules[key] = value
        self.name += f'[{key}]'

    return super(BasicModel, self).__setattr__(key, value)

  def trainable_variables(self, name=None):
    _m = self.modules.get(name) if name else self.modules.values()
    _var = []
    for i in _m:
      _var += filter(lambda p: p.requires_grad, i.parameters())
    return _var

  def to_train(self):
    for _m in self.modules.values():
      _m.train()

  def train(self, *args, **kwargs):
    raise NotImplemented

  def to_eval(self):
    for _m in self.modules.values():
      _m.eval()

  def eval(self, *args, **kwargs):
    raise NotImplemented

  def display(self):
    num_params = 0
    for m in self.modules.values():
      for p in m.parameters():
        num_params += p.nelement()
    logging.getLogger('VSR').info(f"Total params: {num_params}")

  def cuda(self):
    for i in self.modules:
      if torch.cuda.is_available():
        self.modules[i] = self.modules[i].cuda()

  @property
  def trainer(self):
    return self._trainer


class SuperResolution(BasicModel):

  def __init__(self, scale, channel, **kwargs):
    super(SuperResolution, self).__init__(**kwargs)
    self.scale = scale
    self.channel = channel
    self._trainer = SRTrainer
