#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

import importlib

__all__ = ['get_model', 'list_supported_models']

models = {
  # alias: (file, class)
  'srcnn': ('Srcnn', 'SRCNN'),
}


def get_model(name):
  module = f'.Backend.Keras.Models.{models[name][0]}'
  package = 'VSR'
  m = importlib.import_module(module, package)
  return m.__dict__[models[name][1]]


def list_supported_models():
  return models.keys()
