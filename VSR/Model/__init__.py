#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

from importlib import import_module

from ..Backend import BACKEND

__all__ = [
  'get_model',
  'list_supported_models'
]


def get_model(name: str):
  name = name.lower()
  try:
    if BACKEND == 'pytorch':
      return import_module('.Models', 'VSR.Backend.Torch').get_model(name)
    elif BACKEND == 'tensorflow':
      return import_module('.Models', 'VSR.Backend.TF').get_model(name)
    elif BACKEND == 'tensorflow2':
      pass
  except (KeyError, ImportError):
    raise ImportError(f"Using {BACKEND}, can't find model {name}.")


def list_supported_models():
  if BACKEND == 'pytorch':
    return import_module('.Models', 'VSR.Backend.Torch').list_supported_models()
  elif BACKEND == 'tensorflow':
    return import_module('.Models', 'VSR.Backend.TF').list_supported_models()
  elif BACKEND == 'tensorflow2':
    pass
