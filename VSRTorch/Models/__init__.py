#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/2 上午10:54

import importlib

__all__ = ['get_model', 'list_supported_models']

models = {
  # alias: (file, class)
  'espcn': ('Espcn', 'ESPCN'),
  'edsr': ('Edsr', 'EDSR'),
  'carn': ('Carn', 'CARN'),
  'dbpn': ('Dbpn', 'DBPN'),
  'rcan': ('Rcan', 'RCAN'),
  'esrgan': ('Esrgan', 'ESRGAN'),
  'msrn': ('Msrn', 'MSRN'),
  'rsr': ('Rsr', 'RSR'),
  'drn': ('Drn', 'DRN'),
  'sofvsr': ('Sofvsr', 'SOFVSR'),
  'vespcn': ('Vespcn', 'VESPCN'),
}

def get_model(name):
  module = f'VSRTorch.Models.{models[name][0]}'
  package = 'VSR'
  m = importlib.import_module(module, package)
  return m.__dict__[models[name][1]]


def list_supported_models():
  return models.keys()
