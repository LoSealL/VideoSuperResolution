#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2018 - 8 - 1

import importlib

__all__ = ['get_model', 'list_supported_models']

models = {
  # alias: (file, class)
  'srcnn': ('Srcnn', 'SRCNN'),
  'espcn': ('Espcn', 'ESPCN'),
  'vdsr': ('Vdsr', 'VDSR'),
  'drcn': ('Drcn', 'DRCN'),
  'dncnn': ('DnCnn', 'DnCNN'),
  'idn': ('Idn', 'InformationDistillationNetwork'),
  'rdn': ('Rdn', 'ResidualDenseNetwork'),
  'dcscn': ('Dcscn', 'DCSCN'),
  'lapsrn': ('LapSrn', 'LapSRN'),
  'drrn': ('Drrn', 'DRRN'),
  'memnet': ('MemNet', 'MEMNET'),
  'dbpn': ('Dbpn', 'DBPN'),
  'edsr': ('Edsr', 'EDSR'),
  'srgan': ('SrGan', 'SRGAN'),
  'carn': ('Carn', 'CARN'),
  'rcan': ('Rcan', 'RCAN'),
  'msrn': ('Msrn', 'MSRN'),
  'vespcn': ('Vespcn', 'VESPCN'),
  'srdensenet': ('SRDenseNet', 'SRDenseNet'),
  'srfeat': ('SRFeat', 'SRFEAT'),
  'nlrn': ('Nlrn', 'NLRN'),
  'crdn': ('Crdn', 'CRDN'),
}

_generative_models = {
  'sgan': ('Gan', 'SGAN'),
  'gangp': ('Gan', 'SGANGP'),
  'lsgan': ('Gan', 'LSGAN'),
  'wgan': ('Gan', 'WGAN'),
  'wgangp': ('Gan', 'WGANGP'),
  'rgan': ('Gan', 'RGAN'),
  'rgangp': ('Gan', 'RGANGP'),
  'ragan': ('Gan', 'RaGAN'),
  'ragangp': ('Gan', 'RaGANGP'),
  'rlsgan': ('Gan', 'RLSGAN'),
  'ralsgan': ('Gan', 'RaLSGAN'),
}


def get_model(name):
  module = f'.Models.{models[name][0]}'
  package = 'VSR'
  m = importlib.import_module(module, package)
  return m.__dict__[models[name][1]]


def list_supported_models():
  return models.keys()
