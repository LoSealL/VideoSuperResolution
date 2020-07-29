#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:10

import importlib

__all__ = ['get_model', 'list_supported_models']

models = {
  'cubic': ('Bicubic', 'BICUBIC'),
  # alias: (file, class)
  'espcn': ('Classic', 'ESPCN'),
  'srcnn': ('Classic', 'SRCNN'),
  'vdsr': ('Classic', 'VDSR'),
  'dncnn': ('Classic', 'DNCNN'),
  'drcn': ('Classic', 'DRCN'),
  'drrn': ('Classic', 'DRRN'),
  'ffdnet': ('Ffdnet', 'FFDNET'),
  'edsr': ('Edsr', 'EDSR'),
  'carn': ('Carn', 'CARN'),
  'dbpn': ('Dbpn', 'DBPN'),
  'rcan': ('Rcan', 'RCAN'),
  'srfeat': ('SRFeat', 'SRFEAT'),
  'esrgan': ('Esrgan', 'ESRGAN'),
  'msrn': ('Msrn', 'MSRN'),
  'crdn': ('Crdn', 'CRDN'),
  'mldn': ('Mldn', 'MLDN'),
  'drn': ('Drn', 'DRN'),
  'sofvsr': ('Sofvsr', 'SOFVSR'),
  'vespcn': ('Vespcn', 'VESPCN'),
  'frvsr': ('Frvsr', 'FRVSR'),
  'qprn': ('Qprn', 'QPRN'),
  'ufvsr': ('Ufvsr', 'UFVSR'),
  'yovsr': ('Yovsr', 'YOVSR'),
  'tecogan': ('TecoGAN', 'TeCoGAN'),
  'spmc': ('Spmc', 'SPMC'),
  'rbpn': ('Rbpn', 'RBPN'),
  'srmd': ('Srmd', 'SRMD'),
  # NTIRE 2019 Collections
  'didn': ('NTIRE19', 'DIDN'),
  'dhdn': ('NTIRE19', 'DHDN'),
  'grdn': ('NTIRE19', 'GRDN'),
  'resunet': ('NTIRE19', 'ResUNet'),
  'edrn': ('NTIRE19', 'EDRN'),
  'frn': ('NTIRE19', 'FRN'),
  'ran': ('NTIRE19', 'RAN'),
  # NTIRE 2020
  'realsr': ('NTIRE20', 'RealSR'),
  'esr': ('EfficientSR', 'ESR')
}


def get_model(name):
  module = f'.Backend.Torch.Models.{models[name][0]}'
  package = 'VSR'
  m = importlib.import_module(module, package)
  return m.__dict__[models[name][1]]


def list_supported_models():
  return models.keys()
