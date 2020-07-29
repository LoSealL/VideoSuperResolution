#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import os
import logging
from importlib import import_module
from pathlib import Path

import yaml

try:
  from yaml import FullLoader as _Loader
except ImportError:
  # For older versions
  from yaml import Loader as _Loader

LOG = logging.getLogger('VSR')
HOME = os.environ.get('VSR_HOME')
if not HOME:
  HOME = Path('~').expanduser() / '.vsr'
CONFIG = {
  'backend': os.environ.get('VSR_BACKEND', 'pytorch'),
  'verbose': os.environ.get('VSR_VERBOSE', 'info'),
}
if Path(HOME / 'config.yml').exists():
  with open(HOME / 'config.yml', encoding='utf8') as fd:
    CONFIG = yaml.load(fd.read(), Loader=_Loader)

LOG.setLevel(CONFIG['verbose'].upper())
hdl = logging.StreamHandler()
hdl.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
LOG.addHandler(hdl)

BACKEND = CONFIG['backend'].lower()
if BACKEND == 'auto':
  BACKEND = 'tensorflow'
if BACKEND not in ('tensorflow', 'keras', 'pytorch'):
  BACKEND = 'pytorch'

if BACKEND in ('tensorflow', 'keras'):
  try:
    tf = import_module('tensorflow')
    CONFIG['data_format'] = 'channels_last'
    tf_ver_major, tf_ver_minor, _ = [int(s) for s in tf.__version__.split('.')]
    if BACKEND == 'keras' and tf_ver_major < 2:
      LOG.warning(f"[!] Current tensorflow version is {tf.__version__}")
      LOG.info("[*] Fallback to use legacy tensorflow v1.x")
      BACKEND = 'tensorflow'
    if tf_ver_major == 1 and tf_ver_minor < 15:
      LOG.warning("[!!] VSR does not support TF < 1.15.0 any longer.")
      LOG.warning("[!] Considering use an old version of VSR, "
                  "or update your tensorflow version.")
      raise ImportError
  except ImportError:
    LOG.warning("[!] Tensorflow package not found in your system.")
    LOG.info("[*] Fallback to use PyTorch...")
    BACKEND = 'pytorch'

if BACKEND == 'pytorch':
  try:
    torch = import_module('torch')
    CONFIG['data_format'] = 'channels_first'
    _ver = torch.__version__.split('.')
    if _ver[0] != '1' or _ver[1] <= '1':
      LOG.warning(
          f"[!] PyTorch version too low: {torch.__version__}, recommended 1.2.0")
  except ImportError:
    LOG.fatal("[!] PyTorch package not found in your system.")
    raise ImportError("Not an available backend found! Check your environment.")

DATA_FORMAT = CONFIG['data_format'].lower()
