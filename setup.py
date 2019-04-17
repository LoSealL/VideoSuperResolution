"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Last update date: Mar. 25th 2019
"""

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

from setuptools import find_packages
from setuptools import setup

VERSION = '0.7.6'

REQUIRED_PACKAGES = [
  'numpy',
  'scipy',
  'matplotlib',
  'Pillow',
  'pypng',
  'pytest',
  'PyYAML',
  'psutil',
  'tqdm',
  'h5py',
  'easydict >= 1.9',
  'tensorflow >= 1.12.0',
  'google-api-python-client',
  'oauth2client',
]

try:
  import torch

  REQUIRED_PACKAGES.extend([
    'torch >= 1.0.0',
    'torchvision',
    'tensorboardX',
  ])
except ImportError:
  pass

setup(
  name='VSR',
  version=VERSION,
  description='Video Super-Resolution Framework',
  url='https://github.com/LoSealL/VideoSuperResolution',
  packages=find_packages(),
  install_requires=REQUIRED_PACKAGES,
  license='MIT',
  author='Wenyi Tang',
  author_email='wenyitang@outlook.com',
  keywords="super-resolution sr vsr tensorflow pytorch",
)
