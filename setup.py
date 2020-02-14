# ##############################################################################
#  Copyright (c) 2020. LoSealL All Rights Reserved.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Date: 2020 - 1 - 31
# ##############################################################################

from setuptools import find_packages
from setuptools import setup

# Get version from CHANGELOG
try:
  with open('CHANGELOG.md') as fd:
    VERSION = fd.readline()[:-1]
except IOError:
  VERSION = '0.0.0'

REQUIRED_PACKAGES = [
  'numpy',
  'scipy',
  'scikit-image',
  'matplotlib',
  'pillow',
  'pypng',
  'pytest',
  'PyYAML',
  'psutil',
  'tqdm',
  'h5py',
  'easydict >= 1.9',
  'google-api-python-client',
  'oauth2client',
]

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
  keywords="super-resolution sr vsr cnn srcnn vespcn",
)
