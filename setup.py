#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 16

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
]

with open('README.md', 'r', encoding='utf-8') as fd:
  long_desp = fd.read()

setup(
    name='VSR',
    version=VERSION,
    description='Video Super-Resolution Framework',
    long_description=long_desp,
    long_description_content_type="text/markdown",
    url='https://github.com/LoSealL/VideoSuperResolution',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    license='MIT',
    author='Wenyi Tang',
    author_email='wenyitang@outlook.com',
    keywords="super-resolution sr vsr cnn srcnn vespcn",
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
