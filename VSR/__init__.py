"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com

Video/Single-Image Super-Resolution Framework
- Support multiple backends, such as `tensorflow`, `tensorflow v2 (keras)` and `pytorch`
- Support many state-of-the-art SISR/VSR models, networks and architectures
- Provides highly comprehensive API to load dataset and training models
- Detailed documents and sample codes
"""
from __future__ import absolute_import, print_function

from . import Backend
from . import DataLoader
from . import Model
from . import Util

__all__ = [
    'Backend',
    'DataLoader',
    'Model',
    'Util',
]

__version__ = '1.0.8'
