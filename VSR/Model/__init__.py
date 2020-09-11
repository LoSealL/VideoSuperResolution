"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-7

Abstract model getter, detailed model please refer to
- VSR/Backend/Torch (pytorch backend)
- VSR/Backend/Keras (tensorflow v2 backend)
- VSR/Backend/TF (tensorflow v1 backend)
"""
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
        elif BACKEND == 'keras':
            return import_module('.Models', 'VSR.Backend.Keras').get_model(name)
    except (KeyError, ImportError):
        raise ImportError(f"Using {BACKEND}, can't find model {name}.")


def list_supported_models():
    if BACKEND == 'pytorch':
        return import_module('.Models', 'VSR.Backend.Torch').list_supported_models()
    elif BACKEND == 'tensorflow':
        return import_module('.Models', 'VSR.Backend.TF').list_supported_models()
    elif BACKEND == 'keras':
        return import_module('.Models', 'VSR.Backend.Keras').list_supported_models()
