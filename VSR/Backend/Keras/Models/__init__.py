"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com

all keras models
"""
import importlib
import logging

__all__ = ['get_model', 'list_supported_models', 'LOG']

models = {
    # alias: (file, class)
    'srcnn': ('Srcnn', 'SRCNN'),
}

LOG = logging.Logger('VSR.Keras.Models')


def get_model(name):
    module = f'.Backend.Keras.Models.{models[name][0]}'
    package = 'VSR'
    m = importlib.import_module(module, package)
    return m.__dict__[models[name][1]]


def list_supported_models():
    return models.keys()
