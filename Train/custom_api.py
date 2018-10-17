"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: July 31st 2018

custom feature callback.
Usage:
    pass the function name to `run.py` args `--add_custom_callbacks func1 --add_custom_callbacks func2`.
    During training and testing, the `Trainer` will call the given functions to process input data

Note:
    functions must keep `kwargs` to ignore un-parsed arguments
"""

from PIL import Image
import numpy as np


def color_inverse(inputs, name, **kwargs):
    type_old = inputs.dtype
    inputs = inputs.astype('float')
    return np.clip(255 - inputs, 0, 255).astype(type_old)


def color_shift(inputs, name, **kwargs):
    type_old = inputs.dtype
    inputs = inputs.astype('float')
    return np.clip(inputs - 50, 0, 255).astype(type_old)


def red(inputs, **kwargs):
    x = np.zeros_like(inputs)
    x[..., 0] = 255
    return x


def green(inputs, **kwargs):
    x = np.zeros_like(inputs)
    x[..., 1] = 255
    return x


def blue(inputs, **kwargs):
    x = np.zeros_like(inputs)
    x[..., 2] = 255
    return x


def chessboard(inputs, **kwargs):
    x = np.zeros_like(inputs)
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i::64, j::64, :] = c
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i + 32::64, j::64, :] = c
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i::64, j + 32::64, :] = c
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i + 32::64, j + 32::64, :] = c
    return x


def noisy(inputs, **kwargs):
    shape = inputs.shape
    return np.random.normal(0, 1, shape)
