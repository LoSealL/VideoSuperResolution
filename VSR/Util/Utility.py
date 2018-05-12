"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 8th 2018

utility functions
"""
from typing import Generator


def to_list(x, repeat=1):
    """convert x to list object

    Args:
         x: any object to convert
         repeat: if x is to make as [x], repeat `repeat` elements in the list
    """
    if isinstance(x, (Generator, tuple, set)):
        return list(x)
    elif isinstance(x, list):
        return x
    elif isinstance(x, dict):
        return list(x.values())
    else:
        return [x] * repeat

def shrink_mod_scale(x, scale):
    """clip each dim of x to multiple of scale

    """
    scale = to_list(scale, 2)
    mod_x = []
    for _x, _s in zip(x, scale):
        mod_x.append(_x - _x % _s)
    return mod_x
