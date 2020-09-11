"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

PyTorch related unit tests
"""
from .image_test import ImageTest
from .imfilter_test import ImFilterTest
from .motion_test import MotionTest
from .space_to_depth_test import SpaceToDimTest

__all__ = [
    'ImageTest',
    'ImFilterTest',
    'MotionTest',
    'SpaceToDimTest'
]
