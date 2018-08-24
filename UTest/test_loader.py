"""
Unit test for DataLoader.Loader
"""

from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')


def test_loader_init():
    DUT = DATASETS['BSD']
    DUT.setattr(patch_size=48, strides=48, scale=1)
    r = Loader(DUT, 'train')
    r.build_loader()


def test_quickloader_prob():
    DUT = DATASETS['SET5']
    PROB = [0.46196357, 0.14616816, 0.11549089, 0.13816049, 0.13821688]
    r = QuickLoader(1, DUT, 'test', 1)
    MC = 10000
    P = []
    while MC > 0:
        P.append(r._random_select())
        MC -= 1
    epsilon = 1e-2
    assert P.count(0) / 1e4 - PROB[0] <= epsilon
    assert P.count(1) / 1e4 - PROB[1] <= epsilon
    assert P.count(2) / 1e4 - PROB[2] <= epsilon
    assert P.count(3) / 1e4 - PROB[3] <= epsilon
    assert P.count(4) / 1e4 - PROB[4] <= epsilon

    r.change_select_method(Select.EQUAL_FILE)
    MC = 10000
    P = []
    while MC > 0:
        P.append(r._random_select())
        MC -= 1
    epsilon = 1e-2
    PROB = [.2, .2, .2, .2, .2]
    assert P.count(0) / 1e4 - PROB[0] <= epsilon
    assert P.count(1) / 1e4 - PROB[1] <= epsilon
    assert P.count(2) / 1e4 - PROB[2] <= epsilon
    assert P.count(3) / 1e4 - PROB[3] <= epsilon
    assert P.count(4) / 1e4 - PROB[4] <= epsilon


def test_quickloader_prefetch():
    DUT = DATASETS['GOPRO']
    DUT.setattr(patch_size=48, depth=5)
    r = QuickLoader(16, DUT, 'train', 4)
    r.prefetch(0.1)
