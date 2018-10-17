"""
Unit test for DataLoader.Loader
"""

import numpy as np
from VSR.DataLoader.Loader import BasicLoader, QuickLoader, Select
from VSR.Util import ImageProcess
from VSR.Util.Config import Config
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')


def test_quickloader_prob():
    DUT = DATASETS['SET5']
    PROB = [0.46196357, 0.14616816, 0.11549089, 0.13816049, 0.13821688]
    config = Config(batch=1, scale=1, depth=1, steps_per_epoch=-1, convert_to='RGB')
    r = BasicLoader(DUT, 'test', config)
    MC = 10000
    P = r._random_select(MC).values()
    epsilon = 1e-2
    for p, p_hat in zip(P, PROB):
        assert np.abs(p / 1e4 - p_hat) <= epsilon

    r.change_select_method(Select.EQUAL_FILE)
    MC = 10000
    P = r._random_select(MC).values()
    epsilon = 1e-2
    PROB = [.2, .2, .2, .2, .2]
    for p, p_hat in zip(P, PROB):
        assert np.abs(p / 1e4 - p_hat) <= epsilon


def test_quickloader_iter():
    DUT = DATASETS['DIV2K']
    config = Config(batch=16, scale=4, depth=1, steps_per_epoch=200, convert_to='RGB', crop='random')
    r = BasicLoader(DUT, 'train', config, True)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)


def test_mploader_iter():
    DUT = DATASETS['DIV2K']
    config = Config(batch=16, scale=4, depth=1, steps_per_epoch=200, convert_to='RGB', crop='random')
    r = QuickLoader(DUT, 'train', config, True)
    it = r.make_one_shot_iterator('8GB', 8)
    for hr, lr, name in it:
        print(name, flush=True)
    it = r.make_one_shot_iterator('8GB', 8)
    for hr, lr, name in it:
        print(name, flush=True)


def test_benchmark_quickloader():
    DUT = DATASETS['DIV2K']
    EPOCHS = 4
    config = Config(batch=8, scale=4, depth=1, patch_size=196, steps_per_epoch=100, convert_to='RGB', crop='random')
    l = BasicLoader(DUT, 'train', config, True)
    for _ in range(EPOCHS):
        r = l.make_one_shot_iterator()
        for hr, lr, name in r:
            pass


def test_benchmark_mploader():
    DUT = DATASETS['DIV2K']
    EPOCHS = 4
    config = Config(batch=8, scale=4, depth=1, patch_size=196, steps_per_epoch=100, convert_to='RGB', crop='random')
    l = QuickLoader(DUT, 'train', config, True, n_threads=8)
    for _ in range(EPOCHS):
        r = l.make_one_shot_iterator()
        for hr, lr, name in r:
            pass


def test_read_flow():
    DUT = DATASETS['MINICHAIRS']
    DUT.setattr(patch_size=96, depth=2)
    config = Config(batch=8, scale=1, depth=2, patch_size=96, steps_per_epoch=100, convert_to='RGB')
    l = QuickLoader(DUT, 'train', config, True, n_threads=8)
    r = l.make_one_shot_iterator('8GB', shuffle=True)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)
    r = l.make_one_shot_iterator('8GB', shuffle=True)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)

    ref0 = img[0, 0, ...]
    ref1 = img[0, 1, ...]
    u = flow[0, 0, ..., 0]
    v = flow[0, 0, ..., 1]
    H, W = u.shape
    ImageProcess.array_to_img(ref0, 'RGB').show()
    ImageProcess.array_to_img(ref1, 'RGB').show()
    u = (u / W + 1) / 2 * 255
    v = (v / H + 1) / 2 * 255
    ImageProcess.array_to_img(u, 'L').show()
    ImageProcess.array_to_img(v, 'L').show()


def test_mp():
    import time
    DUT = DATASETS['91-IMAGE']
    config = Config(batch=4, scale=4, depth=1, patch_size=56, steps_per_epoch=200, convert_to='RGB', crop='random')
    l1 = QuickLoader(DUT, 'train', config, True, n_threads=4)
    start = time.time()
    it = l1.make_one_shot_iterator()
    res = list(it)
    print(time.time() - start)


if __name__ == '__main__':
    test_mp()
