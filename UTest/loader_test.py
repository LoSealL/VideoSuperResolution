"""
Unit test for DataLoader.Loader
"""

import numpy as np
from VSR.DataLoader.Loader import BasicLoader, QuickLoader, Select
from VSR.Util import ImageProcess
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.yaml')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.yaml')


def test_loader_prob():
    dut = DATASETS['SET5']
    prob = [0.46196357, 0.14616816, 0.11549089, 0.13816049, 0.13821688]
    config = Config(batch=1, scale=1, depth=1, steps_per_epoch=-1,
                    convert_to='RGB')
    r = BasicLoader(dut, 'test', config)
    mc = 10000
    p = r._random_select(mc).values()
    epsilon = 1e-2
    for p, p_hat in zip(p, prob):
        assert np.abs(p / 1e4 - p_hat) <= epsilon

    r.change_select_method(Select.EQUAL_FILE)
    mc = 10000
    p = r._random_select(mc).values()
    epsilon = 1e-2
    prob = [.2, .2, .2, .2, .2]
    for p, p_hat in zip(p, prob):
        assert np.abs(p / 1e4 - p_hat) <= epsilon


def test_basicloader_iter():
    dut = DATASETS['91-IMAGE']
    config = Config(batch=16, scale=4, depth=1, steps_per_epoch=200,
                    convert_to='RGB', crop='random')
    config.patch_size = 48
    r = BasicLoader(dut, 'train', config, True)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)


def test_quickloader_iter():
    dut = DATASETS['DIV2K']
    config = Config(batch=16, scale=4, depth=1, steps_per_epoch=200,
                    convert_to='RGB', crop='random')
    config.patch_size = 48
    r = QuickLoader(dut, 'train', config, True, n_threads=8)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)


def test_benchmark_basic():
    dut = DATASETS['DIV2K']
    epochs = 4
    config = Config(batch=8, scale=4, depth=1, patch_size=196,
                    steps_per_epoch=100, convert_to='RGB', crop='random')
    loader = BasicLoader(dut, 'train', config, True)
    for _ in range(epochs):
        r = loader.make_one_shot_iterator()
        list(r)


def test_benchmark_mp():
    dut = DATASETS['DIV2K']
    epochs = 4
    config = Config(batch=8, scale=4, depth=1, patch_size=196,
                    steps_per_epoch=100, convert_to='RGB', crop='random')
    loader = QuickLoader(dut, 'train', config, True, n_threads=8)
    for _ in range(epochs):
        r = loader.make_one_shot_iterator()
        list(r)


def test_read_flow():
    from VSR.Framework.Callbacks import _viz_flow
    dut = DATASETS['MINICHAIRS']
    config = Config(batch=8, scale=1, depth=2, patch_size=96,
                    steps_per_epoch=100, convert_to='RGB', crop='random')
    loader = QuickLoader(dut, 'train', config, True, n_threads=8)
    r = loader.make_one_shot_iterator('1GB', shuffle=True)
    loader.prefetch('1GB')
    list(r)
    r = loader.make_one_shot_iterator('8GB', shuffle=True)
    img, flow, name = list(r)[0]

    ref0 = img[0, 0, ...]
    ref1 = img[0, 1, ...]
    u = flow[0, 0, ..., 0]
    v = flow[0, 0, ..., 1]
    ImageProcess.array_to_img(ref0, 'RGB').show()
    ImageProcess.array_to_img(ref1, 'RGB').show()
    ImageProcess.array_to_img(_viz_flow(u, v), 'RGB').show()


def test_cifar_loader():
    from tensorflow.keras.datasets import cifar10
    from tqdm import tqdm
    train, test = cifar10.load_data()
    train_data, _ = train
    train_set = Dataset(train=[train_data], mode='numpy')
    config = Config(batch=8, scale=1, depth=2, patch_size=32,
                    steps_per_epoch=100, convert_to='RGB')
    loader = BasicLoader(train_set, 'train', config, False)
    r = loader.make_one_shot_iterator()
    list(tqdm(r))


def test_memory_usage():
    dut = DATASETS['GOPRO']
    epochs = 4
    config = Config(batch=16, scale=4, depth=1, patch_size=196,
                    steps_per_epoch=100, convert_to='RGB', crop='random')
    loader = QuickLoader(dut, 'train', config, True, n_threads=8)
    for i in range(epochs):
        it = loader.make_one_shot_iterator('1GB', True)
        loader.prefetch('1GB')
        list(it)


def main():
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG)
    test_memory_usage()
    pass


if __name__ == '__main__':
    main()
