"""
Unit test for DataLoader.Loader
"""
import time

from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

BATCH_SIZE = 64
RANDOM = False

if __name__ == '__main__':
    """ Test """
    for d in DATASETS.values():
        d.setattr(patch_size=48, strides=48)
        try:
            Loader(d, 'train')
        except ValueError as ex:
            print(f'{D.name} load training set failed: {ex}')
        try:
            Loader(d, 'val')
        except ValueError as ex:
            print(f'{D.name} load validation set failed: {ex}')
        try:
            Loader(d, 'test')
        except ValueError as ex:
            print(f'{D.name} load test set failed: {ex}')

    # Test Reset
    loader = BatchLoader(1, DATASETS['91-IMAGE'], 'train', 3)
    print(len(loader))
    print(len(list(loader)))
    loader.reset()
    print(len(loader))
    print(len(list(loader)))

    print(f'Using batch={BATCH_SIZE}, random={RANDOM}', flush=True)
    for k, v in DATASETS.items():
        print(f'Benchmark for dataset {k}')
        v.setattr(random=RANDOM, max_patches=100 * BATCH_SIZE)
        start = time.time()
        loader = BatchLoader(BATCH_SIZE, v, 'train', 4)
        init_time = time.time()
        print(f'construct time: {(init_time - start)*1e3:.6f}ms')
        for hr, lr in loader:
            pass
        end = time.time()
        print(f'Iterate time: {(end - init_time)*1e3:.6f}ms', flush=True)
