"""
Unit test for DataLoader.Loader
"""
import time

from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *

if __name__ == '__main__':
    """ Test """
    datasets = load_datasets('../Data/datasets.json')
    for d in datasets.values():
        try:
            Loader(d, 'train')
        except ValueError as ex:
            print(f'{d.name} load training set failed: {ex}')
        try:
            Loader(d, 'val')
        except ValueError as ex:
            print(f'{d.name} load validation set failed: {ex}')
        try:
            Loader(d, 'test')
        except ValueError as ex:
            print(f'{d.name} load test set failed: {ex}')
    loader = Loader(datasets['MCL-V'], 'test')
    try:
        next(loader)
        print('Unexpected: load before built')
    except RuntimeError as ex:
        print('Expected: ' + str(ex))
    loader.build_loader(2, 48, 48, 7)
    start = time.time()
    for _ in range(100):
        next(loader)
    print(f'frames per second: {100 / (time.time() - start + 1e-6):.6f} fps')

    batch_loader = BatchLoader(100, datasets['91-IMAGE'], 'train')
    start = time.time()
    load_cnt = 0
    for hr, lr in batch_loader:
        assert hr.ndim == 5 or hr.ndim == 4
        assert lr.ndim == 5 or lr.ndim == 4
        load_cnt += 1
    print(f'Time: {time.time() - start}s. Count: {load_cnt}')
