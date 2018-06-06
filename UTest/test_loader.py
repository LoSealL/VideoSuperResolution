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
    try:
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
    except KeyError:
        pass

    data = datasets['91-IMAGE']
    data.setattr(patch_size=48, strides=12)
    batch_loader = BatchLoader(32, data, 'train', scale=1)
    start = time.time()
    sz = len(list(batch_loader))
    print(f'Time: {time.time() - start}s. Count: {sz}')
    data.setattr(random=True, max_patches=sz * 32)
    start = time.time()
    batch_loader = BatchLoader(32, data, 'train', scale=1)
    sz = len(list(batch_loader))
    print(f'Time: {time.time() - start}s. Count: {sz}')


    def g(a, b):
        for i in range(a):
            for j in range(b):
                yield a * b


    g1 = g(1, 100000)
    g2 = g(100000, 1)
    start = time.time()
    list(g1)
    print(f'Time: {time.time() - start}s.')
    start = time.time()
    list(g2)
    print(f'Time: {time.time() - start}s.')
