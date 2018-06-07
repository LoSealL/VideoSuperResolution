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
        start = time.time()
        loader = Loader(datasets['MCL-VIDEO'], 'train')
        try:
            next(loader)
            print('Unexpected: load before built')
        except RuntimeError as ex:
            print('Expected: ' + str(ex))

        loader.build_loader(scale=2, patch_size=48, strides=48, depth=7)
        print('Len: %d' % len(loader))
        # for _ in loader:
        #     pass
        print(f'frames per second: {100 / (time.time() - start + 1e-6):.6f} fps')
    except KeyError:
        pass

    data = datasets['91-IMAGE']
    data.setattr(patch_size=32, strides=14)
    batch_loader = BatchLoader(32, data, 'train', scale=1)
    print("Len: %d" % len(batch_loader))
    start = time.time()
    sz = len(list(batch_loader))
    print(f'Time: {time.time() - start}s. Count: {sz}')
    data.setattr(random=True, max_patches=sz * 32)
    start = time.time()
    batch_loader = BatchLoader(32, data, 'train', scale=1)
    print("Len: %d" % len(batch_loader))
    sz = len(list(batch_loader))
    print(f'Time: {time.time() - start}s. Count: {sz}')
