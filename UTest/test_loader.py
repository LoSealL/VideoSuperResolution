"""
Unit test for DataLoader.Loader
"""
import time
import cProfile

from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

BATCH_SIZE = 64
RANDOM = False

if __name__ == '__main__':
    div2k = DATASETS['DIV2K']
    div2k.setattr(patch_size=192, max_patches=200 * 64, random=True)
    cProfile.run('BatchLoader(BATCH_SIZE, div2k, \'train\', 4, convert_to=\'RGB\')')
