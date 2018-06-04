from VSR.Util.Utility import Vgg
from VSR.DataLoader.Dataset import load_datasets
from VSR.DataLoader.Loader import BatchLoader

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    data = load_datasets('../Data/datasets.json')['91-IMAGE']
    loader = BatchLoader(1, data, 'test', convert_to_gray=True, crop=False, scale=1)
    m = Vgg(input_shape=[None, None, 3])
    for hr, lr in loader:
        y = m(hr, [2,3], [2,3])
