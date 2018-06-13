from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.DataLoader.Loader import BatchLoader
from VSR.Framework.Callbacks import *
from VSR.Models.Srcnn import SRCNN
from VSR.Models.Espcn import ESPCN

import tensorflow as tf

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

BATCH_SIZE = 32
EPOCHS = 200


def main(*args):
    d = DATASETS['BSD-500']
    d.setattr(patch_size=48, strides=48, random=False, max_patches=BATCH_SIZE * 100)
    loader = BatchLoader(BATCH_SIZE, d, 'train', scale=4)
    with tf.Session() as sess:
        m = SRCNN(scale=4).compile()
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            loader.reset()
            # image_pairs = [x for x in loader]
            rate = 1e-6 if i > 100 else 1e-4
            for hr, lr in loader:
                loss = m.train_batch(lr, hr, rate)
                print(loss, end='\r')
            print('')


if __name__ == '__main__':
    tf.app.run(main)
