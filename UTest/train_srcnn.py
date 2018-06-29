from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.DataLoader.Loader import BatchLoader
from VSR.Framework.Callbacks import *
from VSR.Models.Srcnn import SRCNN

import tensorflow as tf

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

BATCH_SIZE = 32
EPOCHS = 200
D = DATASETS['91-IMAGE']


def use_lagency_session(*args):
    D.setattr(patch_size=48, random=True, max_patches=BATCH_SIZE * 100)
    loader = BatchLoader(BATCH_SIZE, D, 'train', scale=4, loop=True)
    m = SRCNN(scale=4).compile()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            hr, lr = next(loader)
            metric = m.train_batch(lr, hr)
            print(f'Iter: {i}', end=' ')
            print(metric, flush=True)


def use_environemt(*args):
    D.setattr(patch_size=48, strides=48)
    m = SRCNN(scale=3)
    with Environment(m, f'../Results/{m.name}/save', f'../Results/{m.name}/log') as env:
        env.fit(BATCH_SIZE, EPOCHS, D, learning_rate_schedule=lr_decay('stair', 1e-4, decay_step=1000, decay_rate=0.96))
        env.output_callbacks += [save_image(f'../Results/{m.name}/test')]
        env.test(D)


if __name__ == '__main__':
    tf.app.run(use_environemt)
