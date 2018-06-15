from VSR.Framework.Envrionment import Environment as Env
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Callbacks import lr_decay

import tensorflow as tf
import numpy as np

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

BATCH_SIZE = 32
EPOCHS = 50

from Exp import Exp


def main(*args):
    m = Exp.SRCNN(scale=3)
    d = DATASETS['91-IMAGE']
    d.setattr(patch_size=48, strides=48, random=False, max_patches=BATCH_SIZE * 100)
    with Env(m, '../Results/Exp/save', '../Results/Exp/log') as env:
        env.fit(BATCH_SIZE, EPOCHS, d)


if __name__ == '__main__':
    tf.app.run(main)
