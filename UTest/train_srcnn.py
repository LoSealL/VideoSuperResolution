from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Callbacks import *
from VSR.Models.Srcnn import SRCNN

import tensorflow as tf

if __name__ == '__main__':
    model = SRCNN(scale=3)
    dataset = load_datasets('../Data/datasets.json')['91-IMAGE']
    dataset.setattr(patch_size=48, strides=12, random=False, max_patches=64 * 100)
    with Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log') as env:
        env.fit(64, 20, dataset, shuffle=False, restart=True, learning_rate=1e-4)
        env.output_callbacks = [save_image(f'../Results/{model.name}/test')]
        env.test(dataset)
