from Exp.SrGan import SRGAN
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *

import tensorflow as tf


def learning_rate_decay(lr, epochs, steps, **kwargs):
    lr = tf.train.exponential_decay(1e-2, global_step=steps, decay_steps=1000, decay_rate=0.5)
    return lr.eval()


if __name__ == '__main__':
    dataset = load_datasets('../Data/datasets.json')['BSD']
    dataset.setattr(patch_size=96, strides=96, random=True, max_patches=32 * 3000)
    model = SRGAN(scale=4, glayers=16, dlayers=8, vgg_layer=[2, 2], init=True)
    with Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log') as env:
        env.fit(32, 100, dataset, learning_rate_schedule=learning_rate_decay)

    model = SRGAN(scale=4, glayers=16, dlayers=8, vgg_layer=[2, 2], init=False)
    with Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log') as env:
        env.fit(32, 200, dataset, learning_rate_schedule=learning_rate_decay)
        env.feature_callbacks = [to_gray()]
        env.label_callbacks = [to_gray()]
        env.output_callbacks += [lambda output, **kwargs: output[0]]
        env.output_callbacks += [to_rgb()]
        env.output_callbacks += [save_image(f'../Results/{model.name}/test')]
        env.test(dataset, convert_to_gray=False)  # load image with 3 channels
