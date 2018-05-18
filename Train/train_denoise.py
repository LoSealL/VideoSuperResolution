from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *
from Models.Denoise import Denoise
from pathlib import Path


def add_noise(sigma=10):
    import numpy as np

    def feature_callback(feature):
        return feature + np.random.normal(0, sigma, size=feature.shape)

    return feature_callback


if __name__ == '__main__':
    model = Denoise()
    model.compile()
    dataset = load_datasets('../Data/datasets.json')['91-IMAGE']
    dataset.setattr(patch_size=33, strides=11, depth=1)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log', feature_index=0, label_index=0)
    env.feature_callbacks = [add_noise(25)]
    env.fit(128, 100, dataset, restart=False)
    env.feature_callbacks = []
    env.output_callbacks = [lambda output, **kwargs: output[0], reduce_residual()]
    env.output_callbacks += [save_image(model.name)]
    # env.predict(list(Path('srcnn').glob('*.png')))
    # env.test(dataset)
