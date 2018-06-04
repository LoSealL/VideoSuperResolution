from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *
from VSR.Models.DnCnn import DnCNN
from pathlib import Path


def add_noise(low=0, high=55):
    import numpy as np

    def feature_callback(feature):
        sigma = np.random.randint(low, high)
        # sigma = 25
        noise = np.random.normal(0, sigma, size=feature.shape)
        return feature + noise

    return feature_callback


if __name__ == '__main__':
    model = DnCNN(layers=20, rgb_input=False).compile()
    dataset = load_datasets('../Data/datasets.json')['BSD']
    dataset.setattr(patch_size=40, strides=40, depth=1)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log', feature_index=0, label_index=0)
    env.feature_callbacks = [add_noise(0, 55)]
    env.fit(128, 100, dataset, restart=False)
    env.feature_callbacks = []
    env.output_callbacks = [lambda output, **kwargs: output[0], reduce_residual()]
    env.output_callbacks += [save_image(model.name)]
    env.predict(list(Path('srcnn').glob('*.png')))
    # env.test(dataset)
    env.export(f'{model.name}')
