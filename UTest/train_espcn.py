from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Callbacks import *
from VSR.Models.Espcn import Espcn


def add_noise(sigma=10):
    import numpy as np

    def feature_callback(feature):
        return feature + np.random.normal(0, sigma, size=feature.shape)

    return feature_callback


def sub_res(input, output, **kwargs):
    residual, pred_hr = output[0], output[1]
    return pred_hr - residual


if __name__ == '__main__':
    model = Espcn(scale=3)
    model.compile()
    dataset = load_datasets('../Data/datasets.json')['BSD']
    dataset.setattr(patch_size=48, strides=48, depth=1)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log')
    env.fit(128, 200, dataset, 1e-5, shuffle=False, restart=False)
    env.output_callbacks = [sub_res, save_image(model.name)]
    env.test(dataset)
    # env.predict(Path('denoise').glob('*.png'))
