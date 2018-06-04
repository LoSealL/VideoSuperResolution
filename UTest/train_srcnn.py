from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Callbacks import *
from VSR.Models.Srcnn import SRCNN


if __name__ == '__main__':
    model = SRCNN(scale=3).compile()
    dataset = load_datasets('../Data/datasets.json')['91-IMAGE']
    dataset.setattr(patch_size=48, strides=48, depth=1)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log')
    env.fit(64, 1, dataset, shuffle=False, restart=True)
    model.reset()
    model = SRCNN(scale=3).compile()
    env.reset(model)
    env.fit(64, 1, dataset, shuffle=False, restart=True)
    env.output_callbacks = [save_image(f'{model.name}/test')]
    env.test(dataset)
