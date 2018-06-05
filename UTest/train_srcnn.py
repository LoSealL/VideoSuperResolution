from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Callbacks import *
from VSR.Models.Srcnn import SRCNN


if __name__ == '__main__':
    model = SRCNN(scale=3)
    dataset = load_datasets('../Data/datasets.json')['91-IMAGE']
    dataset.setattr(patch_size=48, strides=48, depth=1)
    with Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log') as env:
        env.fit(64, 1, dataset, shuffle=False, restart=True)
    # change model with different hyper parameters
    model = SRCNN(scale=4)
    with Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log') as env:
        env.fit(64, 5, dataset, shuffle=False, restart=False)
        env.output_callbacks = [save_image(f'../Results/{model.name}/test')]
        env.test(dataset)
