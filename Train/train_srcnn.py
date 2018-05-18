from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Callbacks import *
from Models.Srcnn import SRCNN

if __name__ == '__main__':
    model = SRCNN(scale=3)
    model.compile()
    dataset = load_datasets('../Data/datasets.json')['BSD']
    dataset.setattr(patch_size=48, strides=48, depth=1)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log')
    env.fit(64, 10, dataset, shuffle=False, restart=False)
    env.output_callbacks = [save_image(model.name)]
    env.test(dataset)
