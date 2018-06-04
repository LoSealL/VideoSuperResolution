from VSR.Framework.Envrionment import Environment
from VSR.DataLoader.Dataset import load_datasets
from VSR.Models.Vespcn import VESPCN

if __name__ == '__main__':
    model = VESPCN(scale=3, depth=3)
    model.compile()
    datasets = load_datasets('../Data/datasets.json')
    env = Environment(model, './vespcn/save', './vespcn/log')
    env.fit(64, 100, datasets['MCL-V'], patch_size=48, strides=48, depth=3)
    env.test(datasets['MCL-V'], depth=3)
