from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from Models.Srcnn import SRCNN

if __name__ == '__main__':
    model = SRCNN(scale=3)
    model.compile()
    datasets = load_datasets('../Data/datasets.json')
    env = Environment(model, 'srcnn/save', 'srcnn/log')
    env.fit(50, 100, datasets['91-IMAGE'], patch_size=48, strides=48, depth=1)
