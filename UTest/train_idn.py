from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import save_image
from VSR.DataLoader.Dataset import load_datasets
from VSR.Models.Idn import InformationDistillationNetwork


def main():
    model = InformationDistillationNetwork(3, rgb_input=False).compile()
    dataset = load_datasets('../Data/datasets.json')['91-IMAGE']
    dataset.setattr(patch_size=48, depth=1, random=True, max_patches=64*300)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log')
    env.fit(64, 100, dataset, restart=False, learning_rate=1e-5)
    env.output_callbacks = [save_image(f'{model.name}/test')]
    env.test(dataset)


if __name__ == '__main__':
    main()
