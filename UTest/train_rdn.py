from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import save_image
from VSR.DataLoader.Dataset import load_datasets
from VSR.Models.Rdn import ResidualDenseNetwork


def main():
    model = ResidualDenseNetwork(3, rdb_blocks=10, rdb_conv=6, rgb_input=False).compile()
    dataset = load_datasets('../Data/datasets.json')['BSD']
    dataset.setattr(patch_size=96, depth=1, random=True, max_patches=64*1)
    env = Environment(model, f'{model.name}/save', f'{model.name}/log')
    env.fit(64, 1, dataset, restart=True)
    env.output_callbacks = [save_image(f'{model.name}/test')]
    env.test(dataset)


if __name__ == '__main__':
    main()
