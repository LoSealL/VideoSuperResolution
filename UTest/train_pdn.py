from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *
from VSR.DataLoader.Dataset import load_datasets
from Exp.Pdn import PDN


def main():
    model = PDN()
    dataset = load_datasets('../Data/datasets.json')['WATERLOO']
    dataset.setattr(patch_size=40, random=True, max_patches=64 * 200)
    with Environment(model, '../Results/pdn/save', '../Results/pdn/log', feature_index=0, label_index=0) as env:
        env.feature_callbacks = [add_noise(25)]
        env.fit(64, 100, dataset, 1e-4,
                learning_rate_schedule=lr_decay('stair', 1e-3, decay_step=2000, decay_rate=0.94))
        env.output_callbacks += [save_image('../Results/pdn/test')]
        env.test(dataset)


if __name__ == '__main__':
    main()
