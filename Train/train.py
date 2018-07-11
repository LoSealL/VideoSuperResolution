"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 25th 2018

Train models
"""

import argparse, json

from model_alias import get_model, list_supported_models
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *


def main(*args, **kwargs):
    args = argparse.ArgumentParser()
    args.add_argument('name', type=str, choices=list_supported_models(),
                      help='the model name can be found in model_alias.py')
    args.add_argument('--scale', type=int, default=3, help='scale factor')
    args.add_argument('--dataconfig', type=str, default='../Data/datasets.json',
                      help='the path to dataset config json file')
    args.add_argument('--dataset', type=str, default='91-IMAGE',
                      help='specified dataset name(as described in config file')
    args.add_argument('--batch', type=int, default=64, help='training batch size')
    args.add_argument('--epochs', type=int, default=200, help='training epochs')
    args.add_argument('--patch_size', type=int, default=48,
                      help='patch size of cropped training and validating sub-images')
    args.add_argument('--strides', type=int, default=48, help='crop stride if random_patches is set 0')
    args.add_argument('--depth', type=int, default=1, help='image1 depth used for video sources')
    args.add_argument('--random_patches', type=int, default=0,
                      help='if set more than 0, use random crop to generate `random_patches` sub-image1 batches')
    args.add_argument('--retrain', type=int, default=0, help='retrain the model from scratch')
    args.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    args.add_argument('--lr_decay', type=float, default=1, help='learning rate decay')
    args.add_argument('--lr_decay_step', type=int, default=1000, help='learning rate decay step')
    args.add_argument('--add_noise', type=float, default=None,
                      help='if not None, add noise with given stddev to input features')
    args.add_argument('--add_random_noise', type=list, default=None,
                      help='if not None, add random noise with given stddev bound [low, high, step=1]')
    args.add_argument('--test', type=bool, default=True, help='test model and save tested images')
    args.add_argument('--savedir', type=str, default='../Results', help='directory to save model checkpoints')
    args.add_argument('--output_color', type=str, default='RGB', choices=('RGB', 'L', 'GRAY', 'Y'),
                      help='output color mode')
    args.add_argument('--export_pb', type=str, default=None,
                      help='if not None, specify the path that export trained model into pb format')

    args = args.parse_args()
    model_args = json.load(open(f'parameters/{args.name}.json', mode='r'))

    model = get_model(args.name)(scale=args.scale, **model_args)
    dataset = load_datasets(args.dataconfig)[args.dataset.upper()]
    dataset.setattr(patch_size=args.patch_size, strides=args.strides, depth=args.depth)
    if args.random_patches:
        dataset.setattr(random=True, max_patches=args.batch * args.random_patches)
    save_root = f'{args.savedir}/{model.name}/sc{args.scale}'
    with Environment(model, f'{save_root}/save', f'{save_root}/log',
                     feature_index=model.feature_index, label_index=model.label_index) as env:
        if args.add_noise:
            env.feature_callbacks = [add_noise(args.add_noise)]
        if args.add_random_noise:
            env.feature_callbacks = [add_random_noise(*args.add_random_noise)]

        env.fit(args.batch, args.epochs, dataset, restart=args.retrain,
                learning_rate=args.lr,
                learning_rate_schedule=lr_decay('stair', args.lr,
                                                decay_step=args.lr_decay_step,
                                                decay_rate=args.lr_decay))
        if args.test:
            # use callback to generate colored images from grayscale ones
            # all models inputs is gray image1 however
            env.feature_callbacks += [to_gray()]
            env.label_callbacks = [to_gray()]
            if args.output_color == 'RGB':
                env.output_callbacks += [to_rgb()]
            env.output_callbacks += [save_image(f'{save_root}/test')]
            env.test(dataset, convert_to='YUV')  # load image1 with 3 channels
    if args.export_pb:
        model = get_model(args.name)(scale=args.scale, rgb_input=True)
        with Environment(model, f'{save_root}/save', f'{save_root}/log',
                         feature_index=model.feature_index, label_index=model.label_index) as env:
            env.export(args.export_pb)


if __name__ == '__main__':
    main()
