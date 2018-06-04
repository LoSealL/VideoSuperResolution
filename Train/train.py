"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 25th 2018

Train models
"""

import argparse, json
import tensorflow as tf

from model_alias import get_model
from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Envrionment import Environment
from VSR.Framework.Callbacks import *

def main(*args, **kwargs):
    args = argparse.ArgumentParser()
    args.add_argument('name', type=str, help='the model name can be found in model_alias.py')
    args.add_argument('--scale', type=int, default=3, help='scale factor')
    args.add_argument('--dataconfig', type=str, default='../Data/datasets.json', help='the path to dataset config json file')
    args.add_argument('--dataset', type=str, default='BSD', help='specified dataset name(as described in config file')
    args.add_argument('--batch', type=int, default=64, help='training batch size')
    args.add_argument('--epochs', type=int, default=200, help='training epochs')
    args.add_argument('--patch_size', type=int, default=48, help='patch size of cropped training and validating sub-images')
    args.add_argument('--stride', type=int, default=48, help='crop stride if random_patches is set 0')
    args.add_argument('--depth', type=int, default=1, help='image depth used for video sources')
    args.add_argument('--shuffle', type=bool, default=False, help='shuffle files in dataset, this operation will open all files and may be slow')
    args.add_argument('--random_patches', type=int, default=0, help='if set more than 0, use random crop to generate `random_patches` sub-image batches')
    args.add_argument('--retrain', type=bool, default=False, help='retrain the model from scratch')
    args.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    args.add_argument('--add_noise', type=float, default=None, help='if not None, add noise with given stddev to input features')
    args.add_argument('--test', type=bool, default=True, help='test model and save tested images')
    args.add_argument('--savedir', type=str, default='../Results', help='directory to save model checkpoints')
    args.add_argument('--export_pb', type=str, default=None, help='if not None, specify the path that export trained model into pb format')

    args = args.parse_args()
    model_args = json.load(open(f'parameters/{args.name}.json', mode='r'))

    model = get_model(args.name)(scale=args.scale, **model_args)
    model.compile()
    dataset = load_datasets(args.dataconfig)[args.dataset]
    dataset.setattr(patch_size=args.patch_size, strides=args.stride, depth=args.depth)
    if args.random_patches:
        dataset.setattr(random=True, max_patches=args.batch * args.random_patches)
    env = Environment(model, f'{args.savedir}/{model.name}/save', f'{args.savedir}/{model.name}/log',
                      feature_index=model.feature_index, label_index=model.label_index)
    if args.add_noise:
        env.feature_callbacks = [add_noise(args.add_noise)]

    env.fit(args.batch, args.epochs, dataset, restart=args.retrain,
            learning_rate=args.lr,
            learning_rate_schedule=lambda lr, epochs, steps, **kwargs:
                tf.train.exponential_decay(args.lr, steps, 100, 0.96).eval(session=model.sess))
    if args.test:
        # use callback to generate colored images from grayscale ones
        # all models inputs is gray image however
        if args.add_noise:
            env.feature_callbacks = [add_noise(args.add_noise)]
        env.feature_callbacks += [to_gray()]
        env.label_callbacks = [to_gray()]
        env.output_callbacks += [to_rgb()]
        env.output_callbacks += [save_image(f'../Results/{model.name}/test')]
        env.test(dataset, convert_to_gray=False)  # load image with 3 channels
    if args.export_pb:
        model = get_model(args.name)(scale=args.scale, rgb_input=True)
        model.compile()
        env = Environment(model, f'../Results/{model.name}/save', f'../Results/{model.name}/log')
        env.export(args.export_pb)


if __name__ == '__main__':
    tf.app.run(main)
