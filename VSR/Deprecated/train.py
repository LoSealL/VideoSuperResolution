"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 25th 2018

Train models, the examples can be
found in `run.*` scripts
"""

import argparse, json

from VSR.DataLoader.Dataset import load_datasets
from VSR.Framework.Environment import Environment
from VSR.Framework.Callbacks import *

try:
    from .model_alias import get_model, list_supported_models
except ImportError:
    from model_alias import get_model, list_supported_models
try:
    from Train.custom_api import *
except ImportError:
    from custom_api import *


def main(*args, **kwargs):
    args = argparse.ArgumentParser()
    args.add_argument('name', type=str, choices=list_supported_models(), help='the model name can be found in model_alias.py')
    # basic options
    args.add_argument('--scale', type=int, default=4, help='scale factor, default 4')
    args.add_argument('--channel', type=int, default=3, help='image channels, default 3')
    args.add_argument('--batch', type=int, default=16, help='training batch size, default 16')
    args.add_argument('--epochs', type=int, default=200, help='training epochs, default 200')
    args.add_argument('--steps_per_epoch', type=int, default=200, help='training steps each epoch, default 200')
    args.add_argument('--retrain', type=bool, default=False, help='retrain the model from scratch')
    # dataset options
    args.add_argument('--dataconfig', type=str, default='../Data/datasets.json', help='the path to dataset config json file')
    args.add_argument('--dataset', type=str, default='91-IMAGE', help='specified dataset name(as described in config file, default 91-image')
    args.add_argument('--patch_size', type=int, default=48, help='patch size of cropped training and validating sub-images, default 48')
    args.add_argument('--depth', type=int, default=1, help='image depth used for video sources, default 1')
    args.add_argument('--parallel', type=int, default=1, help='number of cores used to load training sets in parallel')
    args.add_argument('--memory', type=str, default=None, help='limit the memory usage. i.e. 4GB, 100MB')
    # learning options
    args.add_argument('--lr', type=float, default=1e-4, help='initial learning rate, default 1e-4')
    args.add_argument('--lr_decay', type=float, default=1, help='learning rate decay, default 1')
    args.add_argument('--lr_decay_step', type=int, default=1000, help='learning rate decay step')
    # output options
    args.add_argument('--test', type=str, default=None, help='specify a dataset used to test, or use --dataset values if None')
    args.add_argument('--predict', type=str, default=None, help='evaluate model on given files')
    args.add_argument('--savedir', type=str, default='../Results', help='directory to save model checkpoints, default ../Results')
    args.add_argument('--output_color', type=str, default='RGB', choices=('RGB', 'L', 'GRAY', 'Y'), help='output color mode, default RGB')
    args.add_argument('--output_index', type=int, default=-1, help='access index of model outputs to save, default -1')
    args.add_argument('--export_pb', type=str, default=None, help='if not None, specify the path that export trained model into pb format')
    args.add_argument('--comment', type=str, default=None, help='add a suffix to output dir to distinguish each experiments')
    # callbacks
    args.add_argument('--add_noise', type=float, default=None, help='if not None, add noise with given stddev to input features')
    args.add_argument('--add_random_noise', type=list, default=None, help='if not None, add random noise with given stddev bound [low, high, step=1]')
    args.add_argument('--custom_feature_cb', type=str, default=None, help='customized callbacks, defined in `custom_api.py`')

    args = args.parse_args()
    if Path(f'parameters/{args.name}.json').exists():
        model_args = json.load(open(f'parameters/{args.name}.json', mode='r'))
    else:
        print(f'[warning] model parameter file not found, use default parameters')
        model_args = dict()
    model = get_model(args.name)(scale=args.scale, channel=args.channel, **model_args)

    dataset = load_datasets(args.dataconfig)[args.dataset.upper()]
    dataset.setattr(patch_size=args.patch_size, depth=args.depth)
    save_root = f'{args.savedir}/{model.name}_sc{args.scale}_c{args.channel}'
    if args.comment:
        save_root += '_' + args.comment
    with Environment(model, f'{save_root}/save', f'{save_root}/log',
                     feature_index=model.feature_index, label_index=model.label_index) as env:
        if args.add_noise:
            env.feature_callbacks = [add_noise(args.add_noise)]
        if args.add_random_noise:
            env.feature_callbacks = [add_random_noise(*args.add_random_noise)]
        if args.custom_feature_cb:
            func = args.custom_feature_cb.split(' ')
            for f_name in func:
                env.feature_callbacks += [globals()[f_name]]
        fit_fn = partial(env.fit, args.batch, args.epochs, args.steps_per_epoch, dataset,
                         augmentation=True,
                         restart=args.retrain,
                         learning_rate=args.lr,
                         learning_rate_schedule=lr_decay('stair',
                                                         args.lr,
                                                         decay_step=args.lr_decay_step,
                                                         decay_rate=args.lr_decay),
                         parallel=args.parallel,
                         memory_usage=args.memory)
        if model.channel > 1:
            fit_fn(convert_to='RGB')
            test_format = 'RGB'
        else:
            fit_fn(convery_to='GRAY')
            # use callback to generate colored images from grayscale ones
            # all models inputs is gray image however
            test_format = 'YUV'
            env.feature_callbacks += [to_gray()]
            env.label_callbacks = [to_gray()]
            if args.output_color == 'RGB':
                env.output_callbacks += [to_rgb()]
        if args.test:
            test_set = load_datasets(args.dataconfig)[args.test.upper()]
            test_set.setattr(patch_size=args.patch_size, depth=args.depth)
        else:
            test_set = dataset
        env.output_callbacks += [save_image(f'{save_root}/test', args.output_index)]
        env.test(test_set, convert_to=test_format)  # load image with 3 channels
        if args.predict:
            pth = Path(args.predict)
            if not pth.exists():
                raise ValueError('[Error] File path does not exist')
            if pth.is_dir():
                images = list(pth.glob('*'))
                if not images:
                    images = pth.iterdir()
            elif pth.is_file():
                images = pth
            env.fi, fi_old = 0, env.fi  # upscale directly
            env.output_callbacks[-1] = save_image(f'{save_root}/output', args.output_index)
            env.predict(images, convert_to=test_format, depth=args.depth)
            env.fi = fi_old
        elif test_set.pred:
            env.fi, fi_old = 0, env.fi  # upscale directly
            env.output_callbacks[-1] = save_image(f'{save_root}/output', args.output_index)
            env.predict(test_set.pred, convert_to=test_format, depth=args.depth)
            env.fi = fi_old
        if args.export_pb:
            env.export(args.export_pb)


if __name__ == '__main__':
    main()
