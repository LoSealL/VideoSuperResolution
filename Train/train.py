"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-7

Help to train all models
"""
import shutil
from pathlib import Path

from common import parse_arguments, parser
from VSR.DataLoader import CenterCrop, Loader, RandomCrop, load_datasets
from VSR.Util import lr_decay

# pylint: disable=line-too-long
g1 = parser.add_argument_group("training options")  # nopep8
g1.add_argument("--dataset", default='none', help="specify a dataset alias for training")  # nopep8
g1.add_argument("--epochs", type=int, default=1, help="specify total epochs to train")  # nopep8
g1.add_argument("--steps", type=int, default=200, help="specify steps of iteration in every training epoch")  # nopep8
g1.add_argument("--val_steps", type=int, default=10, help="steps of iteration of validations during training")  # nopep8
g2 = parser.add_argument_group("advanced options")  # nopep8
g2.add_argument("--traced_val", action="store_true")  # nopep8
g2.add_argument("--distributed", action="store_true")  # nopep8
g2.add_argument("--caching_dataset", action="store_true")  # nopep8


def main():
    args = parse_arguments()
    opt = args.opt
    root = args.root
    model = args.model

    if opt.distributed:
        model.distributed()

    dataset = load_datasets(args.data_config_file, opt.dataset)
    # construct data loader for training
    lt = Loader(dataset.train.hr, dataset.train.lr,
                opt.scale, threads=opt.threads)
    lt.image_augmentation()
    # construct data loader for validating
    lv = None
    if dataset.val is not None:
        lv = Loader(dataset.val.hr, dataset.val.lr,
                    opt.scale, threads=opt.threads)
    lt.cropper(RandomCrop(opt.scale))
    if opt.traced_val and lv is not None:
        lv.cropper(CenterCrop(opt.scale))
    elif lv is not None:
        lv.cropper(RandomCrop(opt.scale))
    if opt.channel == 1:
        # convert data color space to grayscale
        lt.set_color_space('hr', 'L')
        lt.set_color_space('lr', 'L')
        if lv is not None:
            lv.set_color_space('hr', 'L')
            lv.set_color_space('lr', 'L')
    # enter model executor environment
    with model.get_executor(root) as t:
        if hasattr(t, '_logd') and isinstance(t._logd, Path):
            shutil.copy(args.model_config_file, t._logd)
        config = t.query_config(opt)
        if opt.lr_decay:
            config.lr_schedule = lr_decay(lr=opt.lr, **opt.lr_decay)
        config.caching = opt.caching_dataset and opt.memory_limit is None
        t.fit([lt, lv], config)
        if opt.export:
            t.export(opt.export)


if __name__ == '__main__':
    main()
