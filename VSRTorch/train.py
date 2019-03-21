#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14


import argparse
import logging
from pathlib import Path

from VSRTorch.Models import get_model, list_supported_models
from VSR.DataLoader.Dataset import load_datasets
from VSR.DataLoader.Loader import QuickLoader
from VSR.Framework.Callbacks import lr_decay
from VSR.Util.Config import Config
from VSR.Tools.Run import suppress_opt_by_args, dump

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=list_supported_models(),
                    help="Specify the model name")
parser.add_argument("-p", "--parameter",
                    help="Specify the model parameter file (*.yaml)")
parser.add_argument("--save_dir", default='../Results',
                    help="Working directory")
parser.add_argument("--data_config", default="../Data/datasets.yaml",
                    help="Specify dataset config file")
parser.add_argument("--dataset", default='none',
                    help="specify a dataset alias for training")
parser.add_argument("--export", help="export ONNX (needs support from model).")
parser.add_argument("-c", "--comment", default=None,
                    help="extend a comment string after saving folder")
parser.add_argument('--memory_limit', default=None,
                    help="limit the memory usage. i.e. '4GB', '1024MB'")
parser.add_argument("--train_data_crop", default='random',
                    choices=('random', 'stride'),
                    help="how to crop training data")
parser.add_argument("--val_data_crop", default='random',
                    choices=('random', 'center', 'none'),
                    help="how to crop validating data")
parser.add_argument("--val_num", type=int, default=10,
                    help="Number of validations in training.")
parser.add_argument("--epochs", type=int, default=1,
                    help="Specify total epochs to train")
parser.add_argument("--steps_per_epoch", type=int, default=200,
                    help="specify steps in every epoch training")
parser.add_argument("--thread", type=int, default=8,
                    help="Specify loading threads number")
parser.add_argument("--seed", type=int, default=None, help="set random seed")
parser.add_argument("--pth", help="Specify the pre-trained model path. "
                                  "If not given, will search into `save_dir`.")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--traced_val", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")


def main():
  flags, args = parser.parse_known_args()
  opt = Config()
  for pair in flags._get_kwargs():
    opt.setdefault(*pair)

  data_config_file = Path(flags.data_config)
  if not data_config_file.exists():
    raise RuntimeError("dataset config file doesn't exist!")
  for _ext in ('json', 'yaml', 'yml'):  # for compat
    # apply a 2-stage (or master-slave) configuration, master can be
    # override by slave
    model_config_root = Path('Parameters/root.{}'.format(_ext))
    if opt.p:
      model_config_file = Path(opt.p)
    else:
      model_config_file = Path('Parameters/{}.{}'.format(opt.model, _ext))
    if model_config_root.exists():
      opt.update(Config(str(model_config_root)))
    if model_config_file.exists():
      opt.update(Config(str(model_config_file)))

  model_params = opt.get(opt.model, {})
  opt.update(model_params)
  suppress_opt_by_args(model_params, *args)
  model = get_model(flags.model)(**model_params)
  if flags.cuda:
    model.cuda()
  root = f'{flags.save_dir}/{flags.model}'
  if flags.comment:
    root += '_' + flags.comment
  verbosity = logging.DEBUG if flags.verbose else logging.INFO
  trainer = model.trainer

  datasets = load_datasets(data_config_file)
  dataset = datasets[flags.dataset.upper()]

  train_config = Config(crop=opt.train_data_crop, feature_callbacks=[],
                        label_callbacks=[], convert_to='rgb', **opt)
  if opt.lr_decay:
    train_config.lr_schedule = lr_decay(lr=opt.lr, **opt.lr_decay)
  train_config.random_val = not opt.traced_val
  train_config.cuda = flags.cuda

  if opt.verbose:
    dump(opt)
  with trainer(model, root, verbosity, opt.pth) as t:
    if opt.seed is not None:
      t.set_seed(opt.seed)
    loader = QuickLoader(dataset, 'train', train_config, True, flags.thread)
    vloader = QuickLoader(dataset, 'val', train_config, False,
                          batch=1,
                          crop=opt.val_data_crop,
                          steps_per_epoch=opt.val_num)
    t.fit([loader, vloader], train_config)
    if opt.export:
      t.export(opt.export)


if __name__ == '__main__':
  main()
