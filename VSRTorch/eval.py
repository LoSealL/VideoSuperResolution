#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/4 下午2:42

import argparse
import logging
from pathlib import Path

try:
  import torch
except ImportError:
  print(" [!] Couldn't find pytorch. You should install it before starting.")
  exit(0)

from VSRTorch.Models import get_model, list_supported_models
from VSR.DataLoader.Dataset import Dataset, _glob_absolute_pattern, \
  load_datasets
from VSR.DataLoader.Loader import QuickLoader
from VSR.Framework.Callbacks import save_image, to_gray, to_rgb
from VSR.Util.Config import Config
from VSR.Tools.Run import suppress_opt_by_args, dump

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=list_supported_models(),
                    help="Specify the model name")
parser.add_argument("-p", "--parameter",
                    help="Specify the model parameter file (*.yaml)")
parser.add_argument("-t", "--test", nargs='*',
                    help="Specify test dataset name or data path")
parser.add_argument("--save_dir", default='../Results',
                    help="Working directory")
parser.add_argument("--data_config", default="../Data/datasets.yaml",
                    help="Specify dataset config file")
parser.add_argument("-c", "--comment", default=None,
                    help="Extend a comment string after saving folder")
parser.add_argument("--pth", help="Specify the pre-trained model path. "
                                  "If not given, will search into `save_dir`.")
parser.add_argument("--epoch", type=int, default=None,
                    help="Specify an epoch's model to load,"
                         "will use the latest one if not specified.")
parser.add_argument("--thread", type=int, default=8,
                    help="Specify loading threads number")
parser.add_argument("--output_index", default='-1',
                    help="Specify access index of output array (slicable)")
parser.add_argument("--output_color", default='GRAY', choices=('RGB', 'GRAY'),
                    help="Valid only for 1-channel outputs.")
parser.add_argument("--seed", type=int, default=None, help="set random seed")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--auto_rename", action="store_true")
parser.add_argument("--ensemble", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")


def overwrite_from_env(flags):
  import os
  if os.getenv('VSR_AUTO_RENAME'):
    flags.auto_rename = True
  if os.getenv('VSR_OUTPUT_INDEX'):
    flags.output_index = os.getenv('VSR_OUTPUT_INDEX')


def main():
  flags, args = parser.parse_known_args()
  opt = Config()
  for pair in flags._get_kwargs():
    opt.setdefault(*pair)
  overwrite_from_env(opt)
  data_config_file = Path(flags.data_config)
  if not data_config_file.exists():
    raise RuntimeError("dataset config file doesn't exist!")
  for _ext in ('json', 'yaml', 'yml'):  # for compat
    # apply a 2-stage (or master-slave) configuration, master can be
    # override by slave
    model_config_root = Path('Parameters/root.{}'.format(_ext))
    if opt.parameter:
      model_config_file = Path(opt.parameter)
    else:
      model_config_file = Path('Parameters/{}.{}'.format(opt.model, _ext))
    if model_config_root.exists():
      opt.update(Config(str(model_config_root)))
    if model_config_file.exists():
      opt.update(Config(str(model_config_file)))

  model_params = opt.get(opt.model, {})
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  model = get_model(opt.model)(**model_params)
  if opt.cuda:
    model.cuda()
  root = f'{opt.save_dir}/{opt.model}'
  if opt.comment:
    root += '_' + opt.comment
  verbosity = logging.DEBUG if opt.verbose else logging.INFO
  trainer = model.trainer

  datasets = load_datasets(data_config_file)
  try:
    test_datas = [datasets[t.upper()] for t in opt.test]
    run_benchmark = True
  except KeyError:
    test_datas = []
    for pattern in opt.test:
      test_data = Dataset(test=_glob_absolute_pattern(pattern),
                          test_pair=_glob_absolute_pattern(pattern),
                          mode='pil-image1', modcrop=False,
                          parser='custom_pairs')
      father = Path(pattern)
      while not father.is_dir():
        if father.parent == father:
          break
        father = father.parent
      test_data.name = father.stem
      test_datas.append(test_data)
    run_benchmark = False

  if opt.verbose:
    dump(opt)
  for test_data in test_datas:
    loader_config = Config(convert_to='rgb',
                           feature_callbacks=[], label_callbacks=[],
                           output_callbacks=[], **opt)
    loader_config.batch = 1
    loader_config.subdir = test_data.name
    loader_config.output_callbacks += [
      save_image(root, opt.output_index, opt.auto_rename)]
    if opt.channel == 1:
      loader_config.convert_to = 'gray'
      if opt.output_color == 'RGB':
        loader_config.convert_to = 'yuv'
        loader_config.feature_callbacks = [to_gray()]
        loader_config.label_callbacks = [to_gray()]
        loader_config.output_callbacks.insert(0, to_rgb())

    with trainer(model, root, verbosity, opt.pth) as t:
      if opt.seed is not None:
        t.set_seed(opt.seed)
      loader = QuickLoader(test_data, 'test', loader_config,
                           n_threads=opt.thread)
      loader_config.epoch = opt.epoch
      if run_benchmark:
        t.benchmark(loader, loader_config)
      else:
        t.infer(loader, loader_config)


if __name__ == '__main__':
  main()
