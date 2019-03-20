#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import argparse
import logging
from pathlib import Path

from VSRTorch.Models import get_model, list_supported_models
from VSR.DataLoader.Dataset import Dataset, _glob_absolute_pattern, \
  load_datasets
from VSR.DataLoader.Loader import QuickLoader
from VSR.Framework.Callbacks import save_image
from VSR.Util.Config import Config
from VSR.Tools.Run import suppress_opt_by_args, dump

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=list_supported_models(),
                    help="Specify the model name")
parser.add_argument("-p", "--parameter",
                    help="Specify the model parameter file (*.yaml)")
parser.add_argument("-t", "--test",
                    help="Specify test dataset name or data path")
parser.add_argument("--save_dir", default='../Results',
                    help="Working directory")
parser.add_argument("--data_config", default="../Data/datasets.yaml",
                    help="Specify dataset config file")
parser.add_argument("-c", "--comment", default=None,
                    help="extend a comment string after saving folder")
parser.add_argument("--pth", help="Specify the pre-trained model path. "
                                  "If not given, will search into `save_dir`.")
parser.add_argument("--epoch", type=int, default=None,
                    help="Specify an epoch's model to load,"
                         "will use the latest one if not specified.")
parser.add_argument("--thread", type=int, default=8,
                    help="Specify loading threads number")
parser.add_argument("--output_index", type=int, default=-1,
                    help="specify access index of output array")
parser.add_argument("--seed", type=int, default=None, help="set random seed")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--auto_rename", action="store_true")
parser.add_argument("--ensemble", action="store_true")
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
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  model = get_model(flags.model)(**model_params)
  if flags.cuda:
    model.cuda()
  root = f'{flags.save_dir}/{flags.model}'
  if flags.comment:
    root += '_' + flags.comment
  verbosity = logging.DEBUG if flags.verbose else logging.INFO
  trainer = model.trainer

  datasets = load_datasets(data_config_file)
  try:
    test_data = datasets[flags.test.upper()]
    run_benchmark = True
  except KeyError:
    test_data = Dataset(test=_glob_absolute_pattern(flags.test),
                        mode='pil-image1', modcrop=False)
    father = Path(flags.test)
    while not father.is_dir():
      if father.parent == father:
        break
      father = father.parent
    test_data.name = father.stem
    run_benchmark = False

  loader_config = Config(convert_to='rgb',
                         feature_callbacks=[], label_callbacks=[],
                         output_callbacks=[], **opt)
  loader_config.batch = 1
  loader_config.subdir = test_data.name
  loader_config.output_callbacks += [
    save_image(root, flags.output_index, flags.auto_rename)]

  if opt.verbose:
    dump(opt)
  with trainer(model, root, verbosity, flags.pth) as t:
    if flags.seed is not None:
      t.set_seed(flags.seed)
    loader = QuickLoader(test_data, 'test', loader_config,
                         n_threads=flags.thread)
    loader_config.epoch = flags.epoch
    if run_benchmark:
      t.benchmark(loader, loader_config)
    else:
      t.infer(loader, loader_config)


if __name__ == '__main__':
  main()
