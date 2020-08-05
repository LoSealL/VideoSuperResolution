#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 8 - 5

import argparse
from pathlib import Path

from VSR.Backend import BACKEND
from VSR.Model import get_model, list_supported_models
from VSR.Util import Config, compat_param, suppress_opt_by_args

CWD = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=f'VSR ({BACKEND}) Tool v1.0')
g0 = parser.add_argument_group("basic options")
g0.add_argument("model", choices=list_supported_models(), help="specify the model name")
g0.add_argument("-p", "--parameter", help="specify the model parameter file (*.yaml)")
g0.add_argument("--save_dir", default=f'{CWD}/Results', help="working directory")
g0.add_argument("--data_config", default=f"{CWD}/Data/datasets.yaml", help="specify dataset config file")
g0.add_argument("-c", "--comment", default=None, help="extend a comment string after saving folder")
g0.add_argument("--pretrain", help="specify the pre-trained model checkpoint or will search into `save_dir` if not specified")
g0.add_argument("--print", action="store_true", help="print model parameters")
g2 = parser.add_argument_group("device options")
g2.add_argument("--cuda", action="store_true", help="using cuda gpu")
g2.add_argument("--threads", type=int, default=8, help="specify loading threads number")
g2.add_argument('--memory_limit', default=None, help="limit the CPU memory usage. i.e. '4GB', '1024MB'")


def parse_arguments():
  flags, args = parser.parse_known_args()
  opt = Config()
  # overwrite flag values into opt object
  for pair in flags._get_kwargs():
    opt.setdefault(*pair)
  data_config_file = Path(flags.data_config)
  if not data_config_file.exists():
    raise FileNotFoundError("dataset config file doesn't exist!")
  for _ext in ('json', 'yaml', 'yml'):  # for compat
    if opt.parameter:
      model_config_file = Path(opt.parameter)
    else:
      model_config_file = Path(f'{CWD}/Train/par/{BACKEND}/{opt.model}.{_ext}')
    if model_config_file.exists():
      opt.update(compat_param(Config(str(model_config_file))))
      break
  # get model parameters from pre-defined YAML file
  model_params = opt.get(opt.model, {})
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  # construct model
  model = get_model(opt.model)(**model_params)
  if opt.print:
    print(model_params)
  if opt.cuda:
    model.cuda()
  if opt.pretrain:
    model.load(opt.pretrain)
  root = f'{opt.save_dir}/{opt.model}'
  if opt.comment:
    root += '_' + opt.comment
  root = Path(root)

  return Config(
      opt=opt,
      root=root,
      model=model,
      data_config_file=data_config_file,
      model_config_file=model_config_file
  )
