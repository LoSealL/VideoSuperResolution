#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 10

import argparse
from pathlib import Path

from VSR.Backend import BACKEND
from VSR.DataLoader import Dataset, Loader, load_datasets
from VSR.Model import get_model, list_supported_models
from VSR.Util import (
  Config, compat_param, save_inference_images, suppress_opt_by_args
)

parser = argparse.ArgumentParser(description=f'VSR ({BACKEND}) Testing Tool v1.0')
g0 = parser.add_argument_group("basic options")
g0.add_argument("model", choices=list_supported_models(), help="specify the model name")
g0.add_argument("-p", "--parameter", help="specify the model parameter file (*.yaml)")
g0.add_argument("-t", "--test", nargs='*', help="specify test dataset name or data path")
g0.add_argument("--save_dir", default='../Results', help="working directory")
g0.add_argument("--data_config", default="../Data/datasets.yaml", help="specify dataset config file")
g1 = parser.add_argument_group("evaluating options")
g1.add_argument("--pretrain", help="specify the pre-trained model checkpoint or will search into `save_dir` if not specified")
g1.add_argument("--ensemble", action="store_true")
g1.add_argument("--video", action="store_true", help="notify load test data as video stream")
g2 = parser.add_argument_group("device options")
g2.add_argument("--cuda", action="store_true", help="using cuda gpu")
g2.add_argument("--threads", type=int, default=8, help="specify loading threads number")
g3 = parser.add_argument_group("advanced options")
g3.add_argument("--output_index", default='-1', help="specify access index of output array (slicable)")
g3.add_argument("--export", help="export ONNX (torch backend) or protobuf (tf backend) (needs support from model)")
g3.add_argument("--overwrite", action="store_true", help="overwrite the existing predicted output files")
g3.add_argument("-c", "--comment", default=None, help="extend a comment string after saving folder")


def str2boolean(s):
  assert isinstance(s, str)
  if s.lower() in ('true', 'yes', '1'):
    return True
  else:
    return False


def overwrite_from_env(flags):
  import os
  auto_rename = os.getenv('VSR_AUTO_RENAME')
  output_index = os.getenv('VSR_OUTPUT_INDEX')

  if auto_rename and auto_rename != '':
    flags.auto_rename = str2boolean(auto_rename)
  if output_index and output_index != '':
    flags.output_index = output_index


def main():
  flags, args = parser.parse_known_args()
  opt = Config(depth=-1)
  for pair in flags._get_kwargs():
    opt.setdefault(*pair)
  overwrite_from_env(opt)
  data_config_file = Path(flags.data_config)
  if not data_config_file.exists():
    raise FileNotFoundError("dataset config file doesn't exist!")
  for _ext in ('json', 'yaml', 'yml'):  # for compat
    if opt.parameter:
      model_config_file = Path(opt.parameter)
    else:
      model_config_file = Path(f'par/{BACKEND}/{opt.model}.{_ext}')
    if model_config_file.exists():
      opt.update(compat_param(Config(str(model_config_file))))
  # get model parameters from pre-defined YAML file
  model_params = opt.get(opt.model, {})
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  # construct model
  model = get_model(opt.model)(**model_params)
  if opt.cuda:
    model.cuda()
  if opt.pretrain:
    model.load(opt.pretrain)
  root = f'{opt.save_dir}/{opt.model}'
  if opt.comment:
    root += '_' + opt.comment
  root = Path(root)

  datasets = load_datasets(data_config_file)
  try:
    test_datas = [datasets[t.upper()] for t in opt.test] if opt.test else []
  except KeyError:
    test_datas = [Config(test=Config(lr=Dataset(*opt.test)), name='infer')]
    if opt.video:
      test_datas[0].test.lr.use_like_video_()
  # enter model executor environment
  with model.get_executor(root) as t:
    for data in test_datas:
      run_benchmark = False if data.test.hr is None else True
      if run_benchmark:
        ld = Loader(data.test.hr, data.test.lr, opt.scale,
                    threads=opt.threads)
      else:
        ld = Loader(data.test.hr, data.test.lr, threads=opt.threads)
      if opt.channel == 1:
        # convert data color space to grayscale
        ld.set_color_space('hr', 'L')
        ld.set_color_space('lr', 'L')
      config = t.query_config(opt)
      config.inference_results_hooks = [save_inference_images(root / data.name, opt.output_index, not opt.overwrite)]
      config.batch_shape = [1, opt.depth, -1, -1, -1]
      config.traced_val = True
      if run_benchmark:
        t.benchmark(ld, config)
      else:
        t.infer(ld, config)
    if opt.export:
      t.export(opt.export)


if __name__ == '__main__':
  main()
