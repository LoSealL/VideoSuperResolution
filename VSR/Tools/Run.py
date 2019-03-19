"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Improved train/benchmark/infer script
"""

from functools import partial
from pathlib import Path

import tensorflow as tf

from ..DataLoader.Dataset import Dataset, _glob_absolute_pattern, load_datasets
from ..DataLoader.Loader import QuickLoader
from ..Framework.Callbacks import lr_decay, save_image, to_gray, to_rgb
from ..Models import get_model, list_supported_models
from ..Util.Config import Config

tf.flags.DEFINE_enum('model', None, list_supported_models(),
                     help="specify a model to use")
tf.flags.DEFINE_enum('output_color', 'RGB', ('RGB', 'L', 'GRAY', 'Y'),
                     help="specify output color format")
tf.flags.DEFINE_enum('train_data_crop', 'random', ('random', 'stride'),
                     help="how to crop training data")
tf.flags.DEFINE_enum('val_data_crop', 'random', ('random', 'center', 'none'),
                     help="how to crop validating data")
tf.flags.DEFINE_integer('epochs', 50, lower_bound=1, help="training epochs")
tf.flags.DEFINE_integer('steps_per_epoch', 200, lower_bound=1,
                        help="specify steps in every epoch training")
tf.flags.DEFINE_integer('val_num', 10, lower_bound=1,
                        help="Number of validations in training.")
tf.flags.DEFINE_integer('threads', 1, lower_bound=1,
                        help="number of threads to use while reading data")
tf.flags.DEFINE_integer('output_index', -1,
                        help="specify access index of output array")
tf.flags.DEFINE_integer('seed', None, help="set random seed")
tf.flags.DEFINE_string('c', None, help="specify a configure file")
tf.flags.DEFINE_string('p', None,
                       help="specify a parameter file, "
                            "otherwise will use the file in ./parameters")
tf.flags.DEFINE_string('test', None, help="specify another dataset for testing")
tf.flags.DEFINE_string('infer', None,
                       help="specify a file, a path or a dataset for inferring")
tf.flags.DEFINE_string('save_dir', '../Results',
                       "specify a folder to save checkpoint and output images")
tf.flags.DEFINE_string('data_config', '../Data/datasets.yaml',
                       help="path to data config file")
tf.flags.DEFINE_string('dataset', 'none',
                       help="specify a dataset alias for training")
tf.flags.DEFINE_string('memory_limit', None,
                       help="limit the memory usage. i.e. '4GB', '1024MB'")
tf.flags.DEFINE_string('comment', None,
                       help="append a suffix string to save dir")
tf.flags.DEFINE_multi_string('add_custom_callbacks', None,
                             help="add callbacks to feature data. "
                                  "Callbacks are defined in custom_api.py.")
tf.flags.DEFINE_alias('f', "add_custom_callbacks")
tf.flags.DEFINE_multi_string('f2', None, "add callbacks to label data.")
tf.flags.DEFINE_multi_string('f3', None, "add callbacks to output data.")
tf.flags.DEFINE_bool('export', False, help="whether to export tf model")
tf.flags.DEFINE_bool('freeze', False,
                     help="whether to export freeze model, "
                          "ignored if export is False")
tf.flags.DEFINE_bool('auto_rename', True,
                     "Add a suffix and auto rename the conflict output file.")
tf.flags.DEFINE_bool('random_val', True,
                     "Randomly select validation patches. "
                     "Set to false if you want to trace the same patch"
                     " (i.e. GAN).")
tf.flags.DEFINE_bool('ensemble', False,
                     "Enable self-ensemble at inferring. (ONLY INFER)")
tf.flags.DEFINE_bool('v', False, help="show verbose info")


def cross_type_assign(value, dtype):
  """Convert `value` to `dtype`.
    Usually this can be done by simply `dtype(value)`, however, this ain't
    correct for str -> bool conversion.
  """

  if dtype is bool and isinstance(value, str):
    if value.lower() == 'false':
      return False
    elif value.lower() == 'true':
      return True
    else:
      tf.logging.warning(
        "suspect wrong typo {}, do you mean true/false?".format(value))
      return True
  return dtype(value)


def check_args(opt):
  if opt.c:
    opt.update(Config(opt.c))
  _required = ('model',)
  for r in _required:
    if r not in opt or not opt.get(r):
      raise ValueError('--' + r + ' must be set')


def fetch_datasets(data_config_file, opt):
  all_datasets = load_datasets(data_config_file)
  dataset = all_datasets[opt.dataset.upper()]
  if opt.test:
    test_data = all_datasets[opt.test.upper()]
  else:
    test_data = dataset
  if opt.infer:
    images = _glob_absolute_pattern(opt.infer)
    infer_data = Dataset(infer=images, mode='pil-image1', modcrop=False)
  else:
    infer_data = test_data
  return dataset, test_data, infer_data


def init_loader_config(opt):
  train_config = Config(crop=opt.train_data_crop, feature_callbacks=[],
                        label_callbacks=[], **opt)
  benchmark_config = Config(crop=None, feature_callbacks=[],
                            label_callbacks=[], output_callbacks=[], **opt)
  infer_config = Config(feature_callbacks=[], label_callbacks=[],
                        output_callbacks=[], **opt)
  benchmark_config.batch = opt.test_batch or 1
  benchmark_config.steps_per_epoch = -1
  if opt.channel == 1:
    train_config.convert_to = 'gray'
    benchmark_config.convert_to = 'gray'
    if opt.output_color == 'RGB':
      benchmark_config.convert_to = 'yuv'
      benchmark_config.feature_callbacks = [to_gray()]
      benchmark_config.label_callbacks = [to_gray()]
      benchmark_config.output_callbacks = [to_rgb()]
    infer_config.update(benchmark_config)
  else:
    train_config.convert_to = 'rgb'
    benchmark_config.convert_to = 'rgb'
    infer_config.update(benchmark_config)

  def parse_callbacks(fn):
    if '#' in fn:
      fn, args = fn.split('#')
      fn = globals()[fn]
      args = [float(a) for a in args.split(',')]

      def new_fn(x, **kwargs):
        return fn(x, *args)
    else:
      new_fn = globals()[fn]

    return new_fn

  if opt.add_custom_callbacks is not None:
    for func_name in opt.add_custom_callbacks:
      functor = parse_callbacks(func_name)
      train_config.feature_callbacks += [functor]
      benchmark_config.feature_callbacks += [functor]
      infer_config.feature_callbacks += [functor]
  if opt.f2 is not None:
    for func_name in opt.f2:
      functor = parse_callbacks(func_name)
      train_config.label_callbacks += [functor]
      benchmark_config.label_callbacks += [functor]
      infer_config.label_callbacks += [functor]
  if opt.f3 is not None:
    for func_name in opt.f3:
      functor = parse_callbacks(func_name)
      benchmark_config.output_callbacks += [functor]
      infer_config.output_callbacks += [functor]
  # Add image saver at last.
  benchmark_config.output_callbacks += [
    save_image(opt.root, opt.output_index, opt.auto_rename)]
  infer_config.output_callbacks += [
    save_image(opt.root, opt.output_index, opt.auto_rename)]
  if opt.lr_decay:
    train_config.lr_schedule = lr_decay(lr=opt.lr, **opt.lr_decay)
  # modcrop: A boolean to specify whether to crop the edge of images to be
  #   divisible by `scale`. It's useful when to provide batches with original
  #   shapes.
  infer_config.modcrop = False
  infer_config.ensemble = opt.ensemble  # self-ensemble
  train_config.random_val = opt.random_val
  return train_config, benchmark_config, infer_config


def suppress_opt_by_args(opt, *args):
  """Use cmdline arguments to overwrite parameters declared in yaml file.
    Account for safety, writing section not declared in yaml is not allowed.
  """

  def parse_args(argstr: str, prev_argstr: str):
    if prev_argstr:
      k, v = prev_argstr, argstr
    elif argstr.startswith('--'):
      if '=' in argstr:
        k, v = argstr[2:].split('=')
      else:
        k = argstr[2:]
        v = None
    elif argstr.startswith('-'):
      if '=' in argstr:
        k, v = argstr[1:].split('=')
      else:
        k = argstr[1:]
        v = None
    else:
      raise KeyError("Unknown parameter: {}".format(argstr))
    return k, v

  prev_arg = None
  for arg in args:
    key, value = parse_args(arg, prev_arg)
    prev_arg = None  # clear after use
    if key and value:
      # dict support
      keys = key.split('.')
      if keys[0] not in opt:
        raise KeyError("Parameter {} doesn't exist in model!".format(key))
      old_v = opt.get(keys[0])
      if isinstance(old_v, (list, tuple)):
        # list, tuple support
        if not value.startswith('[') and not value.startswith('('):
          raise TypeError("Invalid list syntax: {}".format(value))
        if not value.endswith(']') and not value.endswith(')'):
          raise TypeError("Invalid list syntax: {}".format(value))
        values = value[1:-1].split(',')
        new_v = [cross_type_assign(nv, type(ov)) for ov, nv in
                 zip(old_v, values)]
        opt[keys[0]] = new_v
      elif isinstance(old_v, dict):
        # dict support
        try:
          for k in keys[1:-1]:
            old_v = old_v[k]
          ref_v = old_v
          old_v = old_v[keys[-1]]
        except KeyError:
          raise KeyError("Parameter {} doesn't exist in model!".format(key))
        if isinstance(old_v, (list, tuple)):
          raise NotImplementedError("Don't support nested list type.")
        new_v = cross_type_assign(value, type(old_v))
        ref_v[keys[-1]] = new_v
      else:
        new_v = cross_type_assign(value, type(old_v))
        opt[keys[0]] = new_v
    elif key:
      prev_arg = key

  if prev_arg:
    raise KeyError("Parameter missing value: {}".format(prev_arg))


def dump(config):
  print('=============================')
  for k, v in config.items():
    print('| [{}]: {}'.format(k, v))
  print('=============================')
  print('', flush=True)


def run(*args, **kwargs):
  globals().update(kwargs)
  flags = tf.flags.FLAGS
  opt = Config()
  for key in flags:
    opt.setdefault(key, flags.get_flag_value(key, None))
  check_args(opt)
  data_config_file = Path(opt.data_config)
  if not data_config_file.exists():
    raise RuntimeError("dataset config file doesn't exist!")
  for _ext in ('json', 'yaml', 'yml'):  # for compatibility
    # apply a 2-stage (or master-slave) configuration, master can be
    # override by slave
    model_config_root = Path('parameters/root.{}'.format(_ext))
    if opt.p:
      model_config_file = Path(opt.p)
    else:
      model_config_file = Path('parameters/{}.{}'.format(opt.model, _ext))
    if model_config_root.exists():
      opt.update(Config(str(model_config_root)))
    if model_config_file.exists():
      opt.update(Config(str(model_config_file)))

  model_params = opt.get(opt.model, {})
  suppress_opt_by_args(model_params, *args)
  opt.update(model_params)
  model = get_model(opt.model)(**model_params)
  root = '{}/{}'.format(opt.save_dir, model.name)
  if opt.comment:
    root += '_' + opt.comment
  opt.root = root
  verbosity = tf.logging.DEBUG if opt.v else tf.logging.INFO
  # map model to trainer, ~~manually~~ automatically, by setting `_trainer`
  # attribute in models
  trainer = model.trainer
  train_data, test_data, infer_data = fetch_datasets(data_config_file, opt)
  train_config, test_config, infer_config = init_loader_config(opt)
  test_config.subdir = test_data.name
  infer_config.subdir = 'infer'
  # start fitting!
  if opt.v:
    dump(opt)
  with trainer(model, root, verbosity) as t:
    if opt.seed is not None:
      t.set_seed(opt.seed)
    # prepare loader
    loader = partial(QuickLoader, n_threads=opt.threads)
    train_loader = loader(train_data, 'train', train_config,
                          augmentation=True)
    val_loader = loader(train_data, 'val', train_config,
                        batch=1,
                        crop=opt.val_data_crop,
                        steps_per_epoch=opt.val_num)
    test_loader = loader(test_data, 'test', test_config)
    infer_loader = loader(infer_data, 'infer', infer_config)
    # fit
    t.fit([train_loader, val_loader], train_config)
    # validate
    t.benchmark(test_loader, test_config)
    # do inference
    t.infer(infer_loader, infer_config)
    if opt.export:
      t.export(opt.root + '/exported', opt.freeze)
