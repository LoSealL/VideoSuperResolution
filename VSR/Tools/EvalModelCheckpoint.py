"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Specifies a model and evaluate its corresponded checkpoint.
"""

import tensorflow as tf
import numpy as np
import shutil
from pathlib import Path
from functools import partial

from ..Models import get_model
from ..DataLoader.Loader import QuickLoader
from ..Util.Config import Config
from .Run import check_args, fetch_datasets, init_loader_config
from .Eval import Eval

tf.flags.DEFINE_string("checkpoint_dir", None, help="checkpoint directory.")
FLAGS = tf.flags.FLAGS


def maybe_checkpoint(path):
    path = Path(path)
    epoch = FLAGS.epochs
    candidates = list(path.rglob(f'*{epoch:04d}.ckpt.*'))
    if len(candidates) == 0:
        tf.logging.warning(f"checkpoint of ep-{epoch} is not found.")
        return next(path.rglob(f'*.ckpt.*')).parent.parent
    for d in set([d.parent for d in candidates]):
        if FLAGS.model in str(d):
            candidates = list(d.rglob(f'*{epoch:04d}.ckpt.*'))
            break
    try:
        shutil.rmtree(f'/tmp/{FLAGS.model}/save')
    except FileNotFoundError:
        pass
    dst = Path(f'/tmp/{FLAGS.model}/save')
    dst.mkdir(parents=True, exist_ok=False)
    [shutil.copy(x, str(dst)) for x in candidates]
    return dst.parent


def get_outputs(outputs, config, **kwargs):
    index = config.output_index
    imgs = outputs[index] if isinstance(outputs, list) else outputs
    config.data.append(imgs)


def evaluate():
    opt = Config()
    for key in FLAGS:
        opt.setdefault(key, FLAGS.get_flag_value(key, None))
    check_args(opt)
    data_config_file = Path(opt.data_config)
    if not data_config_file.exists():
        raise RuntimeError("dataset config file doesn't exist!")
    for _suffix in ('json', 'yaml', 'yml'):  # for compatibility
        # apply a 2-stage (or master-slave) configuration, master can be
        # override by slave
        model_config_root = Path(f'parameters/root.{_suffix}')
        if opt.p:
            model_config_file = Path(opt.p)
        else:
            model_config_file = Path(f'parameters/{opt.model}.{_suffix}')
        if model_config_root.exists():
            opt.update(Config(str(model_config_root)))
        if model_config_file.exists():
            opt.update(Config(str(model_config_file)))
    root = maybe_checkpoint(opt.checkpoint_dir)
    model_params = opt.get(opt.model, {})
    opt.update(model_params)
    model = get_model(opt.model)(**model_params)
    trainer = model.trainer
    _, test_data, infer_data = fetch_datasets(data_config_file, opt)
    _, test_config, infer_config = init_loader_config(opt)
    test_config.subdir = test_data.name
    infer_config.subdir = 'infer'
    with trainer(model, root, tf.logging.INFO) as t:
        loader = QuickLoader(test_data, 'test', test_config, n_threads=4)
        opt.data = []
        test_config.output_callbacks = [partial(get_outputs, config=opt)]
        t.benchmark(loader, test_config)
        # this implies 1:1 on label and fake images
        label_images = [x for x, _, _ in loader.make_one_shot_iterator()]
    Eval.evaluate(label_images, opt.data)
