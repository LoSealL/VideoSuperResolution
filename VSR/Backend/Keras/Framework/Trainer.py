#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

import logging

import numpy as np
import tensorflow as tf
import tqdm

from VSR.Util.Config import Config
from VSR.Util.Ensemble import Ensembler
from .Environment import Env

LOG = logging.getLogger('VSR.Framework.Keras')


def to_tensor(x):
  return x / 255.0


def from_tensor(x):
  return x * 255.0


class SRTrainer(Env):
  v = Config()

  def query_config(self, config, **kwargs):
    config = Config(config or {})
    config.update(kwargs)
    self.v.epochs = config.epochs or 1  # total epochs
    self.v.batch_shape = config.batch_shape or [1, -1, -1, -1]
    self.v.steps = config.steps or 200
    self.v.val_steps = config.val_steps or -1
    self.v.lr = config.lr or 1e-4  # learning rate
    self.v.lr_schedule = config.lr_schedule
    self.v.memory_limit = config.memory_limit
    self.v.inference_results_hooks = config.inference_results_hooks or []
    self.v.validate_every_n_epoch = config.validate_every_n_epoch or 1
    self.v.traced_val = config.traced_val
    self.v.ensemble = config.ensemble
    self.v.cuda = config.cuda
    self.v.caching = config.caching_dataset
    return self.v

  def fit_init(self) -> bool:
    v = self.v
    v.epoch = self._restore()
    if v.epoch >= v.epochs:
      LOG.info(f'Found pre-trained epoch {v.epoch}>=target {v.epochs},'
               ' quit fitting.')
      return False
    LOG.info(f'Fitting: {self.model.name.upper()}')
    if self._logd:
      v.writer = tf.summary.create_file_writer(str(self._logd),
                                               name=self.model.name)
      v.writer.set_as_default()
    return True

  def fit_close(self):
    # flush all pending summaries to disk
    LOG.info(f'Training {self.model.name.upper()} finished.')
    if self.v.writer is not None:
      self.v.writer.close()

  def fit(self, loaders, config, **kwargs):
    v = self.query_config(config, **kwargs)
    v.train_loader, v.val_loader = loaders
    if not self.fit_init():
      return
    mem = v.memory_limit
    for epoch in range(self.last_epoch + 1, v.epochs + 1):
      v.epoch = epoch
      train_iter = v.train_loader.make_one_shot_iterator(v.batch_shape,
                                                         v.steps,
                                                         shuffle=True,
                                                         memory_limit=mem,
                                                         caching=v.caching)
      v.train_loader.prefetch(shuffle=True, memory_usage=mem)
      v.avg_meas = {}
      if v.lr_schedule and callable(v.lr_schedule):
        v.lr = v.lr_schedule(steps=v.epoch)
      LOG.info(f"| Epoch: {v.epoch}/{v.epochs} | LR: {v.lr:.2g} |")
      with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
        self.model.to_train()
        for items in r:
          self.fn_train_each_step(items)
          r.set_postfix(v.loss)
      for _k, _v in v.avg_meas.items():
        _v = np.mean(_v)
        tf.summary.scalar(_k, _v, step=v.epoch, description='train')
        LOG.info(f"| Epoch average {_k} = {_v:.6f} |")
      if v.epoch % v.validate_every_n_epoch == 0 and v.val_loader:
        # Hard-coded memory limitation for validating
        self.benchmark(v.val_loader, v, memory_limit='1GB')
      self._save_model(v.epoch)
    self.fit_close()

  def fn_train_each_step(self, pack):
    v = self.v
    feature = to_tensor(pack['lr'])
    label = to_tensor(pack['hr'])
    loss = self.model.train([feature], [label], v.lr)
    for _k, _v in loss.items():
      v.avg_meas[_k] = \
        v.avg_meas[_k] + [_v] if v.avg_meas.get(_k) else [_v]
      loss[_k] = '{:08.5f}'.format(_v)
    v.loss = loss

  def benchmark(self, loader, config, **kwargs):
    """Benchmark/validate the model.

    Args:
        loader: a loader for enumerating LR images
        config: benchmark configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    self._restore(config.epoch)
    v.mean_metrics = {}
    v.loader = loader
    it = v.loader.make_one_shot_iterator(v.batch_shape, v.val_steps,
                                         shuffle=not v.traced_val,
                                         memory_limit=v.memory_limit,
                                         caching=v.caching)
    self.model.to_eval()
    for items in tqdm.tqdm(it, 'Test', ascii=True):
      self.fn_benchmark_each_step(items)
    log_message = str()
    for _k, _v in v.mean_metrics.items():
      _v = np.mean(_v)
      tf.summary.scalar(_k, _v, step=v.epoch, description='eval')
      log_message += f"{_k}: {_v:.6f}, "
    log_message = log_message[:-2] + "."
    LOG.info(log_message)

  def fn_benchmark_each_step(self, pack):
    v = self.v
    feature = to_tensor(pack['lr'])
    label = to_tensor(pack['hr'])
    outputs, metrics = self.model.eval([feature], [label], epoch=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    outputs = [from_tensor(x) for x in outputs]
    for fn in v.inference_results_hooks:
      outputs = fn(outputs, names=pack['name'])
      if outputs is None:
        break

  def infer(self, loader, config, **kwargs):
    """Infer SR images.

    Args:
        loader: a loader for enumerating LR images
        config: inferring configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    self._restore(config.epoch)
    it = loader.make_one_shot_iterator([1, -1, -1, -1], -1)
    if hasattr(it, '__len__'):
      if len(it) == 0:
        return
      LOG.info(f"Inferring {self.model.name} at epoch {self.last_epoch}")
    # use original images in inferring
    self.model.to_eval()
    for items in tqdm.tqdm(it, 'Infer', ascii=True):
      self.fn_infer_each_step(items)

  def fn_infer_each_step(self, pack):
    v = self.v
    if v.ensemble:
      # add self-ensemble boosting metric score
      feature_ensemble = Ensembler.expand(pack['lr'])
      outputs_ensemble = []
      for f in feature_ensemble:
        f = to_tensor(f)
        y, _ = self.model.eval([f])
        y = [from_tensor(x) for x in y]
        outputs_ensemble.append(y)
      outputs = []
      for i in range(len(outputs_ensemble[0])):
        outputs.append([j[i] for j in outputs_ensemble])
      outputs = Ensembler.merge(outputs)
    else:
      feature = to_tensor(pack['lr'])
      outputs, _ = self.model.eval([feature])
      outputs = [from_tensor(x) for x in outputs]
    for fn in v.inference_results_hooks:
      outputs = fn(outputs, names=pack['name'])
      if outputs is None:
        break
