#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import logging
import time

import numpy as np
import torch
import tqdm

from VSR.Util.Config import Config
from .Environment import Env
from .Summary import Summarizer

LOG = logging.getLogger('VSR.Framework')


def _ensemble_expand(feature):
  r0 = feature
  r1 = np.rot90(feature, 1, axes=[-3, -2])
  r2 = np.rot90(feature, 2, axes=[-3, -2])
  r3 = np.rot90(feature, 3, axes=[-3, -2])
  r4 = np.flip(feature, axis=-2)
  r5 = np.rot90(r4, 1, axes=[-3, -2])
  r6 = np.rot90(r4, 2, axes=[-3, -2])
  r7 = np.rot90(r4, 3, axes=[-3, -2])
  return r0, r1, r2, r3, r4, r5, r6, r7


def _ensemble_reduce_mean(outputs):
  results = []
  for i in outputs:
    outputs_ensemble = [
      i[0],
      np.rot90(i[1], 3, axes=[-3, -2]),
      np.rot90(i[2], 2, axes=[-3, -2]),
      np.rot90(i[3], 1, axes=[-3, -2]),
      np.flip(i[4], axis=-2),
      np.flip(np.rot90(i[5], 3, axes=[-3, -2]), axis=-2),
      np.flip(np.rot90(i[6], 2, axes=[-3, -2]), axis=-2),
      np.flip(np.rot90(i[7], 1, axes=[-3, -2]), axis=-2),
    ]
    results.append(np.concatenate(outputs_ensemble).mean(axis=0, keepdims=True))
  return results


def to_tensor(x, cuda=False):
  x = torch.as_tensor(x / 255.0, dtype=torch.float32)
  if cuda and torch.cuda.is_available():
    x = x.cuda()
  return x


def from_tensor(x):
  return x * 255


class SRTrainer(Env):
  v = Config()

  def query_config(self, config, **kwargs):
    config = Config(config)
    config.update(kwargs)
    self.v.epochs = config.epochs or 1  # total epochs
    self.v.batch_shape = config.batch_shape or [1, -1, -1, -1]
    self.v.train_steps = config.steps or 200
    self.v.val_steps = config.val_steps or 10
    self.v.lr = config.lr or 1e-4  # learning rate
    self.v.lr_schedule = config.lr_schedule
    self.v.memory_limit = config.memory_limit
    self.v.inference_results_hooks = config.inference_results_hooks or []
    self.v.validate_every_n_epoch = config.validate_every_n_epoch or 1
    self.v.traced_val = config.traced_val
    self.v.ensemble = config.ensemble
    self.v.cuda = config.cuda
    self.v.map_location = 'cuda:0' if config.cuda and torch.cuda.is_available() else 'cpu'
    return self.v

  def fit_init(self) -> bool:
    v = self.v
    v.epoch = self._restore()
    if v.epoch >= v.epochs:
      LOG.info(f'Found pre-trained epoch {v.epoch}>=target {v.epochs},'
               ' quit fitting.')
      return False
    LOG.info('Fitting: {}'.format(self.model.name.upper()))
    if self._logd:
      v.writer = Summarizer(str(self._logd), self.model.name)
    return True

  def fit_close(self):
    # flush all pending summaries to disk
    if isinstance(self.v.writer, Summarizer):
      self.v.writer.close()
    LOG.info(f'Training {self.model.name.upper()} finished.')

  def fit(self, loaders, config, **kwargs):
    v = self.query_config(config, **kwargs)
    v.train_loader, v.val_loader = loaders
    if not self.fit_init():
      return
    mem = v.memory_limit
    for epoch in range(self.last_epoch + 1, v.epochs + 1):
      v.epoch = epoch
      train_iter = v.train_loader.make_one_shot_iterator(v.batch_shape,
                                                         v.train_steps,
                                                         shuffle=True,
                                                         memory_limit=mem)
      v.train_loader.prefetch(shuffle=True, memory_usage=mem)
      date = time.strftime('%Y-%m-%d %T', time.localtime())
      v.avg_meas = {}
      if v.lr_schedule and callable(v.lr_schedule):
        v.lr = v.lr_schedule(steps=v.epoch)
      print('| {} | Epoch: {}/{} | LR: {:.2g} |'.format(
        date, v.epoch, v.epochs, v.lr))
      with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
        self.model.to_train()
        for items in r:
          self.fn_train_each_step(items)
          r.set_postfix(v.loss)
      for _k, _v in v.avg_meas.items():
        _v = np.mean(_v)
        if isinstance(self.v.writer, Summarizer):
          v.writer.scalar(_k, _v, step=v.epoch, collection='train')
        print('| Epoch average {} = {:.6f} |'.format(_k, _v))
      if v.epoch % v.validate_every_n_epoch == 0 and v.val_loader:
        # Hard-coded memory limitation for validating
        self.benchmark(v.val_loader, v, memory_limit='1GB')
      self._save_model(v.epoch)
    self.fit_close()

  def fn_train_each_step(self, pack):
    v = self.v
    feature = to_tensor(pack['lr'], v.cuda)
    label = to_tensor(pack['hr'], v.cuda)
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
    self._restore(config.epoch, v.map_location)
    v.mean_metrics = {}
    v.loader = loader
    it = v.loader.make_one_shot_iterator(v.batch_shape, v.val_steps,
                                         shuffle=not v.traced_val,
                                         memory_limit=v.memory_limit)
    self.model.to_eval()
    for items in tqdm.tqdm(it, 'Test', ascii=True):
      with torch.no_grad():
        self.fn_benchmark_each_step(items)
    for _k, _v in v.mean_metrics.items():
      _v = np.mean(_v)
      if isinstance(self.v.writer, Summarizer):
        v.writer.scalar(_k, _v, step=v.epoch, collection='eval')
      print('{}: {:.6f}'.format(_k, _v), end=', ')
    print('')

  def fn_benchmark_each_step(self, pack):
    v = self.v
    feature = to_tensor(pack['lr'], v.cuda)
    label = to_tensor(pack['hr'], v.cuda)
    with torch.set_grad_enabled(False):
      outputs, metrics = self.model.eval([feature], [label], epoch=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    outputs = [from_tensor(x) for x in outputs]
    for fn in v.inference_results_hooks:
      outputs = fn(outputs, names=pack['name'])
      if outputs is None: break

  def infer(self, loader, config, **kwargs):
    """Infer SR images.

    Args:
        loader: a loader for enumerating LR images
        config: inferring configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    self._restore(config.epoch, v.map_location)
    it = loader.make_one_shot_iterator([1, -1, -1, -1], -1)
    if hasattr(it, '__len__'):
      if len(it):
        LOG.info('Inferring {} at epoch {}'.format(
          self.model.name, self.last_epoch))
      else:
        return
    # use original images in inferring
    self.model.to_eval()
    for items in tqdm.tqdm(it, 'Infer', ascii=True):
      with torch.no_grad():
        self.fn_infer_each_step(items)

  def fn_infer_each_step(self, pack):
    v = self.v
    with torch.set_grad_enabled(False):
      if v.ensemble:
        # add self-ensemble boosting metric score
        feature_ensemble = _ensemble_expand(pack['lr'])
        outputs_ensemble = []
        for f in feature_ensemble:
          f = to_tensor(f, v.cuda)
          y, _ = self.model.eval([f])
          y = [from_tensor(x) for x in y]
          outputs_ensemble.append(y)
        outputs = []
        for i in range(len(outputs_ensemble[0])):
          outputs.append([j[i] for j in outputs_ensemble])
        outputs = _ensemble_reduce_mean(outputs)
      else:
        feature = to_tensor(pack['lr'], v.cuda)
        outputs, _ = self.model.eval([feature])
        outputs = [from_tensor(x) for x in outputs]
    for fn in v.inference_results_hooks:
      outputs = fn(outputs, names=pack['name'])
      if outputs is None: break
