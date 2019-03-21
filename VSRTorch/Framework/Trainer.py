#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import time

import numpy as np
import torch
import tqdm

from .Environment import Env
from .Summary import Summarizer
from VSR.Util.Config import Config


def _ensemble_expand(feature):
  r0 = feature
  r1 = np.rot90(feature, 1, axes=[1, 2])
  r2 = np.rot90(feature, 2, axes=[1, 2])
  r3 = np.rot90(feature, 3, axes=[1, 2])
  r4 = np.flip(feature, axis=2)
  r5 = np.rot90(r4, 1, axes=[1, 2])
  r6 = np.rot90(r4, 2, axes=[1, 2])
  r7 = np.rot90(r4, 3, axes=[1, 2])
  return r0, r1, r2, r3, r4, r5, r6, r7


def _ensemble_reduce_mean(outputs):
  results = []
  for i in outputs:
    outputs_ensemble = [
      i[0],
      np.rot90(i[1], 3, axes=[1, 2]),
      np.rot90(i[2], 2, axes=[1, 2]),
      np.rot90(i[3], 1, axes=[1, 2]),
      np.flip(i[4], axis=2),
      np.flip(np.rot90(i[5], 3, axes=[1, 2]), axis=2),
      np.flip(np.rot90(i[6], 2, axes=[1, 2]), axis=2),
      np.flip(np.rot90(i[7], 1, axes=[1, 2]), axis=2),
    ]
    results.append(np.concatenate(outputs_ensemble).mean(axis=0, keepdims=True))
  return results


def to_tensor(x, cuda=False):
  x = torch.Tensor(x.copy())
  x = x.transpose(1, 2).transpose(1, 3).contiguous() / 255.0
  if cuda and torch.cuda.is_available():
    x = x.cuda()
  return x


def from_tensor(x):
  y = x.transpose([0, 2, 3, 1]) * 255
  return y


class SRTrainer(Env):
  v = Config()

  def query_config(self, config, **kwargs):
    assert isinstance(config, Config)
    config.update(kwargs)
    self.v.epoch = config.epoch  # current epoch
    self.v.epochs = config.epochs  # total epochs
    self.v.lr = config.lr  # learning rate
    self.v.lr_schedule = config.lr_schedule
    self.v.memory_limit = config.memory_limit
    self.v.feature_callbacks = config.feature_callbacks or []
    self.v.label_callbacks = config.label_callbacks or []
    self.v.output_callbacks = config.output_callbacks or []
    self.v.validate_every_n_epoch = config.validate_every_n_epoch or 1
    self.v.subdir = config.subdir
    self.v.random_val = config.random_val
    self.v.ensemble = config.ensemble
    self.v.cuda = config.cuda
    return self.v

  def fit_init(self) -> bool:
    v = self.v
    v.epoch = self._restore()
    if v.epoch >= v.epochs:
      self._logger.info(f'Found pre-trained epoch {v.epoch}>=target {v.epochs},'
                        ' quit fitting.')
      return False
    self._logger.info('Fitting: {}'.format(self.model.name.upper()))
    v.writer = Summarizer(str(self._logd), self.model.name)
    return True

  def fit_close(self):
    # flush all pending summaries to disk
    if isinstance(self.v.writer, Summarizer):
      self.v.writer.close()
    self._logger.info(f'Training {self.model.name.upper()} finished.')

  def fit(self, loaders, config, **kwargs):
    v = self.query_config(config, **kwargs)
    v.train_loader, v.val_loader = loaders
    if not self.fit_init():
      return
    for epoch in range(self.last_epoch + 1, v.epochs + 1):
      v.epoch = epoch
      train_iter = v.train_loader.make_one_shot_iterator(
        v.memory_limit, shuffle=True)
      if hasattr(v.train_loader, 'prefetch'):
        v.train_loader.prefetch(v.memory_limit)
      date = time.strftime('%Y-%m-%d %T', time.localtime())
      v.avg_meas = {}
      if v.lr_schedule and callable(v.lr_schedule):
        v.lr = v.lr_schedule(steps=v.epoch)
      print('| {} | Epoch: {}/{} | LR: {:.2g} |'.format(
        date, v.epoch, v.epochs, v.lr))
      with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
        self.model.to_train()
        for items in r:
          label, feature, name, post = items[:4]
          self.fn_train_each_step(label, feature, name, post)
          r.set_postfix(v.loss)
      for _k, _v in v.avg_meas.items():
        _v = np.mean(_v)
        v.writer.scalar(_k, _v, step=v.epoch, collection='train')
        print('| Epoch average {} = {:.6f} |'.format(_k, _v))
      if v.epoch % v.validate_every_n_epoch == 0:
        self.benchmark(v.val_loader, v)
      self._save_model(v.epoch)
    self.fit_close()

  def fn_train_each_step(self, label=None, feature=None, name=None, post=None):
    v = self.v
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    feature = to_tensor(feature, v.cuda)
    label = to_tensor(label, v.cuda)
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
    v.color_format = loader.color_format

    self._restore(config.epoch)
    v.mean_metrics = {}
    v.loader = loader
    it = v.loader.make_one_shot_iterator(v.memory_limit, shuffle=v.random_val)
    self.model.to_eval()
    for items in tqdm.tqdm(it, 'Test', ascii=True):
      label, feature, name, post = items[:4]
      self.fn_benchmark_each_step(label, feature, name, post)
    for _k, _v in v.mean_metrics.items():
      _v = np.mean(_v)
      v.writer.scalar(_k, _v, step=v.epoch, collection='eval')
      print('{}: {:.6f}'.format(_k, _v), end=', ')
    print('')

  def fn_benchmark_each_step(self, label=None, feature=None, name=None,
                             post=None):
    v = self.v
    origin_feat = feature
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
    for fn in v.label_callbacks:
      label = fn(label, name=name)
    feature = to_tensor(feature, v.cuda)
    label = to_tensor(label, v.cuda)
    outputs, metrics = self.model.eval([feature], [label], epoch=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    outputs = [from_tensor(x) for x in outputs]
    for fn in v.output_callbacks:
      outputs = fn(outputs, input=origin_feat, label=label, name=name,
                   mode=v.color_format, subdir=v.subdir)

  def infer(self, loader, config, **kwargs):
    """Infer SR images.

    Args:
        loader: a loader for enumerating LR images
        config: inferring configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    v.color_format = loader.color_format

    self._restore(config.epoch)
    it = loader.make_one_shot_iterator()
    if len(it):
      self._logger.info('Inferring {} at epoch {}'.format(
        self.model.name, self.last_epoch))
    else:
      return
    # use original images in inferring
    self.model.to_eval()
    for items in tqdm.tqdm(it, 'Infer', ascii=True):
      feature = items[0]
      name = items[2]
      self.fn_infer_each_step(None, feature, name)

  def fn_infer_each_step(self, label=None, feature=None, name=None, post=None):
    v = self.v
    origin_feat = feature
    for fn in v.feature_callbacks:
      feature = fn(feature, name=name)
    if v.ensemble:
      # add self-ensemble boosting metric score
      feature_ensemble = _ensemble_expand(feature)
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
      feature = to_tensor(feature, v.cuda)
      outputs, _ = self.model.eval([feature])
      outputs = [from_tensor(x) for x in outputs]
    for fn in v.output_callbacks:
      outputs = fn(outputs, input=origin_feat, name=name, subdir=v.subdir,
                   mode=v.color_format)
