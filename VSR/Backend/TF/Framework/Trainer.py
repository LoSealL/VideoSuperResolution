"""
Copyright: Wenyi Tang 2017-2020
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Oct 15th 2018

Extend the pre-Environment module, provide different and extensible
training methodology for SISR, VSR or other image tasks.
"""

import logging
from pathlib import Path

import numpy as np
import tqdm

from VSR.Util import Config, to_list
from .. import tf

LOG = logging.getLogger('VSR.Framework')


def _make_ckpt_name(name, scale, step):
  return '{}-sc{}-ep{:04d}.ckpt'.format(name, scale, step)


def _parse_ckpt_name(name):
  # sample name: {model}-sc{scale}-ep{epoch}.ckpt(.index)
  if not name:
    return 0
  model_name, scale, epochs = Path(name).stem.split('.')[0].split('-')
  return int(epochs[2:])


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


class Trainer:
  """A pure interface trainer.

     A trainer provides following APIs:
       >>> Trainer.fit
       >>> Trainer.infer
       >>> Trainer.benchmark
       >>> Trainer.export

     Args:
         model: the SR model object. @see SuperResolution
         work_dir: the dir to save training checkpoints and logs
         verbose: tf logging level
     """

  def __init__(self, model, work_dir):
    self._m = model
    self._saved = None
    self._logd = None
    if work_dir is not None:
      self._saved = Path(work_dir) / 'save'
      self._logd = Path(work_dir) / 'log'
    self._restored = False

  def _startup(self):
    if isinstance(self._saved, Path):
      self._saved.mkdir(parents=True, exist_ok=True)
    if isinstance(self._logd, Path):
      self._logd.mkdir(parents=True, exist_ok=True)
      _logger = logging.getLogger('VSR')
      if _logger.isEnabledFor(logging.DEBUG):
        fd = logging.FileHandler(self._logd / 'vsr_debug.log', encoding='utf-8')
        fd.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        _logger.addHandler(fd)
    if self.model.compiled:
      self.graph = tf.get_default_graph()
    else:
      with tf.Graph().as_default() as g:
        self.model.compile()
        self.graph = g

  def __enter__(self):
    """Create session of tensorflow and build model graph"""

    self._startup()
    conf = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(graph=self.graph, config=conf)
    sess.__enter__()
    self.model.display()
    self.savers = self.model.savers
    sess.run(tf.global_variables_initializer())
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Close session"""

    sess = tf.get_default_session()
    sess.__exit__(exc_type, exc_val, exc_tb)

  def _find_last_ckpt(self):
    # restore the latest checkpoint in save dir
    if self._saved is not None:
      ckpt = tf.train.get_checkpoint_state(self._saved)
      if ckpt and ckpt.model_checkpoint_path:
        return tf.train.latest_checkpoint(self._saved)
      # try another way
      ckpt = to_list(self._saved.glob('*.ckpt.index'))
      # sort as modification time
      ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
      return self._saved / ckpt[-1].stem if ckpt else None

  def _restore_model(self, sess):
    last_checkpoint_step = 0
    if self.model.pre_ckpt is not None:
      _saved = Path(self.model.pre_ckpt)
    else:
      _saved = self._saved
    if _saved is None:
      return last_checkpoint_step
    for name in self.savers:
      saver = self.savers.get(name)
      ckpt = to_list(_saved.glob('{}*.index'.format(name)))
      if ckpt:
        ckpt = sorted(ckpt, key=lambda x: x.stat().st_mtime_ns)
        ckpt = _saved / ckpt[-1].stem
        try:
          saver.restore(sess, str(ckpt))
        except tf.errors.NotFoundError:
          LOG.warning(
              '{} of model {} could not be restored'.format(
                  name, self.model.name))
        last_checkpoint_step = _parse_ckpt_name(ckpt)
    return last_checkpoint_step

  def _save_model(self, sess, step):
    if self._saved is None:
      return
    for name in self.savers:
      saver = self.savers.get(name)
      file = self._saved / _make_ckpt_name(name, self.model.scale[0], step)
      saver.save(sess, str(file))

  def _restore(self):
    # restore graph
    sess = tf.get_default_session()
    if sess is None:
      raise RuntimeError('No session initialized')
    if self._restored:
      return sess
    self.last_epoch = self._restore_model(sess)
    self._restored = True
    return sess

  def export(self, export_dir='.', freeze_model=False):
    """Export model as protobuf

    Args:
        export_dir: directory to save the exported model
        freeze_model: freeze all trainable variables
    """

    self._restore()
    if freeze_model:
      self.model.export_freeze_model(export_dir)
    else:
      self.model.export_saved_model(export_dir)

  def set_seed(self, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

  def fit(self, *args, **kwargs):
    raise NotImplementedError

  def infer(self, *args, **kwargs):
    raise NotImplementedError

  def benchmark(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def model(self):
    return self._m


class VSR(Trainer):
  """Default trainer for task SISR or VSR"""
  v = Config()  # local variables
  """=======================================
      components, sub-functions, helpers
     =======================================
  """

  def query_config(self, config, **kwargs) -> Config:
    config = Config(config or {})
    config.update(kwargs)  # override parameters
    self.v.epoch = config.epoch  # current epoch
    self.v.epochs = config.epochs or 1  # total epochs
    self.v.lr = config.lr or 1e-4  # learning rate
    self.v.batch_shape = config.batch_shape or [1, -1, -1, -1]
    self.v.steps = config.steps or 200
    self.v.val_steps = config.val_steps or -1
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
    v.sess = self._restore()
    if self.last_epoch >= v.epochs:
      LOG.info(f'Found pre-trained epoch {v.epoch}>=target {v.epochs},'
               ' quit fitting.')
      return False
    LOG.info('Fitting: {}'.format(self.model.name.upper()))
    v.summary_writer = tf.summary.FileWriter(
        str(self._logd), graph=tf.get_default_graph())
    v.global_step = self.model.global_steps.eval()
    return True

  def fit_close(self):
    # flush all pending summaries to disk
    if self.v.summary_writer:
      self.v.summary_writer.close()
    LOG.info(f'Training {self.model.name.upper()} finished.')

  def fn_train_each_epoch(self):
    v = self.v
    mem = v.memory_limit
    train_iter = v.train_loader.make_one_shot_iterator(v.batch_shape,
                                                       v.steps,
                                                       shuffle=True,
                                                       memory_limit=mem,
                                                       caching=v.caching)
    v.train_loader.prefetch(v.memory_limit)
    v.avg_meas = {}
    if v.lr_schedule and callable(v.lr_schedule):
      v.lr = v.lr_schedule(steps=v.global_step)
    LOG.info(f"| Epoch: {v.epoch}/{v.epochs} | LR: {v.lr:.2g} |")
    with tqdm.tqdm(train_iter, unit='batch', ascii=True) as r:
      for items in r:
        self.fn_train_each_step(items)
        r.set_postfix(v.loss)
    for _k, _v in v.avg_meas.items():
      LOG.info(f"| Epoch average {_k} = {np.mean(_v):.6f} |")
    if v.epoch % v.validate_every_n_epoch == 0 and v.val_loader:
      self.benchmark(v.val_loader, v, epoch=v.epoch, memory_limit='1GB')
      v.summary_writer.add_summary(self.model.summary(), v.global_step)
    self._save_model(v.sess, v.epoch)

  def fn_train_each_step(self, pack):
    v = self.v
    loss = self.model.train_batch(pack['lr'], pack['hr'], learning_rate=v.lr,
                                  epochs=v.epoch)
    v.global_step = self.model.global_steps.eval()
    for _k, _v in loss.items():
      v.avg_meas[_k] = \
        v.avg_meas[_k] + [_v] if v.avg_meas.get(_k) else [_v]
      loss[_k] = '{:08.5f}'.format(_v)
    v.loss = loss

  def fn_infer_each_step(self, pack):
    v = self.v
    if v.ensemble:
      # add self-ensemble boosting metric score
      feature_ensemble = _ensemble_expand(pack['lr'])
      outputs_ensemble = []
      for f in feature_ensemble:
        y, _ = self.model.test_batch(f, None)
        outputs_ensemble.append(y)
      outputs = []
      for i in range(len(outputs_ensemble[0])):
        outputs.append([j[i] for j in outputs_ensemble])
      outputs = _ensemble_reduce_mean(outputs)
    else:
      outputs, _ = self.model.test_batch(pack['lr'], None)
    for fn in v.inference_results_hooks:
      outputs = fn(outputs, names=pack['name'])
      if outputs is None:
        break

  def fn_benchmark_each_step(self, pack):
    v = self.v
    outputs, metrics = self.model.test_batch(pack['lr'], pack['hr'],
                                             epochs=v.epoch)
    for _k, _v in metrics.items():
      if _k not in v.mean_metrics:
        v.mean_metrics[_k] = []
      v.mean_metrics[_k] += [_v]
    for fn in v.inference_results_hooks:
      outputs = fn(outputs, names=pack['name'])
      if outputs is None:
        break

  def fn_benchmark_body(self):
    v = self.v
    it = v.loader.make_one_shot_iterator(v.batch_shape, v.val_steps,
                                         shuffle=not v.traced_val,
                                         memory_limit=v.memory_limit,
                                         caching=v.caching)
    for items in tqdm.tqdm(it, 'Test', ascii=True):
      self.fn_benchmark_each_step(items)

  """=======================================
      Interface: fit, benchmark, infer
     =======================================
  """

  def fit(self, loaders, config, **kwargs):
    """Fit the model.

    Args:
        loaders: a tuple of 2 loaders, the 1st one is used for training,
          and the 2nd one is used for validating.
        config: fitting configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    v.train_loader, v.val_loader = loaders
    if not self.fit_init():
      return
    for epoch in range(self.last_epoch + 1, v.epochs + 1):
      v.epoch = epoch
      self.fn_train_each_epoch()
    self.fit_close()

  def infer(self, loader, config, **kwargs):
    """Infer SR images.

    Args:
        loader: a loader for enumerating LR images
        config: inferring configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    self._restore()
    it = loader.make_one_shot_iterator([1, -1, -1, -1], -1)
    if hasattr(it, '__len__'):
      if len(it):
        LOG.info('Inferring {} at epoch {}'.format(
            self.model.name, self.last_epoch))
      else:
        return
    # use original images in inferring
    for items in tqdm.tqdm(it, 'Infer', ascii=True):
      self.fn_infer_each_step(items)

  def benchmark(self, loader, config, **kwargs):
    """Benchmark/validate the model.

    Args:
        loader: a loader for enumerating LR images
        config: benchmark configuration, an instance of `Util.Config.Config`
        kwargs: additional arguments to override the same ones in config.
    """
    v = self.query_config(config, **kwargs)
    self._restore()
    v.mean_metrics = {}
    v.loader = loader
    self.fn_benchmark_body()
    log_message = str()
    for _k, _v in v.mean_metrics.items():
      _v = np.mean(_v)
      log_message += f"{_k}: {_v:.6f}, "
    log_message = log_message[:-2] + "."
    LOG.info(log_message)
