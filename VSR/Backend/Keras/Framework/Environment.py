#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 5 - 30

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

LOG = logging.getLogger('VSR.Framework.Keras')


def _parse_ckpt_name(name):
  if not name:
    return 0
  model_name, epochs = Path(name).stem.split('-')
  return int(epochs)


class Env:
  """Pytorch model runtime Env-ironment.

  Args:
    model: a Model object (note it's NOT nn.Module), representing a container
      of multiple nn.Module objects. See `VSRTorch.Models.Model` for details.
    work_dir: a folder path, working directory of this environment.

  Usage:
    Use `with` syntax to enter the Env:

    >>> with Env(...) as e: ...
  """

  def __init__(self, model, work_dir=None):
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
      self.ckpt = tf.train.Checkpoint(**self.model.modules, **self.model.opts)
      self.saver = tf.train.CheckpointManager(self.ckpt, self._saved, None,
                                              checkpoint_name=self.model.name)
    if isinstance(self._logd, Path):
      self._logd.mkdir(parents=True, exist_ok=True)
      _logger = logging.getLogger('VSR')
      if _logger.isEnabledFor(logging.DEBUG):
        fd = logging.FileHandler(self._logd / 'vsr_debug.log', encoding='utf-8')
        fd.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        _logger.addHandler(fd)

  def _close(self):
    """TODO anything to close?"""
    pass

  def __enter__(self):
    """Create session of tensorflow and build model graph"""

    self._startup()
    self.model.display()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Close session"""

    self._close()

  @property
  def model(self):
    return self._m

  def _restore_model(self):
    last_epoch = 0
    ckpt = self.saver.latest_checkpoint
    if ckpt:
      self.ckpt.restore(ckpt)
    try:
      last_epoch = max(_parse_ckpt_name(str(ckpt)), last_epoch)
    except ValueError:
      last_epoch = 0
    return last_epoch

  def _save_model(self, step):
    if not isinstance(self._saved, Path): return
    self.saver.save(step)

  def _restore(self, epoch=None):
    # restore graph
    if self._restored:
      return self.last_epoch
    self.last_epoch = self._restore_model()
    self._restored = True
    return self.last_epoch

  def set_seed(self, seed):
    """set a seed for RNG

    Note: RNG in tensorflow and numpy is different.
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)

  def export(self, export_dir='.', version=1):
    """export saved model.

    Args:
      export_dir: path to saved_model dirs.
      version: (optional) a child-folder to control output versions.
    """

    export_path = Path(export_dir) / str(version)
    while export_path.exists():
      version += 1  # step ahead 1 version
      export_path = Path(export_dir) / str(version)
    export_path.mkdir(exist_ok=False, parents=True)
    self.model.export(export_path)
    LOG.info(f"Export saved model to {str(export_path)}")
