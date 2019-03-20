#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import logging
from pathlib import Path

import numpy as np
import torch


def _make_ckpt_name(name, step):
  return '{}_ep{:04d}.pth'.format(name, step)


def _parse_ckpt_name(name):
  if not name:
    return 0
  model_name, epochs = Path(name).stem.split('.')[0].split('_')
  return int(epochs[2:])


class Env:

  def __init__(self, model, work_dir, log_level='DEBUG', pre_train_model=None):
    self._m = model
    self._saved = Path(work_dir) / 'save'
    self._logd = Path(work_dir) / 'log'
    self._restored = False
    self._logger = logging.getLogger("VSR.Trainer")
    self._logger.setLevel(log_level)
    self._pth = Path(pre_train_model or '')

  def _startup(self):
    self._saved.mkdir(parents=True, exist_ok=True)
    self._logd.mkdir(parents=True, exist_ok=True)
    if not self._pth.exists() or not self._pth.is_file():
      self._pth = None
    if self._logger.isEnabledFor(logging.DEBUG):
      hdl = logging.FileHandler(self._logd / 'training.txt')
      self._logger.addHandler(hdl)

  def _close(self):
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

  def _find_last_ckpt(self, pattern):
    # restore the latest checkpoint in save dir
    # sort as modification time
    ckpt = sorted(self._saved.glob(pattern), key=lambda x: x.stat().st_mtime_ns)
    return ckpt[-1].resolve() if ckpt else None

  def _restore_model(self, epoch=None, pth=None):
    last_epoch = 0
    for key, model in self.model.modules.items():
      if pth is None:
        if epoch is None:
          ckpt = f'*{key}*.pth'
        else:
          ckpt = _make_ckpt_name(key, epoch)
        fp = self._find_last_ckpt(ckpt)
      else:
        fp = pth
      if fp:
        self._logger.info("Restoring params from %s", str(fp))
        try:
          last_epoch = max(_parse_ckpt_name(str(fp)), last_epoch)
        except ValueError:
          last_epoch = 0
        model.load_state_dict(torch.load(str(fp)))
    if pth is None:
      for key, opt in self.model.opts.items():
        fp = self._saved / f'{key}.pth'
        try:
          opt.load_state_dict(torch.load(str(fp)))
        except (ValueError, FileNotFoundError):
          self._logger.warning(f"trying to restore state for optimizer {key}, "
                               "but failed.")
    return last_epoch

  def _save_model(self, step):
    for key, model in self.model.modules.items():
      fp = self._saved / _make_ckpt_name(key, step)
      torch.save(model.state_dict(), str(fp))
    for key, opt in self.model.opts.items():
      fp = self._saved / f'{key}.pth'
      torch.save(opt.state_dict(), str(fp))

  def _restore(self, epoch=None):
    # restore graph
    if self._restored:
      return self.last_epoch
    self.last_epoch = self._restore_model(epoch, self._pth)
    self._restored = True
    return self.last_epoch

  def set_seed(self, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
