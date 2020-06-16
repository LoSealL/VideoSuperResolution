#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import logging
from pathlib import Path

import numpy as np
import torch

LOG = logging.getLogger('VSR.Framework')


def _make_ckpt_name(name, step):
  return '{}_ep{:04d}.pth'.format(name, step)


def _parse_ckpt_name(name):
  if not name:
    return 0
  model_name, epochs = Path(name).stem.split('.')[0].split('_')
  return int(epochs[2:])


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
    self.last_epoch = 0

  def _startup(self):
    if isinstance(self._saved, Path):
      self._saved.mkdir(parents=True, exist_ok=True)
    if isinstance(self._logd, Path):
      self._logd.mkdir(parents=True, exist_ok=True)
      if LOG.isEnabledFor(logging.DEBUG):
        hdl = logging.FileHandler(self._logd / 'training.txt')
        LOG.addHandler(hdl)

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

  def _find_last_ckpt(self, pattern):
    # restore the latest checkpoint in save dir
    # sort as modification time
    if not isinstance(self._saved, Path): return
    ckpt = sorted(self._saved.glob(pattern), key=lambda x: x.stat().st_mtime_ns)
    return ckpt[-1].resolve() if ckpt else None

  def _restore_model(self, epoch=None, pth=None, map_location=None):
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
        LOG.info(f"Restoring params for {key} from {fp}.")
        try:
          last_epoch = max(_parse_ckpt_name(str(fp)), last_epoch)
        except ValueError:
          last_epoch = 0
        try:
          model.load_state_dict(torch.load(str(fp), map_location=map_location))
        except RuntimeError as ex:
          print(ex)
          LOG.warning(f"Couldn't restore state for {key} from {fp}.")
    if pth is None and isinstance(self._saved, Path):
      for key, opt in self.model.opts.items():
        fp = self._saved / f'{key}.pth'
        try:
          opt.load_state_dict(torch.load(str(fp)))
        except (ValueError, FileNotFoundError):
          LOG.warning(f"trying to restore state for optimizer {key}, "
                      "but failed.")
    return last_epoch

  def _save_model(self, step):
    if not isinstance(self._saved, Path): return
    for key, model in self.model.modules.items():
      fp = self._saved / _make_ckpt_name(key, step)
      torch.save(model.state_dict(), str(fp))
    for key, opt in self.model.opts.items():
      fp = self._saved / f'{key}.pth'
      torch.save(opt.state_dict(), str(fp))

  def _restore(self, epoch=None, map_location=None):
    # restore graph
    if self._restored or self.model.loaded:
      return self.last_epoch
    self.last_epoch = self._restore_model(epoch, map_location=map_location)
    self._restored = True
    return self.last_epoch

  def set_seed(self, seed):
    """set a seed for RNG

    Note: RNG in torch and numpy is different.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

  def export(self, export_dir='.', version=1):
    """export ONNX model.

    Args:
      export_dir: path to save onnx files.
      version: (optional) a child-folder to control output versions.
    """

    export_path = Path(export_dir) / str(version)
    while export_path.exists():
      version += 1  # step ahead 1 version
      export_path = Path(export_dir) / str(version)
    export_path.mkdir(exist_ok=False, parents=True)
    self.model.export(export_path)
    LOG.info(f"Export ONNX to {str(export_path)}")
