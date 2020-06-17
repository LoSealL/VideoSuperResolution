#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/17 下午5:53

import os
import subprocess
from VSR.Model import list_supported_models

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

_WORKDIR = r"/tmp/vsr/utest/"
_TCMD = ("python train.py {} --data_config=../Tests/data/fake_datasets.yml"
         "--dataset=normal --epochs=1 --steps=1 --save_dir={} --val_steps=1")
_ECMD = r"python eval.py {} --save_dir={} --ensemble -t=../Tests/data/set5_x2"


def train(model_name: str):
  cmd = _TCMD.format(str(model_name), _WORKDIR)
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)


def eval(model_name: str):
  cmd = _ECMD.format(str(model_name), _WORKDIR)
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)


def test_train_srcnn():
  train('srcnn')
  eval('srcnn')


def test_train_espcn():
  train('espcn')
  eval('espcn')


def test_other_models():
  for k in list_supported_models():
    if k in (
        'sofvsr', 'vespcn', 'frvsr', 'qprn', 'ufvsr', 'yovsr', 'tecogan',
        'spmc', 'rbpn'
    ):
      # skip video model
      continue
    train(k)
    eval(k)


if __name__ == '__main__':
  test_other_models()
