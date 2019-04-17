#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/17 下午5:53

import os
import subprocess

if not os.getcwd().endswith('UTest'):
  os.chdir('UTest')

_WORKDIR = r"/tmp/vsr/utest/"
_TCMD = r"python run.py --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1 --save_dir={}".format(
  _WORKDIR)
_ECMD = r"python ben.py --enable_psnr --enable_ssim --steps_per_epoch=1 --data_config=../UTest/data/fake_datasets.yml"


def test_train_srcnn():
  cmd = "{} --model={}".format(_TCMD, 'srcnn')
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)


def test_train_espcn():
  cmd = "{} --model={}".format(_TCMD, 'espcn')
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)


def test_train_vespcn():
  cmd = "{} --model={} -p={}".format(_TCMD, 'vespcn',
                                     '../UTest/data/vespcn.yaml')
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)


def test_eval_espcn():
  cmd = "{} --model={} --test={} --checkpoint_dir={}".format(_ECMD, 'espcn',
                                                             'bar',
                                                             _WORKDIR + 'espcn')
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)


def test_eval_vespcn():
  cmd = "{} --model={} --test={} --checkpoint_dir={} --output_index=:".format(
    _ECMD, 'vespcn', 'raw', _WORKDIR + 'vespcn')
  cwd = '../Train'
  subprocess.call(cmd, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)
