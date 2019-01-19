"""
Unit test for DataLoader.Loader
"""
import os

if not os.getcwd().endswith('UTest'):
  os.chdir('UTest')
from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *
from VSR.Util.ImageProcess import *
from VSR.Framework.Callbacks import _viz_flow

DATASETS = load_datasets('./data/fake_datasets.yml')


def test_loader_prob():
  dut = DATASETS['BAR']
  prob = [0.46196357, 0.14616816, 0.11549089, 0.13816049, 0.13821688]
  config = Config(batch=1, scale=1, depth=1, steps_per_epoch=-1,
                  convert_to='RGB')
  r = BasicLoader(dut, 'test', config)
  mc = 10000
  p = r._random_select(mc, seed=1).values()
  epsilon = 1e-2
  for p, p_hat in zip(p, prob):
    assert np.abs(p / 1e4 - p_hat) <= epsilon

  r.change_select_method(Select.EQUAL_FILE)
  mc = 10000
  p = r._random_select(mc, seed=1).values()
  epsilon = 1e-2
  prob = [.2, .2, .2, .2, .2]
  for p, p_hat in zip(p, prob):
    assert np.abs(p / 1e4 - p_hat) <= epsilon


def test_basicloader_iter():
  dut = DATASETS['NORMAL']
  config = Config(batch=16, scale=4, depth=1, steps_per_epoch=200,
                  convert_to='RGB', crop='random')
  config.patch_size = 48
  r = BasicLoader(dut, 'train', config, True)
  it = r.make_one_shot_iterator('8GB')
  for hr, lr, name, _ in it:
    print(name, flush=True)
  it = r.make_one_shot_iterator('8GB')
  for hr, lr, name, _ in it:
    print(name, flush=True)


def test_quickloader_iter():
  dut = DATASETS['NORMAL']
  config = Config(batch=16, scale=4, depth=1, steps_per_epoch=200,
                  convert_to='RGB', crop='random')
  config.patch_size = 48
  r = QuickLoader(dut, 'train', config, True, n_threads=8)
  it = r.make_one_shot_iterator('8GB')
  for hr, lr, name, _ in it:
    print(name, flush=True)
  it = r.make_one_shot_iterator('8GB')
  for hr, lr, name, _ in it:
    print(name, flush=True)


def test_benchmark_basic():
  dut = DATASETS['NORMAL']
  epochs = 4
  config = Config(batch=8, scale=4, depth=1, patch_size=32,
                  steps_per_epoch=100, convert_to='RGB', crop='random')
  loader = BasicLoader(dut, 'train', config, True)
  for _ in range(epochs):
    r = loader.make_one_shot_iterator()
    list(r)


def test_benchmark_mp():
  dut = DATASETS['NORMAL']
  epochs = 4
  config = Config(batch=8, scale=4, depth=1, patch_size=32,
                  steps_per_epoch=100, convert_to='RGB', crop='random')
  loader = QuickLoader(dut, 'train', config, True, n_threads=8)
  for _ in range(epochs):
    r = loader.make_one_shot_iterator()
    list(r)


def test_read_flow():
  dut = DATASETS['FLOW']
  config = Config(batch=8, scale=1, depth=2, patch_size=256,
                  steps_per_epoch=100, convert_to='RGB', crop='random')
  loader = QuickLoader(dut, 'train', config, True, n_threads=8)
  r = loader.make_one_shot_iterator('1GB', shuffle=True)
  loader.prefetch('1GB')
  list(r)
  r = loader.make_one_shot_iterator('8GB', shuffle=True)
  img, flow, name, _ = list(r)[0]

  ref0 = img[0, 0, ...]
  ref1 = img[0, 1, ...]
  u = flow[0, 0, ..., 0]
  v = flow[0, 0, ..., 1]
  # array_to_img(ref0, 'RGB').show()
  # array_to_img(ref1, 'RGB').show()
  # array_to_img(_viz_flow(u, v), 'RGB').show()


def test_read_pair():
  dut = DATASETS['PAIR']
  config = Config(batch=4, scale=1, depth=1, patch_size=64,
                  steps_per_epoch=10, convert_to='RGB', crop='random')
  loader = QuickLoader(dut, 'train', config, True, n_threads=8)
  r = loader.make_one_shot_iterator('1GB', shuffle=True)
  loader.prefetch('1GB')
  list(r)
  r = loader.make_one_shot_iterator('8GB', shuffle=True)
  for hr, pair, name, _ in r:
    assert np.all(hr == pair)


def test_cifar_loader():
  from tqdm import tqdm
  dut = DATASETS['NUMPY']
  config = Config(batch=8, scale=4, depth=1, patch_size=32,
                  steps_per_epoch=100, convert_to='RGB')
  loader = BasicLoader(dut, 'train', config, False)
  r = loader.make_one_shot_iterator()
  list(tqdm(r))


def test_tfrecord():
  dut = DATASETS['TFRECORD']
  loader = QuickLoader(dut, 'train',
                       Config(batch=16, steps_per_epoch=200, convert_to='RGB'))
  with tf.Session() as sess:
    it = loader.make_one_shot_iterator()
    for a, b, c, d in it:
      print(c)
  loader = QuickLoader(dut, 'test',
                       Config(batch=16, steps_per_epoch=200, convert_to='RGB'))
  with tf.Session() as sess:
    it = loader.make_one_shot_iterator()
    for a, b, c, d in it:
      print(c)


def test_crop_center():
  dut = DATASETS['NORMAL']
  config = Config(batch=1, scale=1, depth=1, patch_size=32, crop='center')
  np.random.seed(1)
  loader = QuickLoader(dut, 'test', config, False, n_threads=8)
  ref = QuickLoader(dut, 'test', config, False, n_threads=8, crop='not')
  for t, r in zip(loader.make_one_shot_iterator(),
                  ref.make_one_shot_iterator()):
    h, w = r[0].shape[1:3]
    ph, pw = t[0].shape[1:3]
    center = r[0][:, (h - ph) // 2:(h - ph) // 2 + ph,
             (w - pw) // 2:(w - pw) // 2 + pw, :]
    assert np.all(t[0] == center)


def test_crop_stride():
  dut = DATASETS['NORMAL']
  config = Config(batch=1, scale=1, depth=1, patch_size=32, crop='stride')
  np.random.seed(1)
  loader = QuickLoader(dut, 'test', config, False, n_threads=8)
  ref = QuickLoader(dut, 'test', config, False, n_threads=8, crop='not')
  ref = list(ref.make_one_shot_iterator())[0][0]
  patches = [t[0] for t in loader.make_one_shot_iterator()]
  patches = np.concatenate(patches)
  shape = patches.shape
  rows = ref.shape[1] // 32
  cols = ref.shape[2] // 32
  assert rows * cols == shape[0]
  re = patches.reshape([rows, cols, *shape[1:]])
  re = re.transpose([0, 2, 1, 3, 4])
  re = re.reshape([shape[1] * rows, shape[2] * cols, shape[3]])
  assert np.all(ref[0, :re.shape[0], :re.shape[1], :] == re)
