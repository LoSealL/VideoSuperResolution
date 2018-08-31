"""
Unit test for DataLoader.Loader
"""

from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')


def test_quickloader_prob():
    DUT = DATASETS['SET5']
    PROB = [0.46196357, 0.14616816, 0.11549089, 0.13816049, 0.13821688]
    r = QuickLoader(1, DUT, 'test', 1, 1)
    MC = 10000
    P = r._random_select(MC).values()
    epsilon = 1e-2
    for p, p_hat in zip(P, PROB):
        assert np.abs(p / 1e4 - p_hat) <= epsilon

    r.change_select_method(Select.EQUAL_FILE)
    MC = 10000
    P = r._random_select(MC).values()
    epsilon = 1e-2
    PROB = [.2, .2, .2, .2, .2]
    for p, p_hat in zip(P, PROB):
        assert np.abs(p / 1e4 - p_hat) <= epsilon


def test_quickloader_iter():
    DUT = DATASETS['DIV2K']
    DUT.setattr(patch_size=48, depth=1)
    r = QuickLoader(16, DUT, 'train', 4, 200)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)
    it = r.make_one_shot_iterator('8GB')
    for hr, lr, name in it:
        print(name, flush=True)


def test_mploader_iter():
    DUT = DATASETS['DIV2K']
    DUT.setattr(patch_size=48, depth=1)
    r = MpLoader(16, DUT, 'train', 4, 200)
    it = r.make_one_shot_iterator('8GB', 8)
    for hr, lr, name in it:
        print(name, flush=True)
    it = r.make_one_shot_iterator('8GB', 8)
    for hr, lr, name in it:
        print(name, flush=True)


def test_benchmark_quickloader():
    DUT = DATASETS['DIV2K']
    DUT.setattr(patch_size=196, depth=1, random=True, max_patches=100 * 8)
    EPOCHS = 4
    l = QuickLoader(8, DUT, 'train', 4, 100)
    for _ in range(EPOCHS):
        r = l.make_one_shot_iterator()
        for hr, lr, name in r:
            pass


def test_benchmark_mploader():
    DUT = DATASETS['DIV2K']
    DUT.setattr(patch_size=196, depth=1, random=True, max_patches=100 * 8)
    EPOCHS = 4
    l = MpLoader(8, DUT, 'train', 4, 100)
    for _ in range(EPOCHS):
        r = l.make_one_shot_iterator(shard=8)
        for hr, lr, name in r:
            pass


def test_read_flow():
    DUT = DATASETS['MINICHAIRS']
    DUT.setattr(patch_size=96, depth=2)
    l = MpLoader(8, DUT, 'train', 1, 100, convert_to='RGB', no_patch=True)
    r = l.make_one_shot_iterator('8GB', shard=8, shuffle=True)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)
    r = l.make_one_shot_iterator('8GB', shard=8, shuffle=True)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)
    img, flow, name = next(r)

    ref0 = img[0, 0, ...]
    ref1 = img[0, 1, ...]
    u = flow[0, 0, ..., 0]
    v = flow[0, 0, ..., 1]
    H, W = u.shape
    ImageProcess.array_to_img(ref0, 'RGB').show()
    ImageProcess.array_to_img(ref1, 'RGB').show()
    u = (u / W + 1) / 2 * 255
    v = (v / H + 1) / 2 * 255
    ImageProcess.array_to_img(u, 'L').show()
    ImageProcess.array_to_img(v, 'L').show()
