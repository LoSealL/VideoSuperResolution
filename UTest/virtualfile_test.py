"""
Unit test for DataLoader.VirtualFile
"""

from VSR.DataLoader.VirtualFile import *
from VSR.DataLoader.Loader import *
from VSR.DataLoader.Dataset import *

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')

RAW = DATASETS['MCL-V'].test[0]
IMG = DATASETS['GOPRO'].train[0]


def test_raw_seek():
    vf = RawFile(RAW, 'YV12', [1920, 1080])
    f1 = vf.read_frame(1)[0]
    vf.seek(0, SEEK_SET)
    f2 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_CUR)
    f3 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_END)
    f4 = vf.read_frame(1)[0]
    vf.seek(-2, SEEK_END)
    vf.seek(1, SEEK_CUR)
    f5 = vf.read_frame(1)[0]

    F = [f1, f2, f3, f4, f5]
    F = [ImageProcess.img_to_array(f) for f in F]
    assert np.all(F[0] == F[1])
    assert np.all(F[1] == F[2])
    assert np.all(F[3] == F[4])


def test_image_seek():
    vf = ImageFile(IMG, False)
    f1 = vf.read_frame(1)[0]
    vf.seek(0, SEEK_SET)
    f2 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_CUR)
    f3 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_END)
    f4 = vf.read_frame(1)[0]
    vf.seek(-2, SEEK_END)
    vf.seek(1, SEEK_CUR)
    f5 = vf.read_frame(1)[0]

    F = [f1, f2, f3, f4, f5]
    F = [ImageProcess.img_to_array(f) for f in F]
    assert np.all(F[0] == F[1])
    assert np.all(F[1] == F[2])
    assert np.all(F[3] == F[4])
