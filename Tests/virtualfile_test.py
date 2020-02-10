"""
Unit test for DataLoader.VirtualFile
"""
import os

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')
from VSR.DataLoader.VirtualFile import *
from VSR.Util.ImageProcess import img_to_array

RAW = 'data/video/raw_32x32.yv12'
IMG = 'data/set5_x2/img_001_SRF_2_LR.png'
VID = 'data/video/custom_pair/lr/xiuxian'


def test_image_read():
  vf = ImageFile(IMG)
  assert vf.name == 'img_001_SRF_2_LR'
  img = vf.read_frame()
  assert isinstance(img, list)
  assert len(img) is 1
  assert isinstance(img[0], Image.Image)
  assert img[0].width is 256
  assert img[0].height is 256


def test_video_read():
  vf = ImageFile(VID)
  assert vf.name == 'xiuxian'
  vid = vf.read_frame(3)
  assert isinstance(vid, list)
  assert len(vid) is 3
  assert vid[0].width is 240
  assert vid[0].height is 135


def test_raw_read():
  vf = RawFile(RAW, 'YV12', [32, 32])
  assert vf.name == 'raw_32x32'
  raw = vf.read_frame(vf.frames)
  assert len(raw) == vf.frames
  assert raw[0].width is 32
  assert raw[0].height is 32


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
  f5 = vf.read_frame(1)[0]

  F = [f1, f2, f3, f4, f5]
  F = [img_to_array(f) for f in F]
  assert np.all(F[0] == F[1])
  assert np.all(F[1] == F[2])
  assert np.all(F[3] == F[4])


def test_vid_seek():
  vf = ImageFile(VID, False)
  f1 = vf.read_frame(1)[0]
  vf.seek(0, SEEK_SET)
  f2 = vf.read_frame(1)[0]
  vf.seek(-1, SEEK_CUR)
  f3 = vf.read_frame(1)[0]
  vf.seek(-1, SEEK_END)
  f4 = vf.read_frame(1)[0]
  vf.seek(2, SEEK_SET)
  f5 = vf.read_frame(1)[0]

  F = [f1, f2, f3, f4, f5]
  F = [img_to_array(f) for f in F]
  assert np.all(F[0] == F[1])
  assert np.all(F[1] == F[2])
  assert np.all(F[3] == F[4])


def test_raw_seek():
  vf = RawFile(RAW, 'YV12', [32, 32])
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
  F = [img_to_array(f) for f in F]
  assert np.all(F[0] == F[1])
  assert np.all(F[1] == F[2])
  assert np.all(F[3] == F[4])


def test_vf_copy():
  import copy
  vf0 = ImageFile(IMG, False)
  vf1 = copy.deepcopy(vf0)
  vf0.read_frame(1)
  try:
    vf0.read_frame(1)
    raise RuntimeError("Unreached code")
  except EOFError:
    pass
  vf1.read_frame(1)
