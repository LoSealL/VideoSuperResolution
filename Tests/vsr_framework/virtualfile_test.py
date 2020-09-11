"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-6

Unit test for DataLoader.VirtualFile
"""
import copy
import unittest

import numpy as np
from PIL import Image

from VSR.DataLoader.VirtualFile import (SEEK_CUR, SEEK_END, SEEK_SET,
                                        ImageFile, RawFile)
from VSR.Util.ImageProcess import img_to_array

RAW = 'data/video/raw_32x32.yv12'
IMG = 'data/set5_x2/img_001_SRF_2_LR.png'
VID = 'data/video/custom_pair/lr/xiuxian'


class VirtualFileTest(unittest.TestCase):
    def test_image_read(self):
        vf = ImageFile(IMG)
        self.assertEqual(vf.name, 'img_001_SRF_2_LR')
        img = vf.read_frame()
        self.assertIsInstance(img, list)
        self.assertEqual(len(img), 1)
        self.assertIsInstance(img[0], Image.Image)
        self.assertEqual(img[0].width, 256)
        self.assertEqual(img[0].height, 256)

    def test_video_read(self):
        vf = ImageFile(VID)
        self.assertEqual(vf.name, 'xiuxian')
        vid = vf.read_frame(3)
        self.assertIsInstance(vid, list)
        self.assertEqual(len(vid), 3)
        self.assertEqual(vid[0].width, 240)
        self.assertEqual(vid[0].height, 135)

    def test_raw_read(self):
        vf = RawFile(RAW, 'YV12', [32, 32])
        self.assertEqual(vf.name, 'raw_32x32')
        raw = vf.read_frame(vf.frames)
        self.assertEqual(len(raw), vf.frames)
        self.assertEqual(raw[0].width, 32)
        self.assertEqual(raw[0].height, 32)

    def test_image_seek(self):
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
        self.assertTrue(np.all(F[0] == F[1]))
        self.assertTrue(np.all(F[1] == F[2]))
        self.assertTrue(np.all(F[3] == F[4]))

    def test_vid_seek(self):
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
        self.assertTrue(np.all(F[0] == F[1]))
        self.assertTrue(np.all(F[1] == F[2]))
        self.assertTrue(np.all(F[3] == F[4]))

    def test_raw_seek(self):
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
        self.assertTrue(np.all(F[0] == F[1]))
        self.assertTrue(np.all(F[1] == F[2]))
        self.assertTrue(np.all(F[3] == F[4]))

    def test_vf_copy(self):
        vf0 = ImageFile(IMG, False)
        vf1 = copy.deepcopy(vf0)
        vf0.read_frame(1)
        try:
            vf0.read_frame(1)
            raise RuntimeError("Unreachable code")
        except EOFError:
            pass
        vf1.read_frame(1)
