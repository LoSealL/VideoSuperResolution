"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-16

Test Dataset loading
"""
import unittest

from VSR.DataLoader.Dataset import Dataset, load_datasets


class DatasetTest(unittest.TestCase):
    def test_image_data(self):
        d = Dataset('data/set5_x2')
        data = d.compile()
        self.assertEqual(len(data), 5)
        self.assertEqual(data.capacity, 983040)

    def test_video_data(self):
        d = Dataset('data/video/custom_pair').use_like_video()
        data = d.compile()
        self.assertEqual(len(data), 2)

    def test_multi_url(self):
        d = Dataset('data/set5_x2', 'data/kitti_car')
        data = d.compile()
        self.assertEqual(len(data), 8)

    def test_include_exclude(self):
        d = Dataset('data')
        d.include_('xiuxian*')
        data1 = d.compile()
        d = Dataset('data')
        d.include_reg_('set5')
        data2 = d.compile()
        d = Dataset('data').include_reg('set5').exclude('png')
        data3 = d.compile()

        self.assertEqual(len(data1), 6)
        self.assertEqual(len(data2), 5)
        self.assertEqual(len(data3), 0)

    def test_dataset_desc_file(self):
        ddf = 'data/fake_datasets.yml'
        datasets = load_datasets(ddf)
        self.assertEqual(len(datasets), 9)
        self.assertEqual(len(datasets.NONE.train.hr.compile()), 0)
        self.assertEqual(len(datasets.NORMAL.train.hr.compile()), 7)
        self.assertEqual(len(datasets.NORMAL.val.hr.compile()), 5)
        self.assertEqual(len(datasets.NORMAL.test.hr.compile()), 1)
        self.assertEqual(len(datasets.PAIR.train.hr.compile()), 2)
        self.assertEqual(len(datasets.PAIR.train.lr.compile()), 2)
        self.assertEqual(len(datasets.VIDEOPAIR.train.hr.compile()), 1)
        self.assertEqual(len(datasets.VIDEOPAIR.train.lr.compile()), 1)
        self.assertEqual(len(datasets.VIDEOPAIR.val.hr.compile()), 1)
        self.assertEqual(len(datasets.VIDEOPAIR.val.lr.compile()), 1)
        self.assertEqual(len(datasets.VIDEOPAIR.test.hr.compile()), 1)
        self.assertEqual(len(datasets.VIDEOPAIR.test.lr.compile()), 1)
        self.assertEqual(len(datasets.FOO.test.hr.compile()), 2)
        self.assertEqual(len(datasets.BAR.test.hr.compile()), 5)
        self.assertTrue(datasets.VIDEOPAIR.train.hr.as_video)
        self.assertTrue(datasets.XIUXIAN.test.hr.as_video)

        raw = load_datasets(ddf, 'RAW')
        self.assertEqual(len(raw.train.hr.compile()), 1)
        self.assertEqual(len(raw.val.hr.compile()), 1)
        self.assertEqual(len(raw.test.hr.compile()), 1)
        self.assertTrue(raw.train.hr.as_video)
