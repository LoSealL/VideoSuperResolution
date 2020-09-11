"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-2-5

Test srcnn and vespcn train/eval
"""
import unittest

from VSR.Backend import DATA_FORMAT
from VSR.DataLoader import Dataset, Loader
from VSR.Model import get_model


class ModelTest(unittest.TestCase):
    def test_train_srcnn(self):
        data = Dataset('data').include_reg('set5')
        ld = Loader(data, scale=2)
        ld.set_color_space('lr', 'L')
        ld.set_color_space('hr', 'L')
        m = get_model('srcnn')(scale=2, channel=1)
        with m.executor as t:
            config = t.query_config({})
            config.epochs = 5
            config.steps = 10
            if DATA_FORMAT == 'channels_first':
                config.batch_shape = [16, 1, 16, 16]
            else:
                config.batch_shape = [16, 16, 16, 1]
            t.fit([ld, None], config)

    def test_infer_srcnn(self):
        m = get_model('srcnn')(scale=2, channel=3)
        data = Dataset('data').include_reg('set5')
        ld = Loader(data, scale=2)
        with m.executor as t:
            config = t.query_config({})
            t.infer(ld, config)

    def test_train_vespcn(self):
        data = Dataset('data/video').include_reg("xiuxian").use_like_video()
        ld = Loader(data, scale=2)
        m = get_model('vespcn')(scale=2, channel=3)
        with m.executor as t:
            config = t.query_config({})
            config.epochs = 1
            config.steps = 10
            if DATA_FORMAT == 'channels_first':
                config.batch_shape = [16, 3, 3, 16, 16]
            else:
                config.batch_shape = [16, 3, 16, 16, 3]
            t.fit([ld, None], config)
