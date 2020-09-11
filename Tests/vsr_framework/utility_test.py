"""
Copyright: Wenyi Tang 2020
Author: Wenyi Tang
Email: wenyitang@outlook.com
Created Date: 2020-9-11

Test string/int to bytes description
"""
import unittest

from VSR.Util import Config, str_to_bytes


class UtilityTest(unittest.TestCase):
    def test_str_to_bytes(self):
        _test_str = {
            '1.3': 1.3,
            '2kb': 2048,
            '3 mb': 3145728,
            '4GB': 4294967296,
            '9Zb': 10625324586456701730816,
            '2.3pB': 2589569785738035.2
        }
        for t, a in _test_str.items():
            ans = str_to_bytes(t)
            self.assertAlmostEqual(ans, a, f"{t} != {a}")

    def test_config(self):
        d = Config(a=1, b=2)
        self.assertTrue(hasattr(d, 'a'))
        self.assertTrue(hasattr(d, 'b'))
        self.assertTrue(hasattr(d, 'non-exist'))
        self.assertIs(d.a, 1)
        self.assertIs(d.b, 2)
        d.update(a=2, b=3)
        self.assertIs(d.a, 2)
        self.assertIs(d.b, 3)
        d.a = 9
        self.assertIs(d.a, 9)
        d.update(Config(b=6, f=5))
        self.assertIs(d.b, 6)
        self.assertIs(d.f, 5)
        d.pop('b')
        self.assertIsNone(d.b)
