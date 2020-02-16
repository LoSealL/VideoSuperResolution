import os
import unittest

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')

from VSR.Util import str_to_bytes, Config

TEST_STR = (
  '1.3', '2kb', '3 mb', '4GB', '9Zb', '2.3pB'
)
ANS = (
  1.3, 2048.0, 3145728.0, 4294967296.0, 10625324586456701730816.0,
  2589569785738035.2
)


class UtilityTest(unittest.TestCase):
  def test_str_to_bytes(self):
    for t, a in zip(TEST_STR, ANS):
      ans = str_to_bytes(t)
      self.assertEqual(ans, a, f"{t} != {a}")

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


if __name__ == '__main__':
  unittest.main()
