from VSR.Util import Utility as U
from VSR.Util.Config import Config

TEST_STR = ('1.3', '2kb', '3 mb', '4GB', '9Zb', '2.3pB')
ANS = (1.3, 2048.0, 3145728.0, 4294967296.0, 10625324586456701730816.0,
       2589569785738035.2)


def test_str_to_bytes():
    for t, a in zip(TEST_STR, ANS):
        ans = U.str_to_bytes(t)
        print(t, ans)
        assert ans == a


def test_config():
    d = Config(a=1, b=2)
    d.update(a=2, b=3)
    d.a = 9
    d.update(Config(b=6, f=5))
    d.pop('b')
    print(d)
