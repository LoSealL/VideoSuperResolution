from VSR.Util import Utility as U

TEST_STR = ('1.3', '2kb', '3 mb', '4GB', '9Zb', '2.3pB')
ANS = (1.3, 2048.0, 3145728.0, 4294967296.0, 10625324586456701730816.0, 2589569785738035.2)


def test_str_to_bytes():
    for t, a in zip(TEST_STR, ANS):
        ans = U.str_to_bytes(t)
        print(t, ans)
        assert ans == a


import multiprocessing as mp


class foo:
    def __init__(self):
        self.mp = None
        self.a = 2

    def _hey(self, a, b):
        print(a, b, self.a)
        return self.a + a + b

    def bar(self):
        self.mp = mp.Process(target=self._hey, args=(1, 2))
        self.mp.start()


def test_foo():
    x = foo()
    x.bar()
    x.mp.join()


if __name__ == '__main__':
    test_foo()
