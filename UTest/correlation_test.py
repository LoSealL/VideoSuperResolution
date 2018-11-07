import tensorflow as tf
from VSR.Util.Utility import _make_vector, _make_displacement, correlation


def test_make_vector():
    x = tf.constant([[
        [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]],
        [[4, 4.1, 4.2], [5, 5.1, 5.2], [6, 6.1, 6.2]],
        [[7, 7.1, 7.2], [8, 8.1, 8.2], [9, 9.1, 9.2]]
    ]], 'float32')
    y = _make_vector(x, 3)
    return y


def test_make_displacement():
    x = tf.constant([[
        [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]],
        [[4, 4.1, 4.2], [5, 5.1, 5.2], [6, 6.1, 6.2]],
        [[7, 7.1, 7.2], [8, 8.1, 8.2], [9, 9.1, 9.2]]
    ]], 'float32')
    y = _make_displacement(x, 1)
    return y


def test_correlation():
    x = tf.constant([[
        [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]],
        [[4, 4.1, 4.2], [5, 5.1, 5.2], [6, 6.1, 6.2]],
        [[7, 7.1, 7.2], [8, 8.1, 8.2], [9, 9.1, 9.2]]
    ]], 'float32')
    return correlation(x, x, 3, 1)


def test_correlation_stride():
    x = tf.ones([1, 5, 5, 1], 'float32')
    return correlation(x, x, 3, 2, 2, 2)


if __name__ == '__main__':
    tf.enable_eager_execution()
    test_correlation_stride()
