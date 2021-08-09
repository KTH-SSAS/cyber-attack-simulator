import numpy as np


def _bits(i, size=None):
    if size is None:
        while i:
            yield i & 1
            i >>= 1
    else:
        for _ in range(size):
            yield i & 1
            i >>= 1


def bits(int_value, size=None):
    return list(_bits(int_value, size))


def np_bits(int_value, size=None):
    return np.array(bits(int_value, size))


def test_bits():
    assert bits(42) == [0, 1, 0, 1, 0, 1]
    assert bits(42, size=3) == [0, 1, 0]
    assert bits(42, size=8) == [0, 1, 0, 1, 0, 1, 0, 0]


def test_np_bits():
    i = np.random.randint(42)
    sizes = np.random.randint(16, size=3)
    for size in (None, *sizes):
        assert all(np_bits(i) == bits(i))
