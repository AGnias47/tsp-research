import logging

from utils.decorators import timing

logger = logging.getLogger("test_decorators")


@timing
def two_plus_two():
    return 2 + 2


@timing
def two_return_vals():
    return 2, 4


def test_timing():
    result, time = two_plus_two()
    assert result == 4
    assert isinstance(time, float)
    (two, four), time2 = two_return_vals()
    assert two == 2
    assert four == 4
    assert isinstance(time2, float)
