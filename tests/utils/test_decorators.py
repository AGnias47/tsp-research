from utils.decorators import timing

import logging

logger = logging.getLogger("test_decorators")


@timing
def two_plus_two():
    return 2 + 2


def test_timing():
    result, time = two_plus_two()
    assert result == 4
    assert isinstance(time, float)
