from functools import wraps
from time import time


def timing(f):
    """
    Taken from https://stackoverflow.com/a/27737385/8728749
    Parameters
    ----------
    f: function

    Returns
    -------

    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, (te - ts)

    return wrap
