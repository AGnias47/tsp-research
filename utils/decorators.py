from functools import wraps
from time import time


def timing(f):
    """
    Times a function. Returns the function result as a tuple of (result, time).

    References
    ----------
    Taken from https://stackoverflow.com/a/27737385/8728749 and pulled again from
    https://github.com/AGnias47/brats-challenge-cis-5528/blob/main/nn/nnet.py

    Parameters
    ----------
    f: function

    Returns
    -------
    function result, time
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, (te - ts)

    return wrap
