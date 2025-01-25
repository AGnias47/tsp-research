"""
Directly imported from https://stackoverflow.com/a/17954769/8728749 with added code to redirect stderr
"""

import os
import sys
from contextlib import contextmanager


@contextmanager
def redirect_output_to_null(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()
    fde = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fde)
        sys.stderr = os.fdopen(fde, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with os.fdopen(os.dup(fde), "w") as old_stderr:
            with open(to, "w") as file:
                _redirect_stdout(to=file)
                _redirect_stderr(to=file)
            try:
                yield  # allow code to be run with the redirected stdout
            finally:
                _redirect_stdout(to=old_stdout)  # restore stdout.
                _redirect_stderr(to=old_stderr)
                # buffering and flags such as
                # CLOEXEC may be different
