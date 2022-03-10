import ctypes

from ._c_api import AML_BUFFER_SIZE_ERROR
from .status import AmlError, _save_exception


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except AmlError as e:
            if e.status == AML_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def catch_exceptions(function):
    def inner(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as e:
            _save_exception(e)
            return -1
        return 0

    return inner