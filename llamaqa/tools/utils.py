from typing import Callable
import functools


def output_descriptor(desc: str):
    def decorator_add_descriptor(func: Callable):
        func.__output_desc__ = desc
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator_add_descriptor
