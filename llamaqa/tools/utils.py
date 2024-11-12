from typing import Callable, Dict
import functools


def output_descriptor(desc: str):
    def decorator_add_descriptor(func: Callable):
        func.__output_desc__ = desc
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator_add_descriptor


def tool_metadata(desc: str, default_kwargs: Dict = {}):
    def decorator_add_metadata(func: Callable):
        func.__output_desc__ = desc
        func.__default_kwargs__ = default_kwargs
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator_add_metadata
