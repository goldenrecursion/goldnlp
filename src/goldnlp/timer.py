"""This module contains utilities to time and log timers for callables throughout the codebase"""

import asyncio
import json
import logging
from functools import wraps
from timeit import default_timer


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timer = default_timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs


def time_callable(func):
    """This decorator logs as an INFO message the execution time of the wrapped func.
    The timer also logs whether the function call exited normally or raised an exception."""

    @wraps(func)
    async def timed_func(*args, **kwargs):
        has_failed = False
        result = None
        try:
            with Timer() as t:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
        except:
            has_failed = True
            raise
        finally:
            log_message = {
                "type": "TimeCallable",
                "callable": f"{func.__module__}:{func.__qualname__}",
                "status": "OK" if not has_failed else "KO",
                "elapsed_secs": round(t.elapsed_secs, 4),
                "args": None if not has_failed else str(args),
                "kwargs": None if not has_failed else str(kwargs),
            }
            logging.getLogger(f"{func.__module__}").info(json.dumps(log_message))
            if not has_failed:
                return result

    return timed_func


def sync_time_callable(func):
    """This decorator logs as an INFO message the execution time of the wrapped func.
    The timer also logs whether the function call exited normally or raised an exception."""

    @wraps(func)
    def timed_func(*args, **kwargs):
        has_failed = False
        result = None
        try:
            with Timer() as t:
                result = func(*args, **kwargs)
        except:
            has_failed = True
            raise
        finally:
            log_message = {
                "type": "TimeCallable",
                "callable": f"{func.__module__}:{func.__qualname__}",
                "status": "OK" if not has_failed else "KO",
                "elapsed_secs": round(t.elapsed_secs, 4),
                "args": None if not has_failed else str(args),
                "kwargs": None if not has_failed else str(kwargs),
            }
            logging.getLogger(f"{func.__module__}").info(json.dumps(log_message))
            if not has_failed:
                return result

    return timed_func
