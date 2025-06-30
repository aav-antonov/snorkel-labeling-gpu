import time
from functools import wraps


def time_clock(func):
    """Decorator that times function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High resolution timer
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        print(f"Function '{func.__name__}' executed in {elapsed:.6f} seconds")
        return result

    return wrapper