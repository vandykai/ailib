import timeit
from functools import wraps

def timefn(fn):
    """计算性能的修饰器"""
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = timeit.default_timer()
        result = fn(*args, **kwargs)
        end = timeit.default_timer()
        print("@timefn：" + fn.__name__ + " took " + str(end - start) + " second")
        return result
    return measure_time