import timeit
from functools import wraps
import time

def timefn(fn):
    """计算性能的修饰器
    note: 修饰器
    Args:
        fn (int): seconds.
        n:precision (D,h,m,s,ms)
    Returns:
        print: print time cost.
    Example:
        @timefn
        def f():
            sum = 0
            for i in range(100000):
                sum += i
            return sum
        f()
        #output
        print:@timefn：f took 0.012037459760904312 second
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = timeit.default_timer()
        result = fn(*args, **kwargs)
        end = timeit.default_timer()
        print("@timefn：" + fn.__name__ + " took " + str(end - start) + " second")
        return result
    return measure_time

def format_time(seconds, n=5):
    """Format seconds to std time.
    note:
    Args:
        seconds (int): seconds.
        n:precision (D,h,m,s,ms)
    Returns:
        str: .
    Example:
        seconds = 123456.7
        format_time(seconds)
        #output
        1D10h17m36s700ms
        format_time(seconds, n=2)
        #output
        1D10h
    """
    days = int(seconds / 3600/ 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = round(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= n:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= n:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= n:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= n:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        """Timer constructor."""
        self.reset()

    def reset(self):
        """Reset timer."""
        self.running = True
        self.total = 0
        self.start = time.time()

    def resume(self):
        """Resume."""
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """Stop."""
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    @property
    def time(self):
        """Return time."""
        if self.running:
            return self.total + time.time() - self.start
        return self.total