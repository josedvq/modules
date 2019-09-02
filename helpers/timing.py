from functools import wraps
from time import time
from datetime import timedelta

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:{:s} took: {:s}'.format(f.__name__, str(timedelta(seconds=te-ts))))
        return result
    return wrap

def get_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, timedelta(seconds=te-ts)
    return wrap