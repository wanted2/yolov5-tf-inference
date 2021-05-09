import time
from functools import wraps
import logging
import sys
import os
import threading


def setup_logger(name: str, default_level:int=logging.DEBUG):
    logger = logging.getLogger(name)
    fmt = '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)s] %(message)s'
    formatter = logging.Formatter(fmt)
    handler = logging.StreamHandler(stream=None)
    handler.setFormatter(formatter)
    handler.setLevel(default_level)
    fileHanler = logging.FileHandler(filename=f'{name}.log',
                                     mode='a+', encoding='utf-8')
    fileHanler.setFormatter(formatter)
    fileHanler.setLevel(default_level)
    logger.addHandler(fileHanler)
    logger.addHandler(handler)
    logger.setLevel(level=default_level)
    return logger


LOGGER = setup_logger(__name__)


class Timer:
    @staticmethod
    def measure_runtime(f):
        @wraps(f)
        def func(*args, **kwargs):
            start = time.time()
            results = f(*args, **kwargs)
            end = time.time()
            LOGGER.info(
                f"Execution time of function {f.__name__}: {end-start:.5f} [s]")
            return results, end - start
        return func


class Paralellism:
    @staticmethod
    def threadize(f):
        @wraps(f)
        def func(*args, **kwargs):
            t = threading.Thread(target=f, args=args, kwargs=kwargs, daemon=True)
            t.start()
        return func
