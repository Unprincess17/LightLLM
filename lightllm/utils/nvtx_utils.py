"""
NVTX Annotation Utility for Hybrid CPU-GPU Pipeline Profiling.

Uses the nvtx Python package for NVIDIA Nsight Systems (nsys).

Usage:
    @NvtxAnnotate("name")
    def func(...):

    with NvtxAnnotate("name"):
        ...
"""

import hashlib
from collections.abc import Callable

import functools


# from nvtx import annotate  # type: ignore
from torch.cuda import nvtx as torch_nvtx  # type: ignore


# NVTX colors for consistent visual distinction in Nsight Systems traces
_NVTX_COLORS = [
    "green",
    "blue",
    "purple",
    "rapids",
    "orange",
    "yellow",
    "red",
]
_NUM_COLORS = len(_NVTX_COLORS)


def _get_color(name: str) -> str:
    """Get a consistent color from a range name using hash.

    Uses SHA256 for cross-run consistent hashing.
    """
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    return _NVTX_COLORS[hash_value % _NUM_COLORS]

class NvtxAnnotate:
    """
    Hybrid NVTX utility that works as both a decorator and a context manager.
    
    Usage 1: Context Manager
        with NvtxAnnotate("My Scope"):
            ...
            
    Usage 2: Decorator (auto-naming)
        @NvtxAnnotate
        def my_func(): ...
        
    Usage 3: Decorator (custom name)
        @NvtxAnnotate("My Label")
        def my_func(): ...
    """
    def __init__(self, msg_or_func=None):
        self.msg = None
        self.func = None
        self._is_wrapping_func = False

        if callable(msg_or_func):
            # Case: @NvtxAnnotate (no parentheses)
            # Used as: @NvtxAnnotate
            #          def func(): ...
            self.func = msg_or_func
            self.msg = msg_or_func.__qualname__
            self._is_wrapping_func = True
            functools.update_wrapper(self, msg_or_func)
        else:
            # Case: @NvtxAnnotate("msg") or with NvtxAnnotate("msg")
            self.msg = msg_or_func
            self.func = None
            self._is_wrapping_func = False

    def __enter__(self):
        # Support for 'with' statement
        if self.msg is None:
            self.msg = "NVTX Range"
        torch_nvtx.range_push(self.msg)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Support for 'with' statement
        torch_nvtx.range_pop()

    def __call__(self, *args, **kwargs):
        if self._is_wrapping_func:
            # Case 1: Act as the wrapper (executing the decorated function)
            torch_nvtx.range_push(self.msg)
            try:
                return self.func(*args, **kwargs)
            finally:
                torch_nvtx.range_pop()
        else:
            # Case 2: Act as the decorator factory (receiving the function to decorate)
            # This happens when using @NvtxAnnotate("msg")
            func = args[0]
            name = self.msg if self.msg else func.__qualname__
            
            @functools.wraps(func)
            def wrapper(*a, **kw):
                torch_nvtx.range_push(name)
                try:
                    return func(*a, **kw)
                finally:
                    torch_nvtx.range_pop()
            return wrapper

    def __get__(self, instance, owner):
        # Essential for supporting class methods when using @NvtxAnnotate (no parens)
        # It ensures 'self' is passed correctly to the method.
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)