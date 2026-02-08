"""
NVTX Annotation Utility for Hybrid CPU-GPU Pipeline Profiling.

Provides a wrapper around torch.cuda.nvtx.range_push/range_pop
for use with NVIDIA Nsight Systems (nsys).

Colors are automatically assigned using hash(name) % 7 for consistent
coloring across different range names.
"""

from functools import wraps
from contextlib import ContextDecorator
from typing import Optional

import torch

# NVTX reserved color IDs: 0=red, 1=orange, 2=yellow, 3=green, 4=blue, 5=purple, 6=white
_NUM_COLORS = 7


class NvtxAnnotate(ContextDecorator):
    """
    NVTX range annotation utility supporting both context manager and decorator usage.

    Colors are automatically assigned using hash(name) % 7 for consistent
    visual distinction in Nsight Systems traces.

    Args:
        name: Range name. If None, uses function's __name__ (decorator mode).
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name

    @staticmethod
    def get_color_id(name: str) -> int:
        """Get a consistent color ID from a range name using hash."""
        return hash(name) % _NUM_COLORS

    def __enter__(self):
        """Enter context manager: push NVTX range."""
        name = self.name
        if name is None:
            raise ValueError(
                "NvtxAnnotate requires a name when used as a context manager. "
                "Use: NvtxAnnotate('YourName') or @NvtxAnnotate('YourName')"
            )
        torch.cuda.nvtx.range_push(name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager: pop NVTX range (always called, even on exception)."""
        torch.cuda.nvtx.range_pop()
        return False  # Don't suppress exceptions

    def __call__(self, func):
        """Decorator mode: wrap function with NVTX ranges."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name if no explicit name provided
            range_name = self.name if self.name else func.__name__
            torch.cuda.nvtx.range_push(range_name)
            try:
                return func(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        return wrapper


# Backwards compatibility alias
NvtxScope = NvtxAnnotate
