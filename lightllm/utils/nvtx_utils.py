"""
NVTX annotation utility used by profiling paths.

This module keeps the default `torch.cuda.nvtx` behavior, and enables
colorized ranges when the CUDA NVTX C API (`libnvToolsExt`) is available.
"""

import ctypes
import ctypes.util
import functools
import hashlib
import threading
from collections.abc import Callable

from torch.cuda import nvtx as torch_nvtx  # type: ignore


# NVTX colors for consistent visual distinction in Nsight Systems traces.
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


# ARGB colors used by nvtxRangePushEx.
_COLOR_NAME_TO_ARGB = {
    "green": 0xFF2ECC71,
    "blue": 0xFF3498DB,
    "purple": 0xFF9B59B6,
    "rapids": 0xFF76B900,  # NVIDIA RAPIDS/NVIDIA green
    "orange": 0xFFE67E22,
    "yellow": 0xFFF1C40F,
    "red": 0xFFE74C3C,
}


class _NvtxPayloadUnion(ctypes.Union):
    _fields_ = [
        ("ullValue", ctypes.c_uint64),
        ("llValue", ctypes.c_int64),
        ("dValue", ctypes.c_double),
        ("uiValue", ctypes.c_uint32),
        ("iValue", ctypes.c_int32),
        ("fValue", ctypes.c_float),
    ]


class _NvtxMessageUnion(ctypes.Union):
    _fields_ = [
        ("ascii", ctypes.c_char_p),
        ("unicode", ctypes.c_wchar_p),
        ("registered", ctypes.c_void_p),
    ]


class _NvtxEventAttributesV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("size", ctypes.c_uint16),
        ("category", ctypes.c_uint32),
        ("colorType", ctypes.c_int32),
        ("color", ctypes.c_uint32),
        ("payloadType", ctypes.c_int32),
        ("payload", _NvtxPayloadUnion),
        ("messageType", ctypes.c_int32),
        ("message", _NvtxMessageUnion),
    ]


_NVTX_VERSION = 2
_NVTX_COLOR_ARGB = 1
_NVTX_MESSAGE_TYPE_ASCII = 1

_thread_local = threading.local()


def _get_color(name: str) -> str:
    """Get a deterministic color name from a range name via SHA256 hash."""
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    return _NVTX_COLORS[hash_value % _NUM_COLORS]


def _get_backend_stack() -> list[str]:
    stack = getattr(_thread_local, "nvtx_backend_stack", None)
    if stack is None:
        stack = []
        _thread_local.nvtx_backend_stack = stack
    return stack


def _parse_color_to_argb(color: str | int | None) -> int | None:
    if color is None:
        return None

    if isinstance(color, int):
        if 0 <= color <= 0xFFFFFFFF:
            return color
        return None

    lowered = color.strip().lower()
    named = _COLOR_NAME_TO_ARGB.get(lowered)
    if named is not None:
        return named

    if lowered.startswith("#"):
        hex_part = lowered[1:]
        if len(hex_part) == 6:
            try:
                return int(f"ff{hex_part}", 16)
            except ValueError:
                return None
        if len(hex_part) == 8:
            try:
                return int(hex_part, 16)
            except ValueError:
                return None
        return None

    if lowered.startswith("0x"):
        try:
            value = int(lowered, 16)
            if 0 <= value <= 0xFFFFFFFF:
                return value
        except ValueError:
            return None
    return None


def _load_nvtx_ext() -> ctypes.CDLL | None:
    candidates = []
    resolved = ctypes.util.find_library("nvToolsExt")
    if resolved:
        candidates.append(resolved)
    candidates.extend(["libnvToolsExt.so.1", "libnvToolsExt.so"])

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


_nvtx_ext = _load_nvtx_ext()
if _nvtx_ext is not None:
    try:
        _nvtx_range_push_ex = _nvtx_ext.nvtxRangePushEx
        _nvtx_range_push_ex.argtypes = [ctypes.POINTER(_NvtxEventAttributesV2)]
        _nvtx_range_push_ex.restype = ctypes.c_int

        _nvtx_range_pop = _nvtx_ext.nvtxRangePop
        _nvtx_range_pop.argtypes = []
        _nvtx_range_pop.restype = ctypes.c_int
    except AttributeError:
        _nvtx_range_push_ex = None
        _nvtx_range_pop = None
else:
    _nvtx_range_push_ex = None
    _nvtx_range_pop = None


def is_nvtx_color_available() -> bool:
    """Return True when CUDA NVTX color backend is available."""
    return _nvtx_range_push_ex is not None and _nvtx_range_pop is not None


def get_nvtx_backend() -> str:
    """Return active NVTX backend type for diagnostics."""
    if is_nvtx_color_available():
        return "cuda_nvtx_range_push_ex_with_color"
    return "torch_cuda_nvtx_range_push_without_color"


def _range_push(msg: str, color: str | int | None = None) -> None:
    color_argb = _parse_color_to_argb(color)
    if is_nvtx_color_available() and color_argb is not None:
        attrs = _NvtxEventAttributesV2()
        attrs.version = _NVTX_VERSION
        attrs.size = ctypes.sizeof(_NvtxEventAttributesV2)
        attrs.category = 0
        attrs.colorType = _NVTX_COLOR_ARGB
        attrs.color = color_argb
        attrs.payloadType = 0
        attrs.messageType = _NVTX_MESSAGE_TYPE_ASCII
        msg_bytes = msg.encode("utf-8", errors="replace")
        attrs.message.ascii = ctypes.c_char_p(msg_bytes)
        _nvtx_range_push_ex(ctypes.byref(attrs))
        _get_backend_stack().append("cuda_color")
        return

    torch_nvtx.range_push(msg)
    _get_backend_stack().append("torch")


def _range_pop() -> None:
    stack = _get_backend_stack()
    backend = stack.pop() if stack else "torch"
    if backend == "cuda_color" and _nvtx_range_pop is not None:
        _nvtx_range_pop()
        return
    torch_nvtx.range_pop()


class NvtxAnnotate:
    """
    NVTX utility that works as both a decorator and a context manager.

    Usage:
        with NvtxAnnotate("My Scope"):
            ...

        @NvtxAnnotate
        def my_func(): ...

        @NvtxAnnotate("My Label")
        def my_func(): ...

        with NvtxAnnotate("My Colored Scope", color="red"):
            ...
    """

    def __init__(
        self,
        msg_or_func: str | Callable | None = None,
        color: str | int | None = None,
    ):
        self.msg = None
        self.func = None
        self.color = color
        self._is_wrapping_func = False

        if callable(msg_or_func):
            # Case: @NvtxAnnotate (no parentheses)
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
        if self.msg is None:
            self.msg = "NVTX Range"
        resolved_color = self.color if self.color is not None else _get_color(self.msg)
        _range_push(self.msg, resolved_color)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _range_pop()

    def __call__(self, *args, **kwargs):
        if self._is_wrapping_func:
            # Case 1: Act as the wrapper (executing the decorated function)
            resolved_color = self.color if self.color is not None else _get_color(self.msg)
            _range_push(self.msg, resolved_color)
            try:
                return self.func(*args, **kwargs)
            finally:
                _range_pop()

        # Case 2: Act as decorator factory (@NvtxAnnotate("msg") / @NvtxAnnotate(color="..."))
        func = args[0]
        name = self.msg if self.msg else func.__qualname__
        resolved_color = self.color if self.color is not None else _get_color(name)

        @functools.wraps(func)
        def wrapper(*a, **kw):
            _range_push(name, resolved_color)
            try:
                return func(*a, **kw)
            finally:
                _range_pop()

        return wrapper

    def __get__(self, instance, owner):
        # Supports class methods when using @NvtxAnnotate (without parentheses).
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)
