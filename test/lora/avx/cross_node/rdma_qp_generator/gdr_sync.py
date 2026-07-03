"""GPU flag-byte synchronization and cudaMemcpy helpers.

Uses ctypes wrappers around libcudart.so for cudaMemcpy, cudaMemset.
Avoids cuda-python dependency (broken on some kernels).
"""
import ctypes
import time

_CUDART = ctypes.CDLL("libcudart.so")

# cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
_CUDART.cudaMemcpy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
]
_CUDART.cudaMemcpy.restype = ctypes.c_int

# cudaError_t cudaMemset(void *devPtr, int value, size_t count)
_CUDART.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_CUDART.cudaMemset.restype = ctypes.c_int

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


def _check(err: int, msg: str) -> None:
    if err != 0:
        raise RuntimeError(f"{msg}: cudaError {err}")


def copy_host_to_gpu(gpu_ptr: int, host_ptr: int, nbytes: int) -> None:
    err = _CUDART.cudaMemcpy(
        ctypes.c_void_p(gpu_ptr),
        ctypes.c_void_p(host_ptr),
        nbytes,
        _CUDA_MEMCPY_HOST_TO_DEVICE,
    )
    _check(err, "cudaMemcpy H2D")


def copy_gpu_to_host(host_ptr: int, gpu_ptr: int, nbytes: int) -> None:
    err = _CUDART.cudaMemcpy(
        ctypes.c_void_p(host_ptr),
        ctypes.c_void_p(gpu_ptr),
        nbytes,
        _CUDA_MEMCPY_DEVICE_TO_HOST,
    )
    _check(err, "cudaMemcpy D2H")


def copy_gpu_to_gpu(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    err = _CUDART.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        nbytes,
        _CUDA_MEMCPY_DEVICE_TO_DEVICE,
    )
    _check(err, "cudaMemcpy D2D")


def set_flag(gpu_ptr: int, offset: int, value: int) -> None:
    err = _CUDART.cudaMemset(
        ctypes.c_void_p(gpu_ptr + offset), value, 1,
    )
    _check(err, "cudaMemset set_flag")
    # Flush GPU L2 cache so RDMA WRITEs from remote are visible to
    # subsequent poll_flag reads. Without this, cudaMemset caches the
    # value in L2, and the cudaMemcpy D2H in poll_flag reads the stale
    # cached value instead of the RDMA-written value in VRAM.
    _CUDART.cudaDeviceSynchronize()


def poll_flag(
    gpu_ptr: int, offset: int, expected: int, timeout_s: float = 10.0
) -> None:
    """Busy-poll a single byte at gpu_ptr+offset until it equals expected.

    Uses cudaMemcpy D2H to read one byte per iteration. 1 us yield between polls.
    """
    buf = (ctypes.c_uint8 * 1)()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        err = _CUDART.cudaMemcpy(
            ctypes.c_void_p(ctypes.addressof(buf)),
            ctypes.c_void_p(gpu_ptr + offset),
            1,
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy D2H in poll_flag: cudaError {err}")
        if buf[0] == expected:
            return
        time.sleep(0.000001)  # 1 us backoff
    raise TimeoutError(
        f"poll_flag: flag at offset {offset} did not become {expected} "
        f"within {timeout_s}s (last value={buf[0]})"
    )
