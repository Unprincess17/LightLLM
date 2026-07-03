"""Pytest for gdr_sync.py — GPU flag-byte sync and cudaMemcpy helpers."""
import pytest
import torch
from rdma_qp_generator.gdr_sync import (
    copy_host_to_gpu,
    copy_gpu_to_host,
    copy_gpu_to_gpu,
    set_flag,
    poll_flag,
)

GPU_BYTES = 1 << 16  # 64 KB


@pytest.fixture
def gpu_buf():
    """Allocate a GPU buffer and return the tensor to keep it alive."""
    return torch.zeros(GPU_BYTES, dtype=torch.uint8, device="cuda")


@pytest.fixture
def host_buf():
    return torch.zeros(GPU_BYTES, dtype=torch.uint8)


def test_copy_host_to_gpu_roundtrip(gpu_buf, host_buf):
    host_buf[0] = 42
    host_buf[GPU_BYTES - 1] = 99
    copy_host_to_gpu(gpu_buf.data_ptr(), host_buf.data_ptr(), GPU_BYTES)
    result = torch.empty(GPU_BYTES, dtype=torch.uint8, device="cuda")
    copy_gpu_to_gpu(result.data_ptr(), gpu_buf.data_ptr(), GPU_BYTES)
    torch.cuda.synchronize()
    assert result[0].item() == 42
    assert result[GPU_BYTES - 1].item() == 99


def test_copy_gpu_to_host(gpu_buf, host_buf):
    src = torch.ones(GPU_BYTES, dtype=torch.uint8, device="cuda")
    src[10] = 77
    copy_gpu_to_host(host_buf.data_ptr(), src.data_ptr(), GPU_BYTES)
    torch.cuda.synchronize()
    assert host_buf[0].item() == 1
    assert host_buf[10].item() == 77


def test_copy_gpu_to_gpu(gpu_buf):
    src = torch.full((GPU_BYTES,), 55, dtype=torch.uint8, device="cuda")
    copy_gpu_to_gpu(gpu_buf.data_ptr(), src.data_ptr(), GPU_BYTES)
    torch.cuda.synchronize()
    result = torch.empty(GPU_BYTES, dtype=torch.uint8, device="cuda")
    copy_gpu_to_gpu(result.data_ptr(), gpu_buf.data_ptr(), GPU_BYTES)
    torch.cuda.synchronize()
    assert result[0].item() == 55


def test_set_flag_and_poll(gpu_buf):
    flag_offset = GPU_BYTES - 1
    set_flag(gpu_buf.data_ptr(), flag_offset, 0)
    import time, threading

    def set_later():
        import time
        time.sleep(0.05)
        set_flag(gpu_buf.data_ptr(), flag_offset, 1)

    t0 = time.perf_counter()
    t = threading.Thread(target=set_later, daemon=True)
    t.start()
    poll_flag(gpu_buf.data_ptr(), flag_offset, 1, timeout_s=2.0)
    elapsed = time.perf_counter() - t0
    t.join()
    assert 0.04 < elapsed < 1.0


def test_poll_flag_timeout(gpu_buf):
    with pytest.raises(TimeoutError, match="flag at offset"):
        poll_flag(gpu_buf.data_ptr(), GPU_BYTES - 1, 99, timeout_s=0.1)
