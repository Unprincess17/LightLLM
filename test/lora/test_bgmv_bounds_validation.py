"""Unit tests for BGMV host-side bounds validation (``LIGHTLLM_BGMV_DEBUG_BOUNDS`` helpers)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightllm._kernels.lora.bgmv import (
    _coerce_req_bins_for_bgmv,
    validate_bgmv_dispatch_inputs,
)


def _cpu_meta(n_adapters: int, pool_size: int, layers: int = 4):
    """Minimal valid metadata for ``validate_bgmv_dispatch_inputs``."""
    # Contiguous per-adapter blocks of size ``layers`` (matches fixed reservation).
    starts = torch.arange(0, n_adapters * layers, layers, dtype=torch.int32)
    a_len = torch.full((n_adapters,), layers, dtype=torch.int32)
    scaling = torch.ones(n_adapters, dtype=torch.float32)
    return starts, a_len, scaling


def test_coerce_req_bins_contiguous_int32():
    # Column slice of a 2-D tensor is typically non-contiguous (stride > 1).
    m = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    bins = m[:, 1]
    assert bins.shape == (3,) and not bins.is_contiguous()
    c = _coerce_req_bins_for_bgmv(bins, 3)
    assert c.is_contiguous()
    assert c.dtype == torch.int32
    assert torch.equal(c, torch.tensor([1, 5, 9], dtype=torch.int32))


def test_coerce_req_bins_too_short():
    with pytest.raises(ValueError, match="length"):
        _coerce_req_bins_for_bgmv(torch.tensor([0], dtype=torch.int32), 3)


def test_validate_passes_16_and_32_adapters():
    for n in (16, 32):
        pool_size = n * 4 + 8
        h_in, h_out, max_rank = 64, 64, 8
        batch = 5
        a_buf = torch.zeros(pool_size, max_rank, h_in)
        b_buf = torch.zeros(pool_size, max_rank, h_out)
        x = torch.zeros(batch, h_in)
        y = torch.zeros(batch, h_out)
        starts, a_len, scaling = _cpu_meta(n, pool_size, layers=4)
        req = torch.tensor([0, n - 1, 1, 0, 2], dtype=torch.int32)
        a_rank = torch.full((n,), 8, dtype=torch.int32)
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
            a_rank=a_rank,
        )


def test_validate_req_bins_out_of_range():
    n = 4
    pool_size = 32
    h_in = h_out = max_rank = 16
    batch = 2
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    req = torch.tensor([0, 9], dtype=torch.int32)
    with pytest.raises(AssertionError, match="out of range"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
        )


def test_validate_negative_req_bins():
    n = 2
    pool_size = 16
    h_in = h_out = max_rank = 8
    batch = 2
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    req = torch.tensor([-1, 0], dtype=torch.int32)
    with pytest.raises(AssertionError, match="negative"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
        )


def test_validate_noncontiguous_req_bins():
    n = 2
    pool_size = 16
    h_in = h_out = max_rank = 8
    batch = 3
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    m = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    req = m[:3, 1]
    assert req.shape == (3,) and not req.is_contiguous()
    with pytest.raises(AssertionError, match="contiguous"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
        )


def test_validate_wrong_req_bins_dtype():
    n = 2
    pool_size = 16
    h_in = h_out = max_rank = 8
    batch = 2
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    req = torch.tensor([0.0, 1.0])
    with pytest.raises(AssertionError, match="int32/int64"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
        )


def test_validate_slot_out_of_pool():
    n = 2
    pool_size = 5  # deliberately too small for layer_id=3
    h_in = h_out = max_rank = 8
    batch = 1
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    req = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(AssertionError, match="out of pool"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=5,
        )


def test_validate_zero_a_len_selected():
    n = 3
    pool_size = 32
    h_in = h_out = max_rank = 8
    batch = 1
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    a_len = a_len.clone()
    a_len[1] = 0
    req = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(AssertionError, match="a_len"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
        )


def test_validate_bad_a_rank_optional():
    n = 3
    pool_size = 32
    h_in = h_out = max_rank = 8
    batch = 1
    a_buf = torch.zeros(pool_size, max_rank, h_in)
    b_buf = torch.zeros(pool_size, max_rank, h_out)
    x = torch.zeros(batch, h_in)
    y = torch.zeros(batch, h_out)
    starts, a_len, scaling = _cpu_meta(n, pool_size)
    req = torch.tensor([1], dtype=torch.int32)
    a_rank = torch.tensor([8, 0, 8], dtype=torch.int32)
    with pytest.raises(AssertionError, match="a_rank"):
        validate_bgmv_dispatch_inputs(
            projection="test",
            y=y,
            x=x,
            a_buffer=a_buf,
            b_buffer=b_buf,
            a_start=starts,
            a_len=a_len,
            a_scaling=scaling,
            req_bins=req,
            h_in=h_in,
            h_out=h_out,
            max_rank=max_rank,
            pool_size=pool_size,
            layer_id=0,
            a_rank=a_rank,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_bgmv_negative_bins_raises_before_kernel():
    from lightllm._kernels.lora.bgmv import dispatch_bgmv

    device = "cuda"
    batch, h_in, h_out, max_rank, pool_size, n_adapters = 2, 32, 32, 8, 64, 2
    x = torch.randn(batch, h_in, device=device)
    y = torch.randn(batch, h_out, device=device)
    a_buf = torch.zeros(pool_size, max_rank, h_in, device=device)
    b_buf = torch.zeros(pool_size, max_rank, h_out, device=device)
    starts = torch.tensor([0, 16], dtype=torch.int32, device=device)
    a_len = torch.tensor([16, 16], dtype=torch.int32, device=device)
    scaling = torch.ones(n_adapters, device=device)
    req = torch.tensor([-1, 0], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="negative"):
        dispatch_bgmv(
            y,
            x,
            a_buf,
            b_buf,
            starts,
            a_len,
            scaling,
            req,
            layer_id=0,
            projection="pytest",
        )
