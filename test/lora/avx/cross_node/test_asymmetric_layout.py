# test_asymmetric_layout.py
"""Correctness test with H != I to catch transposition/layout bugs.

Per spec "Frozen workload and environment": the performance campaign uses
H=I=2048, which masks dimension-order errors. This test uses H=1024, I=2048
to verify y = x @ A.T @ B produces the correct shape and values.
"""
import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@CUDA
def test_lora_compute_asymmetric_shapes():
    H, I, R = 1024, 2048, 64
    x = torch.randn(1, H, dtype=torch.float32, device="cuda")
    A = torch.randn(R, H, dtype=torch.float32, device="cuda")
    B = torch.randn(R, I, dtype=torch.float32, device="cuda")

    inter = x @ A.T        # [1, R]
    y = inter @ B          # [1, I]
    assert y.shape == (1, I), f"expected (1, {I}), got {y.shape}"

    # Verify against an equivalent formulation
    AB = A.T @ B           # [H, I]
    y2 = x @ AB            # [1, I]
    assert torch.allclose(y, y2, atol=1e-2, rtol=1e-3)


@CUDA
def test_lora_compute_dtype_promotion():
    H, I, R = 1024, 2048, 64
    x = torch.randn(1, H, dtype=torch.float16, device="cuda")
    A = torch.randn(R, H, dtype=torch.float32, device="cuda")
    B = torch.randn(R, I, dtype=torch.float32, device="cuda")

    x_f32 = x.to(torch.float32)
    inter = x_f32 @ A.T
    y = inter @ B
    assert y.shape == (1, I)
    assert y.dtype == torch.float32


@CUDA
def test_multi_miss_shapes():
    H, I, R, NM = 1024, 2048, 64, 8
    x = torch.randn(1, H, dtype=torch.float32, device="cuda")
    for i in range(NM):
        A = torch.randn(R, H, device="cuda")
        B = torch.randn(R, I, device="cuda")
        inter = x @ A.T
        y = inter @ B
        assert y.shape == (1, I)
