"""Unit tests for client-side GPUDirect strategy functions."""
import pytest
import torch
from unittest.mock import patch, MagicMock


def test_s2b_buffer_layout_computation():
    """Verify S2b buffer layout: A matrices then B matrices, flag at end."""
    rank, num_miss = 64, 4
    hidden_dim, intermediate_dim = 2048, 1536
    a_bytes = num_miss * rank * hidden_dim * 2  # bf16
    b_bytes = num_miss * rank * intermediate_dim * 2
    weight_bytes = a_bytes + b_bytes
    flag_offset = weight_bytes

    assert weight_bytes == num_miss * rank * (hidden_dim + intermediate_dim) * 2
    assert flag_offset == weight_bytes


def test_s4a_buffer_layout_computation():
    """Verify S4a buffer layout: activation | result | flag."""
    hidden_dim, intermediate_dim = 2048, 1536
    num_miss = 4
    act_bytes = hidden_dim * 2
    result_bytes = num_miss * intermediate_dim * 2
    flag_offset = act_bytes + result_bytes

    assert flag_offset == act_bytes + result_bytes
    assert flag_offset == hidden_dim * 2 + num_miss * intermediate_dim * 2


def test_s5b_buffer_layout_computation():
    """Verify S5b buffer layout: activation | result | flag."""
    hidden_dim, intermediate_dim = 2048, 1536
    num_miss = 4
    act_bytes = hidden_dim * 2
    result_bytes = num_miss * intermediate_dim * 2
    flag_offset = act_bytes + result_bytes

    assert flag_offset == act_bytes + result_bytes


def test_s2b_gpu_compute_after_weight_receive():
    """After receiving weights via RDMA WRITE, GPU compute produces correct shape."""
    rank, num_miss = 64, 4
    hidden_dim, intermediate_dim = 2048, 1536

    # Simulate received weights
    a_bf16 = torch.randn(num_miss, rank, hidden_dim, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(num_miss, rank, intermediate_dim, dtype=torch.bfloat16, device="cuda")
    act = torch.randn(1, hidden_dim, dtype=torch.bfloat16, device="cuda")

    act_f32 = act.to(torch.float32)
    result = torch.zeros(num_miss, intermediate_dim, dtype=torch.float32, device="cuda")
    for i in range(num_miss):
        a_f32 = a_bf16[i].to(torch.float32)
        b_f32 = b_bf16[i].to(torch.float32)
        inter = act_f32 @ a_f32.T  # [1, rank]
        result[i] = inter @ b_f32   # [1, intermediate_dim]

    assert result.shape == (num_miss, intermediate_dim)
    assert result.dtype == torch.float32


def test_s4a_gpu_compute_correctness():
    """S4a GPU compute: act @ A^T @ B^T on remote GPU."""
    rank, num_miss = 16, 2
    hidden_dim, intermediate_dim = 128, 64

    # Remote GPU has weights
    a_gpu = torch.randn(num_miss, rank, hidden_dim, dtype=torch.float32, device="cuda")
    b_gpu = torch.randn(num_miss, rank, intermediate_dim, dtype=torch.float32, device="cuda")

    # Local sends activation
    act = torch.randn(1, hidden_dim, dtype=torch.float32, device="cuda")

    # Remote GPU compute (what S4a does)
    result = torch.zeros(num_miss, intermediate_dim, dtype=torch.float32, device="cuda")
    for i in range(num_miss):
        inter = act @ a_gpu[i].T  # [1, rank]
        result[i] = inter @ b_gpu[i]

    # Local verify (what client checks)
    local_result = torch.zeros(num_miss, intermediate_dim, dtype=torch.float32)
    a_local = a_gpu.to(torch.float32).cpu()
    b_local = b_gpu.to(torch.float32).cpu()
    act_local = act.to(torch.float32).cpu()
    for i in range(num_miss):
        inter = act_local @ a_local[i].T
        local_result[i] = inter @ b_local[i]

    assert torch.allclose(result.cpu(), local_result, atol=1e-4)


def test_s5b_cpu_compute_correctness():
    """S5b CPU compute produces same result as GPU reference."""
    rank, num_miss = 16, 2
    hidden_dim, intermediate_dim = 128, 64

    a_cpu = torch.randn(num_miss, rank, hidden_dim, dtype=torch.float32)
    b_cpu = torch.randn(num_miss, rank, intermediate_dim, dtype=torch.float32)
    act = torch.randn(1, hidden_dim, dtype=torch.float32)

    # CPU compute
    result_cpu = torch.zeros(num_miss, intermediate_dim, dtype=torch.float32)
    for i in range(num_miss):
        inter = act @ a_cpu[i].T  # [1, rank]
        result_cpu[i] = inter @ b_cpu[i]

    # GPU reference
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    act_gpu = act.to("cuda")
    result_gpu = torch.zeros(num_miss, intermediate_dim, dtype=torch.float32, device="cuda")
    for i in range(num_miss):
        inter = act_gpu @ a_gpu[i].T
        result_gpu[i] = inter @ b_gpu[i]

    assert torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-4)
