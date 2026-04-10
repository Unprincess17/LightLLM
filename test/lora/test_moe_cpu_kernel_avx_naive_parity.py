"""
Parity checks: MoE AVX CPU LoRA kernels vs the PyTorch reference used in
``COLORA_CPU_KERNEL_MODE=naive`` (see ``lightllm.models.qwen3_vl_moe.lora_dispatch``).

Skipped automatically when the MoE AVX extension is unavailable on the host.
"""

from __future__ import annotations

import pytest
import torch

_moe = pytest.importorskip(
    "lightllm._kernels.lora.moe_lora_cpu_kernel",
    reason="MoE LoRA AVX extension not importable",
)
if not _moe.is_available():
    pytest.skip(
        "MoE LoRA AVX kernel not available on this CPU",
        allow_module_level=True,
    )

moe_batch_lora_gate_avx = _moe.moe_batch_lora_gate_avx
moe_batch_lora_up_avx = _moe.moe_batch_lora_up_avx
moe_batch_lora_down_avx = _moe.moe_batch_lora_down_avx

# Must match ``_naive_moe_lora_gate`` / ``_naive_moe_lora_stage2`` in lora_dispatch.py
def _naive_gate(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, A.transpose(0, 1))


def _naive_stage2(inter: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    return torch.matmul(inter, B) * scaling


def _max_rel_diff_bf16(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    max_abs = (af - bf).abs().max().item()
    scale = max(bf.abs().max().item(), 1e-6)
    return max_abs / scale


@pytest.fixture
def rng():
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    return g


@pytest.mark.parametrize("n,h,r", [(1, 256, 8), (4, 2048, 16), (32, 4096, 32)])
def test_gate_avx_matches_naive_matmul(rng, n, h, r):
    x = torch.randn(n, h, dtype=torch.bfloat16, generator=rng)
    A = torch.randn(r, h, dtype=torch.bfloat16, generator=rng)
    out_avx = moe_batch_lora_gate_avx(x, A, scaling=1.0)
    out_ref = _naive_gate(x, A)
    assert out_avx.shape == (n, r)
    rel = _max_rel_diff_bf16(out_avx, out_ref)
    assert rel <= 0.02, f"gate rel_diff={rel:.4f} n={n} h={h} r={r}"


@pytest.mark.parametrize("n,h,r", [(1, 256, 8), (8, 2048, 32), (64, 4096, 64)])
def test_up_avx_matches_naive_matmul(rng, n, h, r):
    inter = torch.randn(n, r, dtype=torch.bfloat16, generator=rng)
    B = torch.randn(r, h, dtype=torch.bfloat16, generator=rng)
    scaling = 0.125
    out_avx = moe_batch_lora_up_avx(inter, B, scaling=scaling)
    out_ref = _naive_stage2(inter, B, scaling)
    assert out_avx.shape == (n, h)
    rel = _max_rel_diff_bf16(out_avx, out_ref)
    assert rel <= 0.02, f"up rel_diff={rel:.4f} n={n} h={h} r={r}"


@pytest.mark.parametrize("n,h,r", [(1, 256, 8), (8, 2048, 32), (64, 4096, 64)])
def test_down_avx_matches_naive_matmul(rng, n, h, r):
    inter = torch.randn(n, r, dtype=torch.bfloat16, generator=rng)
    B = torch.randn(r, h, dtype=torch.bfloat16, generator=rng)
    scaling = 0.25
    out_avx = moe_batch_lora_down_avx(inter, B, scaling=scaling)
    out_ref = _naive_stage2(inter, B, scaling)
    assert out_avx.shape == (n, h)
    rel = _max_rel_diff_bf16(out_avx, out_ref)
    assert rel <= 0.02, f"down rel_diff={rel:.4f} n={n} h={h} r={r}"


@pytest.mark.parametrize("n,h,r", [(2, 1024, 16), (16, 2048, 32)])
def test_gate_then_up_pipeline_matches_naive_two_stage(rng, n, h, r):
    """Same composition as strict MoE CPU path: gate AVX then stage-2 up AVX."""
    x = torch.randn(n, h, dtype=torch.bfloat16, generator=rng)
    A = torch.randn(r, h, dtype=torch.bfloat16, generator=rng)
    B = torch.randn(r, h, dtype=torch.bfloat16, generator=rng)
    scaling = 0.1
    inter_avx = moe_batch_lora_gate_avx(x, A, scaling=1.0)
    out_avx = moe_batch_lora_up_avx(inter_avx, B, scaling=scaling)
    inter = _naive_gate(x, A)
    out_ref = _naive_stage2(inter, B, scaling)
    assert out_avx.shape == (n, h)
    rel = _max_rel_diff_bf16(out_avx, out_ref)
    assert rel <= 0.03, f"pipeline rel_diff={rel:.4f}"
