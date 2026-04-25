"""Phase 5 regression: GPU vs CPU hit-indexing must produce byte-identical outputs.

Runs the full COLoRA hybrid dispatch on a synthetic mixed-hit/miss batch with
``colora_hit_indexing="gpu"`` and ``"cpu"`` and asserts the final output tensor
and the hit/miss token counts agree exactly.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import (
    ExpertCacheKey,
    MoEExpertCacheConfig,
    MoEExpertCacheManager,
)
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod_parity", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
sys.modules[_DISPATCH_SPEC.name] = dispatch_mod
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)


def _install_fake_moe_kernel(monkeypatch):
    def _gate(x, A, scaling):
        return torch.matmul(x, A.t()) * scaling

    def _updown(x, B, scaling):
        return torch.matmul(x, B) * scaling

    monkeypatch.setattr(dispatch_mod, "MOE_AVX_AVAILABLE", True, raising=False)
    monkeypatch.setattr(dispatch_mod, "moe_batch_lora_gate_avx", _gate, raising=False)
    monkeypatch.setattr(dispatch_mod, "moe_batch_lora_up_avx", _updown, raising=False)
    monkeypatch.setattr(dispatch_mod, "moe_batch_lora_down_avx", _updown, raising=False)


def _build_cpu_pool(dtype=torch.float16, num_adapters=4, rank=2, hidden=8):
    pool = LoRAModulePool.create(
        pool_size=16 * num_adapters,
        max_rank=rank,
        input_dim=hidden,
        output_dim=hidden,
        dtype=dtype,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    for adapter_idx in range(num_adapters):
        A = torch.randn(rank, hidden, dtype=dtype)
        B = torch.randn(rank, hidden, dtype=dtype)
        assert pool.load_adapter(
            adapter_idx=adapter_idx,
            rank=rank,
            scaling=1.0,
            layer_weights={0: {"A": A, "B": B}},
        )
    return pool


def _run_dispatch(hit_indexing: str, pool, bins: torch.Tensor, x: torch.Tensor):
    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=64,
            promote_min_hits=1,
            promote_window=16,
            max_promote_per_step=8,
            decay=0.9,
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    # Promote adapters 0 and 2 -> hit; leave 1 and 3 as miss.
    for adapter_idx in (0, 2):
        key = ExpertCacheKey(projection="gate", adapter_idx=adapter_idx, layer_id=0, expert_id=0)
        cache_mgr.record_access([key])
        cache_mgr.schedule_promotion([key])
    promoted = cache_mgr.apply_completed_promotions()
    assert promoted >= 2

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        colora_hit_indexing=hit_indexing,
    )
    dispatcher.expert_cache_manager = cache_mgr

    out = dispatcher._batch_apply_moe_lora_hybrid(
        input_tensor=x,
        layer_id=0,
        buffer_layer_id=0,
        pool=pool,
        bins=bins,
        projection="gate",
        expert_id=0,
    )
    stats = dispatcher.pop_colora_stats()
    return out.detach().clone(), stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="COLoRA GPU hit-path needs CUDA")
@pytest.mark.skipif(not dispatch_mod.BGMV_AVAILABLE, reason="BGMV kernel is required for GPU hit-path")
@pytest.mark.parametrize(
    "bins_py",
    [
        # Mixed hits (0, 2) and misses (1, 3) with repeats.
        [0, 1, 2, 3, 0, 2, 1, 3],
        # All-hit case.
        [0, 2, 0, 2, 0, 2, 0, 2],
        # All-miss case.
        [1, 3, 1, 3, 1, 3, 1, 3],
    ],
)
def test_colora_hit_indexing_gpu_vs_cpu_parity(monkeypatch, bins_py):
    _install_fake_moe_kernel(monkeypatch)
    torch.manual_seed(1234)
    pool = _build_cpu_pool()

    device = torch.device("cuda")
    x = torch.randn(len(bins_py), 8, dtype=torch.float16, device=device)
    bins = torch.tensor(bins_py, dtype=torch.long, device=device)

    out_gpu, stats_gpu = _run_dispatch("gpu", pool, bins, x)
    out_cpu, stats_cpu = _run_dispatch("cpu", pool, bins, x)

    assert out_gpu.shape == out_cpu.shape
    assert torch.equal(out_gpu, out_cpu), (
        f"GPU vs CPU hit-indexing divergence: max abs diff = "
        f"{(out_gpu.float() - out_cpu.float()).abs().max().item()}"
    )
    assert int(stats_gpu["colora_hit_tokens"]) == int(stats_cpu["colora_hit_tokens"])
    assert int(stats_gpu["colora_miss_tokens"]) == int(stats_cpu["colora_miss_tokens"])
