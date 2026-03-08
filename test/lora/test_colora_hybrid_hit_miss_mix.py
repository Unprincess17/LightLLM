import pytest
import torch
import importlib.util
from pathlib import Path

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import (
    ExpertCacheKey,
    MoEExpertCacheConfig,
    MoEExpertCacheManager,
)
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)



def _build_cpu_pool_two_adapters(dtype=torch.float16):
    pool = LoRAModulePool.create(
        pool_size=16,
        max_rank=4,
        input_dim=8,
        output_dim=8,
        dtype=dtype,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )

    rank = 2
    for adapter_idx in (0, 1):
        A = torch.randn(rank, 8, dtype=dtype)
        B = torch.randn(rank, 8, dtype=dtype)
        ok = pool.load_adapter(
            adapter_idx=adapter_idx,
            rank=rank,
            scaling=1.0,
            layer_weights={0: {"A": A, "B": B}},
        )
        assert ok

    return pool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="COLoRA GPU hit-path needs CUDA")
@pytest.mark.skipif(not dispatch_mod.BGMV_AVAILABLE, reason="BGMV kernel is required for GPU hit-path")
def test_colora_hybrid_mixed_hit_and_miss_tokens():
    pool = _build_cpu_pool_two_adapters(dtype=torch.float16)

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

    hit_key = ExpertCacheKey(projection="gate", adapter_idx=0, layer_id=0, expert_id=0)
    cache_mgr.record_access([hit_key])
    cache_mgr.schedule_promotion([hit_key])
    promoted = cache_mgr.apply_completed_promotions()
    assert promoted >= 1

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
    )
    dispatcher.expert_cache_manager = cache_mgr

    x = torch.randn(4, 8, dtype=torch.float16, device="cuda")
    bins = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="cuda")

    out = dispatcher._batch_apply_moe_lora_hybrid(
        input_tensor=x,
        layer_id=0,
        buffer_layer_id=0,
        pool=pool,
        bins=bins,
        projection="gate",
        expert_id=0,
    )

    assert out.shape == x.shape
    stats = dispatcher.pop_colora_stats()
    assert stats["colora_hit_tokens"] > 0
    assert stats["colora_miss_tokens"] > 0
    assert "cpu_queue_wait_time" in stats
    assert "d2h_bytes" in stats
    assert "h2d_bytes" in stats
    assert "overlap_ratio" in stats
    assert "fallback_degrade_count" in stats
