import pytest
import torch

from lightllm.server.lora.expert_cache import (
    ExpertCacheKey,
    MoEExpertCacheConfig,
    MoEExpertCacheManager,
)
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


def _build_pool_many_adapters(num_adapters: int = 8):
    pool = LoRAModulePool.create(
        pool_size=64,
        max_rank=4,
        input_dim=16,
        output_dim=16,
        dtype=torch.float16,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )

    for adapter_idx in range(num_adapters):
        A = torch.full((2, 16), float(adapter_idx + 1), dtype=torch.float16)
        B = torch.full((2, 16), float(adapter_idx + 2), dtype=torch.float16)
        ok = pool.load_adapter(
            adapter_idx=adapter_idx,
            rank=2,
            scaling=1.0,
            layer_weights={0: {"A": A, "B": B}},
        )
        assert ok

    return pool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU cache allocation requires CUDA")
def test_colora_eviction_does_not_write_back_to_cpu_pool():
    pool = _build_pool_many_adapters(num_adapters=10)
    cpu_key_before = pool.key_buffer.clone()
    cpu_val_before = pool.value_buffer.clone()

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=1,
            promote_min_hits=1,
            promote_window=64,
            max_promote_per_step=64,
            decay=0.9,
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    keys = [ExpertCacheKey("gate", i, 0, 0) for i in range(10)]
    cache_mgr.record_access(keys)
    cache_mgr.schedule_promotion(keys)
    cache_mgr.apply_completed_promotions()

    ready = cache_mgr.lookup_many(keys)
    # Capacity is bounded, so at most max_slots entries can stay READY.
    state = cache_mgr._states["gate"]
    assert len(ready) <= state.max_slots

    # CPU source pool remains immutable for eviction/promotion operations.
    assert torch.equal(pool.key_buffer, cpu_key_before)
    assert torch.equal(pool.value_buffer, cpu_val_before)
