import torch

from lightllm.server.lora.expert_cache import (
    ExpertCacheKey,
    ExpertCacheSlotState,
    MoEExpertCacheConfig,
    MoEExpertCacheManager,
)
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


def _build_pool(num_adapters: int = 4):
    pool = LoRAModulePool.create(
        pool_size=32,
        max_rank=4,
        input_dim=16,
        output_dim=16,
        dtype=torch.float16,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    for adapter_idx in range(num_adapters):
        ok = pool.load_adapter(
            adapter_idx=adapter_idx,
            rank=2,
            scaling=1.0,
            layer_weights={
                0: {
                    "A": torch.randn(2, 16, dtype=torch.float16),
                    "B": torch.randn(2, 16, dtype=torch.float16),
                }
            },
        )
        assert ok
    return pool


def test_colora_cache_config_defaults_cover_new_controls():
    cfg = MoEExpertCacheConfig()
    assert cfg.miss_policy == "cpu_first"
    assert cfg.queue_high_watermark is None
    assert cfg.promote_cooldown_steps == 4


def test_colora_queue_high_watermark_limits_promotions():
    pool = _build_pool(num_adapters=4)
    mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=1,
            promote_min_hits=1,
            promote_window=8,
            max_promote_per_step=8,
            queue_high_watermark=1,
        )
    )
    mgr.register_projection_pool("gate", pool)

    keys = [ExpertCacheKey("gate", adapter_idx, 0, 0) for adapter_idx in range(4)]
    mgr.record_access(keys)
    queued = mgr.schedule_promotion(keys)
    drops = mgr.get_promotion_drop_breakdown()

    assert queued == 1
    assert drops["queue_high_watermark"] >= 3


def test_colora_promotion_cooldown_blocks_requeue():
    pool = _build_pool(num_adapters=1)
    mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=1,
            promote_min_hits=1,
            promote_window=8,
            max_promote_per_step=8,
            queue_high_watermark=8,
            promote_cooldown_steps=16,
        )
    )
    mgr.register_projection_pool("gate", pool)

    key = ExpertCacheKey("gate", 0, 0, 0)
    mgr.record_access([key])
    first = mgr.schedule_promotion([key])
    assert first == 1

    # Clear queued marker to emulate "key consumed from queue but still cooling down".
    state = mgr._states["gate"]
    entry = state.entries[key]
    state.promotion_queue.clear()
    state.queued.clear()
    entry.state = ExpertCacheSlotState.INVALID

    second = mgr.schedule_promotion([key])
    drops = mgr.get_promotion_drop_breakdown()
    assert second == 0
    assert drops["cooldown"] >= 1
