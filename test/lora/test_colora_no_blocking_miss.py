import torch
import importlib.util
import sys
from pathlib import Path

import pytest

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import MoEExpertCacheConfig, MoEExpertCacheManager, PromotionApplyResult
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_dispatch_mod", _DISPATCH_PATH)
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



def _build_pool_single_adapter(dtype=torch.float32):
    pool = LoRAModulePool.create(
        pool_size=8,
        max_rank=4,
        input_dim=4,
        output_dim=4,
        dtype=dtype,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    ok = pool.load_adapter(
        adapter_idx=0,
        rank=2,
        scaling=1.0,
        layer_weights={
            0: {
                "A": torch.randn(2, 4, dtype=dtype),
                "B": torch.randn(2, 4, dtype=dtype),
            }
        },
    )
    assert ok
    return pool


def test_colora_hybrid_miss_path_returns_without_waiting_for_promotion(monkeypatch):
    _install_fake_moe_kernel(monkeypatch)
    pool = _build_pool_single_adapter()

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=16,
            promote_min_hits=100,  # ensure this call does not trigger promotion
            promote_window=8,
            max_promote_per_step=1,
            decay=0.9,
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
    )
    dispatcher.expert_cache_manager = cache_mgr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 4, dtype=torch.float32, device=device)
    bins = torch.tensor([0, 0], dtype=torch.long, device=device)

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
    required_observability_fields = {
        # hit/miss and queueing
        "colora_miss_tokens",
        "promotion_queue_depth",
        "cpu_queue_depth",
        # cache observability (total + per projection)
        "cache_capacity_slots",
        "cache_resident_slots",
        "cache_free_slots",
        "cache_evictions_total",
        "gate_capacity_slots",
        "gate_resident_slots",
        "gate_free_slots",
        "gate_evictions_total",
        "up_capacity_slots",
        "up_resident_slots",
        "up_free_slots",
        "up_evictions_total",
        "down_capacity_slots",
        "down_resident_slots",
        "down_free_slots",
        "down_evictions_total",
        # transfer/overlap/fallback
        "cpu_queue_wait_time",
        "d2h_bytes",
        "h2d_bytes",
        "overlap_ratio",
        "fallback_degrade_count",
        # promotion policy/drop accounting
        "promotion_drop_total",
        "promotion_drop_queue_high_watermark",
        "promotion_drop_cooldown",
        "promotion_admitted",
        "promotion_reject_delta",
        "promotion_reject_no_ema",
        "tracker_queue_drop",
        # prefetch observability
        "prefetch_submitted",
        "prefetch_ready_hits",
        "prefetch_not_ready",
        "prefetch_stale",
        "prefetch_false_positives",
        "prefetch_slot_overwrite",
        # kernel accounting
        "moe_kernel_calls",
        "moe_kernel_tokens",
    }
    missing = sorted(required_observability_fields - set(stats.keys()))
    assert not missing, f"missing COLoRA observability fields: {missing}"

    assert stats["colora_hit_tokens"] == 0
    assert stats["colora_miss_tokens"] == 2
    assert stats["cache_hit_rate"] == 0.0
    assert stats["promotion_queue_depth"] == 0
    assert stats["promotion_admitted"] == 0
    assert stats["cpu_queue_wait_time"] == 0.0
    assert stats["fallback_degrade_count"] == 0
    assert stats["cpu_compute_time"] > 0.0
    assert stats["gpu_compute_time"] == 0.0
    assert stats["h2d_bytes"] > 0.0
    assert stats["moe_kernel_calls"] > 0
    assert stats["moe_kernel_tokens"] == 2


def test_no_cpu_path_promotes_blocking_and_rejects_cpu_fallback(monkeypatch):
    _install_fake_moe_kernel(monkeypatch)
    pool = _build_pool_single_adapter()

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=16,
            promote_min_hits=1,
            promote_window=8,
            max_promote_per_step=1,
            decay=0.9,
            miss_policy="no_cpu_path",
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    promoted_keys = []

    def _promote_blocking(keys):
        promoted_keys.extend(keys)
        return PromotionApplyResult(
            ready_slots={key: 0 for key in keys},
            promoted_count=len(keys),
            transferred_bytes=4096 * len(keys),
        )

    monkeypatch.setattr(cache_mgr, "promote_blocking", _promote_blocking)

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
    )
    dispatcher.expert_cache_manager = cache_mgr

    x = torch.randn(2, 4, dtype=torch.float32, device="cpu")
    bins = torch.tensor([0, 0], dtype=torch.long, device="cpu")

    with pytest.raises(RuntimeError, match="requires GPU cached execution"):
        dispatcher._batch_apply_moe_lora_hybrid(
            input_tensor=x,
            layer_id=0,
            buffer_layer_id=0,
            pool=pool,
            bins=bins,
            projection="gate",
            expert_id=0,
        )

    assert len(promoted_keys) == 1
    assert promoted_keys[0].adapter_idx == 0


def test_no_deferred_sync_promotes_after_cpu_miss_completion(monkeypatch):
    _install_fake_moe_kernel(monkeypatch)
    pool = _build_pool_single_adapter()

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=16,
            promote_min_hits=1,
            promote_window=8,
            max_promote_per_step=1,
            decay=0.9,
            miss_policy="no_deferred_sync",
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    promoted_batches = []

    def _promote_blocking(keys):
        promoted_batches.append(list(keys))
        return PromotionApplyResult(
            ready_slots={key: 0 for key in keys},
            promoted_count=len(keys),
            transferred_bytes=8192 * len(keys),
        )

    monkeypatch.setattr(cache_mgr, "promote_blocking", _promote_blocking)

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        colora_async_fallback=False,
    )
    dispatcher.expert_cache_manager = cache_mgr

    x = torch.randn(2, 4, dtype=torch.float32, device="cpu")
    bins = torch.tensor([0, 0], dtype=torch.long, device="cpu")

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
    assert len(promoted_batches) == 1
    stats = dispatcher.pop_colora_stats()
    assert stats["miss_policy"] == "no_deferred_sync"
    assert stats["blocking_promotion_count"] == 1
    assert stats["weight_h2d_bytes"] == 8192.0
