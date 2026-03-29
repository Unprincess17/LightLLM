import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import MoEExpertCacheConfig, MoEExpertCacheManager
from lightllm.server.lora.lora_mem_pool import LoRAModulePool


_DISPATCH_PATH = Path(__file__).resolve().parents[2] / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"
_DISPATCH_SPEC = importlib.util.spec_from_file_location("colora_background_dispatch_mod", _DISPATCH_PATH)
dispatch_mod = importlib.util.module_from_spec(_DISPATCH_SPEC)
assert _DISPATCH_SPEC is not None and _DISPATCH_SPEC.loader is not None
sys.modules[_DISPATCH_SPEC.name] = dispatch_mod
_DISPATCH_SPEC.loader.exec_module(dispatch_mod)


def _wait_until(predicate, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _build_projection_pool(dtype=torch.float32):
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
        scaling=1.5,
        layer_weights={
            0: {
                "A": torch.arange(8, dtype=dtype).reshape(2, 4),
                "B": torch.arange(8, dtype=dtype).reshape(2, 4),
            }
        },
    )
    assert ok
    return pool


def _build_dummy_lora_mem_pool():
    return SimpleNamespace(
        moe_gate_pool=_build_projection_pool(),
        moe_up_pool=_build_projection_pool(),
        moe_down_pool=_build_projection_pool(),
    )


def test_joint_access_interval_tracker_updates_ema_asynchronously():
    tracker = dispatch_mod.JointAccessIntervalTracker(ema_alpha=0.5, max_queue_size=8)
    key = dispatch_mod.JointObjectKey(layer_id=3, adapter_bin=1, expert_id=7)
    try:
        assert tracker.get_ema_interval_steps(key) is None
        assert tracker.submit_access(key, 10) is True
        assert tracker.submit_access(key, 14) is True
        assert _wait_until(lambda: tracker.get_ema_interval_steps(key) is not None)
        assert tracker.get_ema_interval_steps(key) == 4.0

        assert tracker.submit_access(key, 20) is True
        assert _wait_until(lambda: abs(tracker.get_ema_interval_steps(key) - 5.0) < 1e-6)
    finally:
        tracker.close()


def test_note_decode_joint_access_admits_joint_promotion_once():
    lora_mem_pool = _build_dummy_lora_mem_pool()
    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=1,
            promote_min_hits=1,
            promote_window=8,
            max_promote_per_step=8,
            queue_high_watermark=8,
        )
    )
    cache_mgr.register_projection_pool("gate", lora_mem_pool.moe_gate_pool)
    cache_mgr.register_projection_pool("up", lora_mem_pool.moe_up_pool)
    cache_mgr.register_projection_pool("down", lora_mem_pool.moe_down_pool)

    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        up_lora_rank=2,
        down_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        colora_deferred_promotion_delta_steps=4,
    )
    dispatcher.expert_cache_manager = cache_mgr

    joint_key = dispatch_mod.JointObjectKey(layer_id=0, adapter_bin=0, expert_id=0)
    with dispatcher._promotion_interval_tracker._lock:
        dispatcher._promotion_interval_tracker._states[joint_key] = dispatch_mod._JointAccessState(
            last_decode_step_id=5,
            ema_interval_steps=2.0,
            interval_sample_count=1,
        )

    try:
        dispatcher.note_decode_joint_access(
            decode_step_id=8,
            layer_id=0,
            expert_id=0,
            adapter_bins=[0, 0],
        )
        assert cache_mgr.get_promotion_queue_depth() == 3
        assert dispatcher._pending_background_stats["promotion_admitted"] == 1
    finally:
        dispatcher._promotion_interval_tracker.close()


def test_temporal_hot_cache_exact_lookup_uses_step_scoped_key():
    lora_mem_pool = _build_dummy_lora_mem_pool()
    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        up_lora_rank=2,
        down_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        colora_temporal_prefetch=True,
        colora_temporal_hot_cache_slots=2,
    )
    dispatcher.init_batched_mode(
        lora_mem_pool=lora_mem_pool,
        req_bins=torch.tensor([0], dtype=torch.long),
        expert_cache_manager=None,
    )
    assert dispatcher._temporal_hot_cache is not None

    try:
        assert dispatcher.begin_temporal_prefetch_step(9)["retired_active"] == 0
        assert dispatcher.maybe_submit_temporal_prefetch_job(
            decode_step_id=9,
            layer_id=0,
            adapter_bin=0,
            expert_id=0,
        )
        key = dispatch_mod.TemporalPrefetchJobKey(layer_id=0, decode_step_id=9, adapter_bin=0, expert_id=0)
        assert _wait_until(lambda: dispatcher._temporal_hot_cache.get_status(key) == "ready")

        weights, handle, status = dispatcher._maybe_acquire_prefetched_projection(
            projection="gate",
            adapter_idx=0,
            temporal_prefetch_context=(9, 0, 0),
        )
        assert status == "ready"
        assert weights is not None
        assert handle is not None
        assert weights.rank == 2
        handle.release()

        stale_weights, stale_handle, stale_status = dispatcher._maybe_acquire_prefetched_projection(
            projection="gate",
            adapter_idx=0,
            temporal_prefetch_context=(10, 0, 0),
        )
        assert stale_weights is None
        assert stale_handle is None
        assert stale_status == "missing"
    finally:
        dispatcher.cleanup_temporal_prefetch_state()
        dispatcher._promotion_interval_tracker.close()
