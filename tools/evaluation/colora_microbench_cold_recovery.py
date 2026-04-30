#!/usr/bin/env python3
"""P2 Cold Miss Recovery Microbenchmark.

Isolates one-layer MoE expert-LoRA decode cold-miss recovery and compares
load_then_run (promotion-first) vs cpu_first (execution-first) with
component-level latency decomposition and payload-size accounting.

Produces:
  - microbench_cold_recovery.csv
  - microbench_payload_size.csv
"""

import os

# Set triton cache dir before any other imports
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

# Force naive CPU kernel mode and availability flag for systems without AVX-512 BF16
os.environ.setdefault("COLORA_CPU_KERNEL_MODE", "naive")
import lightllm.models.qwen3_vl_moe.lora_dispatch as lora_dispatch_module
lora_dispatch_module.MOE_AVX_AVAILABLE = True

import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml


# ---------------------------------------------------------------------------
# Schema: component latency names per strategy
# ---------------------------------------------------------------------------

EXECUTION_FIRST_COMPONENTS = [
    "Tpack_us",
    "TD2H_activation_us",
    "Tcpu_us",
    "TH2D_residual_us",
    "Tmerge_us",
]

PROMOTION_FIRST_COMPONENTS = [
    "Tadmit_us",
    "TH2D_weights_us",
    "Tgpu_us",
]

ALL_COMPONENTS = sorted(set(EXECUTION_FIRST_COMPONENTS + PROMOTION_FIRST_COMPONENTS))

# Maps from dispatcher stats -> P2 component names (filled at runtime after
# we measure the actual path).  The sweep runner builds these mappings from
# the fine-grained NVTX phases and dispatcher counters.

_EXECUTION_FIRST_MAP = {
    "Tpack_us": "pack_time_us",
    "TD2H_activation_us": "d2h_activation_time_us",
    "Tcpu_us": "cpu_compute_time_us",
    "TH2D_residual_us": "h2d_residual_time_us",
    "Tmerge_us": "merge_time_us",
}

_PROMOTION_FIRST_MAP = {
    "Tadmit_us": "admit_time_us",
    "TH2D_weights_us": "weight_h2d_time_us",
    "Tgpu_us": "gpu_compute_time_us",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MicrobenchConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_experts: int = 64
    topk: int = 8
    ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dtype: str = "fp16"
    projection: str = "gate"
    repeats: int = 3
    warmup_iters: int = 10
    measurement_iters: int = 100
    cache_budget_mb: int = 64
    cpu_threads: int = 1
    cpu_kernel_mode: str = "avx"
    numa_node: Optional[int] = None
    promote_min_hits: int = 1
    promote_window: int = 16
    max_promote_per_step: int = 8
    decay: float = 0.9
    output_dir: str = "results/microbench_cold_recovery"

    @classmethod
    def from_yaml(cls, path: Path) -> "MicrobenchConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected mapping in YAML file: {path}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

@dataclass
class ColdRecoveryRow:
    policy: str
    lora_rank: int
    batch_size: int
    repeat_idx: int
    service_time_us: float
    component_sum_us: float
    residual_us: float
    Tpack_us: float = 0.0
    TD2H_activation_us: float = 0.0
    Tcpu_us: float = 0.0
    TH2D_residual_us: float = 0.0
    Tmerge_us: float = 0.0
    Tadmit_us: float = 0.0
    TH2D_weights_us: float = 0.0
    Tgpu_us: float = 0.0


@dataclass
class PayloadSizeRow:
    policy: str
    lora_rank: int
    batch_size: int
    promotion_weight_bytes: float
    execution_activation_bytes: float
    execution_residual_bytes: float
    payload_ratio: float  # weight_bytes / (activation_bytes + residual_bytes)


# ---------------------------------------------------------------------------
# One-layer benchmark driver (component-level)
# ---------------------------------------------------------------------------

def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _run_cold_miss_single(
    *,
    policy: str,
    lora_rank: int,
    batch_size: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    projection: str,
    dtype: torch.dtype,
    cache_budget_mb: int,
    warmup_iters: int,
    measurement_iters: int,
    promote_min_hits: int,
    promote_window: int,
    max_promote_per_step: int,
    decay: float,
) -> Dict[str, Any]:
    """Run the one-layer cold-miss benchmark for a single (policy, rank, batch).

    Returns a dict with per-iteration component timings and payload sizes.
    """
    from lightllm.models.qwen3_vl_moe.lora_dispatch import (
        BGMV_AVAILABLE,
        Qwen3VLMoELoRADispatcher,
    )
    from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
    from lightllm.server.lora.expert_cache import (
        ExpertCacheKey,
        ExpertCacheSlotState,
        MoEExpertCacheConfig,
        MoEExpertCacheManager,
    )
    from lightllm.server.lora.lora_mem_pool import LoRAModulePool

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not BGMV_AVAILABLE:
        raise RuntimeError("BGMV kernel is required")

    device = torch.device("cuda")
    layer_id = 0
    expert_id = 0
    hit_adapter_id = 0

    pool = LoRAModulePool.create(
        pool_size=32,
        max_rank=lora_rank,
        input_dim=hidden_size,
        output_dim=hidden_size,
        dtype=dtype,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )

    # Load hit adapter (always in cache for GPU path)
    lora_a = torch.randn(lora_rank, hidden_size, dtype=dtype)
    lora_b = torch.randn(lora_rank, hidden_size, dtype=dtype)
    ok = pool.load_adapter(
        adapter_idx=hit_adapter_id,
        rank=lora_rank,
        scaling=1.0,
        layer_weights={0: {"A": lora_a, "B": lora_b}},
    )
    if not ok:
        raise RuntimeError(f"failed to load hit adapter {hit_adapter_id}")

    # Load miss adapters: 1..batch_size (one per token for diversified cold miss)
    # Each token maps to a DIFFERENT adapter for realistic MoE routing
    miss_adapter_ids = list(range(1, 1 + batch_size))
    for adapter_idx in miss_adapter_ids:
        lora_a = torch.randn(lora_rank, hidden_size, dtype=dtype)
        lora_b = torch.randn(lora_rank, hidden_size, dtype=dtype)
        ok = pool.load_adapter(
            adapter_idx=adapter_idx,
            rank=lora_rank,
            scaling=1.0,
            layer_weights={0: {"A": lora_a, "B": lora_b}},
        )
        if not ok:
            raise RuntimeError(f"failed to load adapter {adapter_idx}")

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=cache_budget_mb,
            promote_min_hits=promote_min_hits,
            promote_window=promote_window,
            # For load_then_run: set max_promote_per_step to 0 to disable
            # async apply_completed_promotions, forcing all promotion to
            # happen synchronously in promote_blocking where timing is tracked
            max_promote_per_step=0 if policy == "load_then_run" else max_promote_per_step,
            decay=decay,
            miss_policy=policy,
        )
    )
    cache_mgr.register_projection_pool(projection, pool)

    # Pre-promote hit adapter so only miss_adapter is cold
    hit_key = ExpertCacheKey(
        projection=projection,
        adapter_idx=hit_adapter_id,
        layer_id=layer_id,
        expert_id=expert_id,
    )
    cache_mgr.record_access([hit_key])
    cache_mgr.schedule_promotion([hit_key])
    cache_mgr.apply_completed_promotions()

    rank_kwargs = {"gate_lora_rank": 0, "up_lora_rank": 0, "down_lora_rank": 0}
    rank_kwargs[f"{projection}_lora_rank"] = lora_rank
    dispatcher = Qwen3VLMoELoRADispatcher(
        num_layers=1,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        **rank_kwargs,
    )
    dispatcher.expert_cache_manager = cache_mgr

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    # Each token maps to a DIFFERENT miss adapter (diversified cold miss)
    # Adapter 0 = hit (pre-promoted), Adapters 1..batch_size = cold
    bins = torch.tensor(miss_adapter_ids, dtype=torch.long, device=device)

    # Create cache keys for all miss adapters
    miss_keys = [
        ExpertCacheKey(
            projection=projection,
            adapter_idx=adapter_idx,
            layer_id=layer_id,
            expert_id=expert_id,
        )
        for adapter_idx in miss_adapter_ids
    ]

    def _evict_miss_adapters() -> None:
        """Evict all miss adapters from cache so next iteration starts cold."""
        # First, ensure any async promotions are complete
        cache_mgr.apply_completed_promotions()
        # Evict by directly removing from cache manager entries
        if hasattr(cache_mgr, "_states"):
            state = cache_mgr._states.get(projection)
            if state:
                for miss_key in miss_keys:
                    if miss_key in state.entries:
                        entry = state.entries[miss_key]
                        slot_id = entry.slot_id
                        # Mark entry as INVALID before removal
                        entry.state = ExpertCacheSlotState.INVALID
                        entry.slot_id = -1
                        del state.entries[miss_key]
                        if slot_id >= 0:
                            state.slot_to_key.pop(slot_id, None)
                            # Return slot to free list
                            if slot_id not in state.free_slots:
                                state.free_slots.append(slot_id)
                        # Remove from queued set if present
                        state.queued.discard(miss_key)

    def _run_once_cpu_first():
        """cpu_first: suppress promotion to measure pure CPU execution path."""
        original_record = cache_mgr.record_access
        original_schedule = cache_mgr.schedule_promotion
        try:
            cache_mgr.record_access = lambda keys: None
            cache_mgr.schedule_promotion = lambda keys: None
            lora_out = dispatcher._batch_apply_moe_lora_hybrid(
                input_tensor=x,
                layer_id=layer_id,
                buffer_layer_id=layer_id,
                pool=pool,
                bins=bins,
                projection=projection,
                expert_id=expert_id,
            )
        finally:
            cache_mgr.record_access = original_record
            cache_mgr.schedule_promotion = original_schedule
        if lora_out.device.type == "cuda":
            torch.cuda.synchronize()
        return dispatcher.pop_colora_stats()

    def _run_once_load_then_run():
        """load_then_run: allow promotion to measure real H2D weight transfer + GPU compute."""
        # Ensure miss adapters are evicted before each run to guarantee cold start
        _evict_miss_adapters()
        # Record access TWICE to ensure access_count >= promote_min_hits
        # (required for schedule_promotion to actually queue the keys)
        cache_mgr.record_access(miss_keys)
        cache_mgr.record_access(miss_keys)
        # Run with promotion enabled
        lora_out = dispatcher._batch_apply_moe_lora_hybrid(
            input_tensor=x,
            layer_id=layer_id,
            buffer_layer_id=layer_id,
            pool=pool,
            bins=bins,
            projection=projection,
            expert_id=expert_id,
        )
        if lora_out.device.type == "cuda":
            torch.cuda.synchronize()
        return dispatcher.pop_colora_stats()

    _run_once = _run_once_cpu_first if policy == "cpu_first" else _run_once_load_then_run

    # Warmup Strategy: compile kernels BUT KEEP miss_adapters COLD
    if policy == "cpu_first":
        # For cpu_first: warm up CPU path directly
        for _ in range(warmup_iters):
            _run_once_cpu_first()
    else:
        # For load_then_run: warmup GPU kernels WITHOUT warming miss adapters
        # Use hit_adapter to warm up kernels
        warmup_bins = torch.tensor([0] * batch_size, dtype=torch.long, device=device)
        for _ in range(max(warmup_iters, 3)):
            _ = dispatcher._batch_apply_moe_lora_hybrid(
                input_tensor=x, layer_id=layer_id, buffer_layer_id=layer_id,
                pool=pool, bins=warmup_bins, projection=projection, expert_id=expert_id)
        torch.cuda.synchronize()

    # Measurement: collect per-iteration stats
    per_iter_stats: List[Dict[str, Any]] = []
    for _ in range(measurement_iters):
        stats = _run_once()
        per_iter_stats.append(stats)

    # Compute payload sizes from first iteration
    sample_stats = per_iter_stats[0]

    # Full expert-LoRA weight bytes: 2 * rank * hidden_size * element_size
    element_bytes = x.element_size()
    full_weight_bytes = 2.0 * lora_rank * hidden_size * element_bytes

    # Activation bytes: batch_size * hidden_size * element_size
    activation_bytes = float(batch_size) * hidden_size * element_bytes

    # Residual bytes: same as activation bytes (LoRA residual = same shape)
    residual_bytes = activation_bytes

    # Extract component timings from dispatcher stats
    # We compute from the already-accumulated per-iter counters
    n_iters = len(per_iter_stats)

    results: Dict[str, Any] = {
        "policy": policy,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "per_iter_stats": per_iter_stats,
        "payload": {
            "promotion_weight_bytes": full_weight_bytes,
            "execution_activation_bytes": activation_bytes,
            "execution_residual_bytes": residual_bytes,
        },
    }
    return results


def _extract_component_timings(
    per_iter_stats: List[Dict[str, Any]],
    policy: str,
) -> List[Dict[str, float]]:
    """Convert per-iteration dispatcher stats to P2 component timings.

    Mapping from dispatcher counters to P2 decomposition:
      Execution-first:
        Tpack    = pack_time (gather/index_select of miss tokens)
        TD2H     = d2h_activation_time (activation D2H transfer)
        Tcpu     = cpu_compute_time - d2h_activation_time - h2d_residual_time
        TH2D     = h2d_residual_time (residual H2D transfer)
        Tmerge   = merge_time (index_copy_ writeback)

      Promotion-first:
        Tadmit   = admit_time (cache admission)
        TH2D_w   = weight_h2d_time (full weight H2D transfer)
        Tgpu     = gpu_compute_time (GPU LoRA residual compute)
    """
    rows: List[Dict[str, float]] = []
    for stats in per_iter_stats:
        cpu_compute_s = float(stats.get("cpu_compute_time", 0.0))
        gpu_compute_s = float(stats.get("gpu_compute_time", 0.0))
        weight_h2d_s = float(stats.get("weight_h2d_time", 0.0))
        pack_s = float(stats.get("pack_time", 0.0))
        d2h_act_s = float(stats.get("d2h_activation_time", 0.0))
        h2d_res_s = float(stats.get("h2d_residual_time", 0.0))
        merge_s = float(stats.get("merge_time", 0.0))
        admit_s = float(stats.get("admit_time", 0.0))

        if policy == "cpu_first":
            pure_cpu_s = max(cpu_compute_s - d2h_act_s - h2d_res_s, 0.0)
            service_time_us = (pack_s + d2h_act_s + pure_cpu_s + h2d_res_s + merge_s + admit_s) * 1e6
            component_sum_us = service_time_us  # direct sum, no estimation

            row = {
                "service_time_us": service_time_us,
                "component_sum_us": component_sum_us,
                "residual_us": abs(service_time_us - component_sum_us),
                "Tpack_us": pack_s * 1e6,
                "TD2H_activation_us": d2h_act_s * 1e6,
                "Tcpu_us": pure_cpu_s * 1e6,
                "TH2D_residual_us": h2d_res_s * 1e6,
                "Tmerge_us": merge_s * 1e6,
                "Tadmit_us": admit_s * 1e6,
                "TH2D_weights_us": 0.0,
                "Tgpu_us": 0.0,
            }
        elif policy == "load_then_run":
            service_time_us = (admit_s + weight_h2d_s + gpu_compute_s + merge_s) * 1e6
            component_sum_us = service_time_us

            row = {
                "service_time_us": service_time_us,
                "component_sum_us": component_sum_us,
                "residual_us": abs(service_time_us - component_sum_us),
                "Tpack_us": 0.0,
                "TD2H_activation_us": 0.0,
                "Tcpu_us": 0.0,
                "TH2D_residual_us": 0.0,
                "Tmerge_us": merge_s * 1e6,
                "Tadmit_us": admit_s * 1e6,
                "TH2D_weights_us": weight_h2d_s * 1e6,
                "Tgpu_us": gpu_compute_s * 1e6,
            }
        else:
            raise ValueError(f"unknown policy: {policy}")

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Optional: overlap / exposed-time benchmark
# ---------------------------------------------------------------------------

def _run_overlap_benchmark(
    *,
    policy: str,
    lora_rank: int,
    batch_size: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    projection: str,
    dtype: torch.dtype,
    cache_budget_mb: int,
    warmup_iters: int,
    measurement_iters: int,
    promote_min_hits: int,
    promote_window: int,
    max_promote_per_step: int,
    decay: float,
    dummy_gpu_iters: int = 10,
) -> float:
    """Run the cold-miss benchmark with concurrent dummy GPU work.

    Measures the exposed stall time: how long the decode stream is
    actually blocked when the CPU cold path runs in parallel with
    dummy base/hot GPU computation.

    Returns median exposed_time_us across measurement iterations.
    """
    from lightllm.models.qwen3_vl_moe.lora_dispatch import (
        BGMV_AVAILABLE,
        Qwen3VLMoELoRADispatcher,
    )
    from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
    from lightllm.server.lora.expert_cache import (
        ExpertCacheKey,
        MoEExpertCacheConfig,
        MoEExpertCacheManager,
    )
    from lightllm.server.lora.lora_mem_pool import LoRAModulePool

    if not torch.cuda.is_available() or not BGMV_AVAILABLE:
        return 0.0

    device = torch.device("cuda")
    layer_id = 0
    expert_id = 0
    hit_adapter_id = 0
    miss_adapter_id = 1

    pool = LoRAModulePool.create(
        pool_size=32, max_rank=lora_rank, input_dim=hidden_size,
        output_dim=hidden_size, dtype=dtype, device="cpu",
        num_layers=1, num_experts=1,
    )
    for adapter_idx in (hit_adapter_id, miss_adapter_id):
        lora_a = torch.randn(lora_rank, hidden_size, dtype=dtype)
        lora_b = torch.randn(lora_rank, hidden_size, dtype=dtype)
        pool.load_adapter(
            adapter_idx=adapter_idx, rank=lora_rank, scaling=1.0,
            layer_weights={0: {"A": lora_a, "B": lora_b}},
        )

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=cache_budget_mb, promote_min_hits=promote_min_hits,
            promote_window=promote_window,
            # For load_then_run: set max_promote_per_step to 0 to disable
            # async apply_completed_promotions, forcing all promotion to
            # happen synchronously in promote_blocking where timing is tracked
            max_promote_per_step=0 if policy == "load_then_run" else max_promote_per_step,
            decay=decay, miss_policy=policy,
        )
    )
    cache_mgr.register_projection_pool(projection, pool)

    hit_key = ExpertCacheKey(
        projection=projection, adapter_idx=hit_adapter_id,
        layer_id=layer_id, expert_id=expert_id,
    )
    cache_mgr.record_access([hit_key])
    cache_mgr.schedule_promotion([hit_key])
    cache_mgr.apply_completed_promotions()

    rank_kwargs = {"gate_lora_rank": 0, "up_lora_rank": 0, "down_lora_rank": 0}
    rank_kwargs[f"{projection}_lora_rank"] = lora_rank
    dispatcher = Qwen3VLMoELoRADispatcher(
        num_layers=1,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        **rank_kwargs,
    )
    dispatcher.expert_cache_manager = cache_mgr

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    bins = torch.full((batch_size,), miss_adapter_id, dtype=torch.long, device=device)

    # Dummy GPU work: matmul to simulate base/hot computation
    dummy_w = torch.randn(hidden_size, hidden_size, dtype=dtype, device=device)
    dummy_bias = torch.randn(hidden_size, dtype=dtype, device=device)

    def _dummy_gpu_work():
        """Run dummy GPU matmul to simulate concurrent base computation."""
        out = torch.matmul(x, dummy_w) + dummy_bias
        torch.cuda.synchronize()
        return out

    def _run_once_with_overlap():
        # Start timing
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()

        # Launch dummy GPU work (simulates base/hot computation)
        import threading
        gpu_done_event = threading.Event()

        def _gpu_thread():
            _dummy_gpu_work()
            gpu_done_event.set()

        gpu_thread = threading.Thread(target=_gpu_thread)
        gpu_thread.start()

        # Simultaneously run the cold-miss LoRA path
        original_record = cache_mgr.record_access
        original_schedule = cache_mgr.schedule_promotion
        try:
            cache_mgr.record_access = lambda keys: None
            cache_mgr.schedule_promotion = lambda keys: None
            lora_out = dispatcher._batch_apply_moe_lora_hybrid(
                input_tensor=x, layer_id=layer_id, buffer_layer_id=layer_id,
                pool=pool, bins=bins, projection=projection, expert_id=expert_id,
            )
        finally:
            cache_mgr.record_access = original_record
            cache_mgr.schedule_promotion = original_schedule
        if lora_out.device.type == "cuda":
            torch.cuda.synchronize()

        # Wait for GPU thread to complete
        gpu_thread.join()
        wall_elapsed = time.perf_counter() - wall_t0

        # Get isolated service time from stats
        stats = dispatcher.pop_colora_stats()
        if policy == "cpu_first":
            service_s = stats.get("cpu_compute_time", 0.0) + stats.get("admit_time", 0.0)
        else:
            service_s = stats.get("gpu_compute_time", 0.0) + stats.get("weight_h2d_time", 0.0) + stats.get("admit_time", 0.0)

        # exposed_time = wall_time - max(gpu_dummy_time, service_time)
        # Approximation: exposed_time = wall_time - isolated_service_time
        # (since GPU dummy work would be running anyway)
        exposed_s = max(wall_elapsed - service_s, 0.0)
        return exposed_s * 1e6

    # Warmup
    for _ in range(warmup_iters):
        _run_once_with_overlap()

    # Measurement
    exposed_samples = []
    for _ in range(measurement_iters):
        exposed_samples.append(_run_once_with_overlap())

    return statistics.median(exposed_samples) if exposed_samples else 0.0


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(config: MicrobenchConfig, overlap_benchmark: bool = False) -> None:
    """Execute the full rank x batch x policy sweep with repeats."""
    dtype = _dtype_from_name(config.dtype)

    if config.cpu_threads > 0:
        torch.set_num_threads(config.cpu_threads)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recovery_rows: List[Dict[str, Any]] = []
    payload_rows: List[Dict[str, Any]] = []

    policies = ["cpu_first", "load_then_run"]

    for policy in policies:
        for lora_rank in config.ranks:
            for batch_size in config.batch_sizes:
                print(f"policy={policy} rank={lora_rank} batch={batch_size} ...", end=" ", flush=True)

                # Compute payload sizes (deterministic, same for all repeats)
                element_bytes = torch.tensor([], dtype=dtype).element_size()
                full_weight_bytes = 2.0 * lora_rank * config.hidden_size * element_bytes
                activation_bytes = float(batch_size) * config.hidden_size * element_bytes
                residual_bytes = activation_bytes
                payload_ratio = full_weight_bytes / max(activation_bytes + residual_bytes, 1.0)

                payload_rows.append({
                    "policy": policy,
                    "lora_rank": lora_rank,
                    "batch_size": batch_size,
                    "promotion_weight_bytes": full_weight_bytes,
                    "execution_activation_bytes": activation_bytes,
                    "execution_residual_bytes": residual_bytes,
                    "payload_ratio": round(payload_ratio, 4),
                })

                for repeat_idx in range(1, config.repeats + 1):
                    result = _run_cold_miss_single(
                        policy=policy,
                        lora_rank=lora_rank,
                        batch_size=batch_size,
                        hidden_size=config.hidden_size,
                        num_experts=config.num_experts,
                        topk=config.topk,
                        projection=config.projection,
                        dtype=dtype,
                        cache_budget_mb=config.cache_budget_mb,
                        warmup_iters=config.warmup_iters,
                        measurement_iters=config.measurement_iters,
                        promote_min_hits=config.promote_min_hits,
                        promote_window=config.promote_window,
                        max_promote_per_step=config.max_promote_per_step,
                        decay=config.decay,
                    )

                    component_rows = _extract_component_timings(
                        result["per_iter_stats"], policy
                    )

                    # Aggregate across iterations: median
                    def _median_field(field: str) -> float:
                        vals = [r[field] for r in component_rows if r[field] > 0]
                        return statistics.median(vals) if vals else 0.0

                    service_time_us = _median_field("service_time_us")
                    component_sum_us = _median_field("component_sum_us")
                    residual_us = abs(service_time_us - component_sum_us)

                    row = {
                        "policy": policy,
                        "lora_rank": lora_rank,
                        "batch_size": batch_size,
                        "repeat_idx": repeat_idx,
                        "service_time_us": round(service_time_us, 3),
                        "component_sum_us": round(component_sum_us, 3),
                        "residual_us": round(residual_us, 3),
                        "exposed_time_us": 0.0,  # filled by overlap benchmark if enabled
                    }
                    for comp in ALL_COMPONENTS:
                        row[comp] = round(_median_field(comp), 3)

                    recovery_rows.append(row)

                print("done")

    # Optional: overlap benchmark for exposed time
    if overlap_benchmark:
        print("\n--- Overlap benchmark (exposed time) ---")
        for policy in policies:
            for lora_rank in config.ranks:
                for batch_size in config.batch_sizes:
                    print(f"  overlap: policy={policy} rank={lora_rank} batch={batch_size} ...", end=" ", flush=True)
                    exposed_us = _run_overlap_benchmark(
                        policy=policy,
                        lora_rank=lora_rank,
                        batch_size=batch_size,
                        hidden_size=config.hidden_size,
                        num_experts=config.num_experts,
                        topk=config.topk,
                        projection=config.projection,
                        dtype=dtype,
                        cache_budget_mb=config.cache_budget_mb,
                        warmup_iters=config.warmup_iters,
                        measurement_iters=config.measurement_iters,
                        promote_min_hits=config.promote_min_hits,
                        promote_window=config.promote_window,
                        max_promote_per_step=config.max_promote_per_step,
                        decay=config.decay,
                    )
                    # Write exposed time to matching recovery rows
                    for row in recovery_rows:
                        if (row["policy"] == policy and row["lora_rank"] == lora_rank
                                and row["batch_size"] == batch_size):
                            row["exposed_time_us"] = round(exposed_us, 3)
                    print(f"exposed={exposed_us:.1f}us")

    # Write CSVs
    _write_recovery_csv(output_dir / "microbench_cold_recovery.csv", recovery_rows)
    _write_payload_csv(output_dir / "microbench_payload_size.csv", payload_rows)

    # Also write aggregated (mean across repeats)
    agg_rows = _aggregate_repeats(recovery_rows)
    _write_aggregated_csv(output_dir / "microbench_cold_recovery_aggregated.csv", agg_rows)

    print(f"\nResults written to {output_dir}/")
    print(f"  microbench_cold_recovery.csv       ({len(recovery_rows)} rows)")
    print(f"  microbench_cold_recovery_aggregated.csv ({len(agg_rows)} rows)")
    print(f"  microbench_payload_size.csv         ({len(payload_rows)} rows)")


def _aggregate_repeats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-repeat rows into median + MAD per (policy, rank, batch).

    Uses median instead of mean for robustness against warmup artifacts and outliers.
    Median absolute deviation (MAD) is reported as the dispersion metric.
    """
    from collections import defaultdict
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["policy"], row["lora_rank"], row["batch_size"])
        groups[key].append(row)

    numeric_fields = [
        "service_time_us", "component_sum_us", "residual_us", "exposed_time_us",
    ] + ALL_COMPONENTS

    agg_rows: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        policy, lora_rank, batch_size = key
        agg: Dict[str, Any] = {
            "policy": policy,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "n_repeats": len(group),
        }
        for f in numeric_fields:
            vals = [r[f] for r in group]
            # Median for robustness against warmup artifacts and outliers
            agg[f"{f}_median"] = round(statistics.median(vals), 3) if vals else 0.0
            # Median absolute deviation (MAD) as robust dispersion metric
            if vals and len(vals) > 1:
                med = statistics.median(vals)
                mad = statistics.median([abs(v - med) for v in vals])
                agg[f"{f}_mad"] = round(mad, 3)
            else:
                agg[f"{f}_mad"] = 0.0
        agg_rows.append(agg)
    return agg_rows


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

RECOVERY_FIELDS = [
    "policy", "lora_rank", "batch_size", "repeat_idx",
    "service_time_us", "component_sum_us", "residual_us",
] + ALL_COMPONENTS + [
    "exposed_time_us",
]

PAYLOAD_FIELDS = [
    "policy", "lora_rank", "batch_size",
    "promotion_weight_bytes", "execution_activation_bytes",
    "execution_residual_bytes", "payload_ratio",
]

AGG_IDENTITY_FIELDS = ["policy", "lora_rank", "batch_size", "n_repeats"]
AGG_NUMERIC_FIELDS = [
    "service_time_us", "component_sum_us", "residual_us", "exposed_time_us",
] + ALL_COMPONENTS
AGG_FIELDS = AGG_IDENTITY_FIELDS + [
    f"{f}_{stat}"
    for f in AGG_NUMERIC_FIELDS
    for stat in ("median", "mad")  # median + MAD for robustness
]


def _write_aggregated_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_recovery_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECOVERY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_payload_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAYLOAD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P2 Cold Miss Recovery Microbenchmark sweep runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/colora_kpi/microbench_config.yaml"),
        help="Path to microbench_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default=None,
        help="Comma-separated rank list (overrides config), e.g. 16,64",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch list (overrides config), e.g. 1,4,8",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Comma-separated policy list (overrides config), e.g. cpu_first,load_then_run",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override number of repeats from config",
    )
    parser.add_argument(
        "--overlap-benchmark",
        action="store_true",
        default=False,
        help="Run optional overlap benchmark for exposed-time measurement",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MicrobenchConfig.from_yaml(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.ranks:
        config.ranks = [int(x) for x in args.ranks.split(",")]
    if args.batch_sizes:
        config.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if args.repeats is not None:
        config.repeats = args.repeats

    print("=" * 72)
    print("P2 Cold Miss Recovery Microbenchmark")
    print("=" * 72)
    print(f"  ranks:          {config.ranks}")
    print(f"  batch_sizes:    {config.batch_sizes}")
    print(f"  policies:       cpu_first, load_then_run")
    print(f"  repeats:        {config.repeats}")
    print(f"  hidden_size:    {config.hidden_size}")
    print(f"  dtype:          {config.dtype}")
    print(f"  output_dir:     {config.output_dir}")
    print("=" * 72)

    run_sweep(config, overlap_benchmark=args.overlap_benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
