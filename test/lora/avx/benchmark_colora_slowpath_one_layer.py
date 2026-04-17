#!/usr/bin/env python3
"""
One-layer COLoRA slow-path microbenchmark (real hybrid path).

This benchmark replays one MoE layer decode-style flow with:
1) Base model compute (QKV/O projection + router top-k simulation).
2) COLoRA slow-path LoRA compute via _batch_apply_moe_lora_hybrid.

It is designed to align with the NVTX flow shown in the profiler figure.
"""

import argparse
import collections
import time
from typing import Dict, List, NamedTuple

import torch


class PhaseSpec(NamedTuple):
    name: str
    description: str


def resolve_measurement_policy(hit_rate: float) -> Dict[str, bool]:
    """Map hit-rate to stable benchmark residency policy."""
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate must be in [0, 1], got {hit_rate}")
    if hit_rate <= 0.0:
        return {
            "prepromote_hit_adapter": False,
            "disable_measurement_promotion": True,
        }
    if hit_rate >= 1.0:
        return {
            "prepromote_hit_adapter": True,
            "disable_measurement_promotion": False,
        }
    return {
        "prepromote_hit_adapter": True,
        "disable_measurement_promotion": True,
    }


def build_phase_plan() -> List[PhaseSpec]:
    """Execution phases aligned with the figure's slow-path COLoRA flow."""
    return [
        PhaseSpec("COLoRA_Hybrid_Prepare", "apply promotions + valid token view + key build"),
        PhaseSpec("COLoRA_CacheLookupAndPolicy", "cache lookup and miss policy"),
        PhaseSpec("COLoRA_BuildHitMissMasks", "construct hit/miss token masks"),
        PhaseSpec("COLoRA_HitPath_Prepare", "group hit tokens and gather cache slots"),
        PhaseSpec("COLoRA_GPU_Hit_Path", "run GPU hit-path BGMV"),
        PhaseSpec("COLoRA_MissPath_CheckAndPrepare", "prepare miss-side CPU fallback"),
        PhaseSpec("COLoRA_MissPath_ExecuteAndCommit", "execute CPU miss path and write back"),
        PhaseSpec("COLoRA_PostCompute_StatsFinalize", "finalize per-call COLoRA stats"),
    ]


def build_bins_for_hit_rate(
    batch_size: int,
    hit_rate: float,
    hit_adapter_id: int,
    miss_adapter_id: int,
    device: str,
) -> torch.Tensor:
    """Build adapter bins to control hit/miss token composition per batch."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate must be in [0, 1], got {hit_rate}")

    hit_tokens = int(round(batch_size * hit_rate))
    miss_tokens = batch_size - hit_tokens
    values = [int(hit_adapter_id)] * hit_tokens + [int(miss_adapter_id)] * miss_tokens
    return torch.tensor(values, dtype=torch.long, device=device)


class NvtxPhaseTracer:
    """Intercept NVTX push/pop to collect inclusive phase durations."""

    def __init__(self) -> None:
        self._stack: List[tuple[str, float]] = []
        self.totals_sec: Dict[str, float] = collections.defaultdict(float)
        self.counts: Dict[str, int] = collections.defaultdict(int)
        self._orig_push = None
        self._orig_pop = None
        self._nvtx_mod = None

    def __enter__(self):
        from lightllm.utils import nvtx_utils

        self._nvtx_mod = nvtx_utils
        self._orig_push = nvtx_utils._range_push
        self._orig_pop = nvtx_utils._range_pop

        def _wrapped_push(msg: str, color=None):
            self._stack.append((str(msg), time.perf_counter()))
            return self._orig_push(msg, color)

        def _wrapped_pop():
            end_ts = time.perf_counter()
            if self._stack:
                msg, start_ts = self._stack.pop()
                self.totals_sec[msg] += max(end_ts - start_ts, 0.0)
                self.counts[msg] += 1
            return self._orig_pop()

        nvtx_utils._range_push = _wrapped_push
        nvtx_utils._range_pop = _wrapped_pop
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._nvtx_mod is not None:
            self._nvtx_mod._range_push = self._orig_push
            self._nvtx_mod._range_pop = self._orig_pop


def _simulate_base_one_layer(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    w_router: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    from lightllm.utils.nvtx_utils import NvtxAnnotate

    with NvtxAnnotate("Qwen3VL_QKV"):
        qkv = torch.matmul(x, w_qkv)
        if qkv.device.type == "cuda":
            torch.cuda.synchronize()

    with NvtxAnnotate("Qwen3VL_O_Proj"):
        proj = torch.matmul(qkv, w_o)
        if proj.device.type == "cuda":
            torch.cuda.synchronize()

    with NvtxAnnotate("MoE_SlowPath_TopKRouting"):
        router_logits = torch.matmul(proj, w_router)
        _ = torch.topk(router_logits, k=min(topk, router_logits.shape[1]), dim=-1)
        if proj.device.type == "cuda":
            torch.cuda.synchronize()

    return proj


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _format_us(value_sec: float, denom: int) -> float:
    return (value_sec / max(denom, 1)) * 1e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-layer COLoRA slow-path microbenchmark (base + LoRA hybrid)"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--hit-rate", type=float, default=0.5)
    parser.add_argument("--projection", type=str, default="gate", choices=["gate", "up", "down"])
    parser.add_argument("--miss-policy", type=str, default="cpu_first")
    parser.add_argument("--cache-budget-mb", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument("--hit-adapter-id", type=int, default=0)
    parser.add_argument("--miss-adapter-id", type=int, default=1)
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> int:
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Approach A benchmark but is not available.")
    if not BGMV_AVAILABLE:
        raise RuntimeError("BGMV kernel is required for GPU hit-path but is unavailable.")

    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    layer_id = 0
    expert_id = 0
    measurement_policy = resolve_measurement_policy(args.hit_rate)

    pool = LoRAModulePool.create(
        pool_size=32,
        max_rank=args.lora_rank,
        input_dim=args.hidden_size,
        output_dim=args.hidden_size,
        dtype=dtype,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    for adapter_idx in (args.hit_adapter_id, args.miss_adapter_id):
        lora_a = torch.randn(args.lora_rank, args.hidden_size, dtype=dtype)
        lora_b = torch.randn(args.lora_rank, args.hidden_size, dtype=dtype)
        ok = pool.load_adapter(
            adapter_idx=int(adapter_idx),
            rank=args.lora_rank,
            scaling=1.0,
            layer_weights={0: {"A": lora_a, "B": lora_b}},
        )
        if not ok:
            raise RuntimeError(f"failed to load adapter {adapter_idx} into LoRA pool")

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=args.cache_budget_mb,
            promote_min_hits=1,
            promote_window=16,
            max_promote_per_step=8,
            decay=0.9,
            miss_policy=str(args.miss_policy),
        )
    )
    cache_mgr.register_projection_pool(args.projection, pool)

    hit_key = ExpertCacheKey(
        projection=args.projection,
        adapter_idx=int(args.hit_adapter_id),
        layer_id=layer_id,
        expert_id=expert_id,
    )
    if measurement_policy["prepromote_hit_adapter"]:
        cache_mgr.record_access([hit_key])
        cache_mgr.schedule_promotion([hit_key])
        cache_mgr.apply_completed_promotions()

    rank_kwargs = {"gate_lora_rank": 0, "up_lora_rank": 0, "down_lora_rank": 0}
    rank_kwargs[f"{args.projection}_lora_rank"] = int(args.lora_rank)
    dispatcher = Qwen3VLMoELoRADispatcher(
        num_layers=1,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        **rank_kwargs,
    )
    dispatcher.expert_cache_manager = cache_mgr

    x = torch.randn(args.batch_size, args.hidden_size, dtype=dtype, device=device)
    bins = build_bins_for_hit_rate(
        batch_size=args.batch_size,
        hit_rate=args.hit_rate,
        hit_adapter_id=args.hit_adapter_id,
        miss_adapter_id=args.miss_adapter_id,
        device=args.device,
    )

    w_qkv = torch.randn(args.hidden_size, args.hidden_size, dtype=dtype, device=device)
    w_o = torch.randn(args.hidden_size, args.hidden_size, dtype=dtype, device=device)
    w_router = torch.randn(args.hidden_size, args.num_experts, dtype=dtype, device=device)

    def _run_once():
        base_out = _simulate_base_one_layer(
            x=x,
            w_qkv=w_qkv,
            w_o=w_o,
            w_router=w_router,
            topk=args.topk,
        )
        if measurement_policy["disable_measurement_promotion"]:
            original_record_access = cache_mgr.record_access
            original_schedule_promotion = cache_mgr.schedule_promotion
            try:
                cache_mgr.record_access = lambda keys: None
                cache_mgr.schedule_promotion = lambda keys: None
                lora_out = dispatcher._batch_apply_moe_lora_hybrid(
                    input_tensor=base_out,
                    layer_id=layer_id,
                    buffer_layer_id=layer_id,
                    pool=pool,
                    bins=bins,
                    projection=args.projection,
                    expert_id=expert_id,
                )
            finally:
                cache_mgr.record_access = original_record_access
                cache_mgr.schedule_promotion = original_schedule_promotion
        else:
            lora_out = dispatcher._batch_apply_moe_lora_hybrid(
                input_tensor=base_out,
                layer_id=layer_id,
                buffer_layer_id=layer_id,
                pool=pool,
                bins=bins,
                projection=args.projection,
                expert_id=expert_id,
            )
        if lora_out.device.type == "cuda":
            torch.cuda.synchronize()
        return dispatcher.pop_colora_stats()

    for _ in range(args.warmup_iters):
        _run_once()

    stats_totals = collections.defaultdict(float)
    with NvtxPhaseTracer() as tracer:
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            stats = _run_once()
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stats_totals[key] += float(value)
        total_elapsed = max(time.perf_counter() - t0, 0.0)

    print("=" * 88)
    print("COLoRA one-layer slow-path microbenchmark (real hybrid path)")
    print("=" * 88)
    print(
        f"batch={args.batch_size} hidden={args.hidden_size} rank={args.lora_rank} "
        f"projection={args.projection} hit_rate={args.hit_rate:.2f} iters={args.iterations}"
    )
    print(
        "flow: Qwen3VL_QKV -> Qwen3VL_O_Proj -> MoE_SlowPath_TopKRouting -> "
        + " -> ".join(phase.name for phase in build_phase_plan())
    )

    phase_names = [
        "Qwen3VL_QKV",
        "Qwen3VL_O_Proj",
        "MoE_SlowPath_TopKRouting",
        *[phase.name for phase in build_phase_plan()],
    ]
    print("\nPer-phase avg latency (us):")
    print("-" * 88)
    for name in phase_names:
        phase_us = _format_us(tracer.totals_sec.get(name, 0.0), args.iterations)
        count = tracer.counts.get(name, 0)
        print(f"{name:<40} {phase_us:>12.3f} us    count={count}")
    print("-" * 88)
    print(f"{'TOTAL_PER_ITER':<40} {_format_us(total_elapsed, args.iterations):>12.3f} us")

    print("\nAveraged COLoRA runtime stats:")
    print("-" * 88)
    stat_keys = [
        "colora_hit_tokens",
        "colora_miss_tokens",
        "cache_hit_rate",
        "gpu_compute_time",
        "cpu_compute_time",
        "cpu_queue_wait_time",
        "d2h_bytes",
        "h2d_bytes",
        "weight_h2d_time",
        "weight_h2d_bytes",
        "overlap_ratio",
        "moe_kernel_calls",
        "moe_kernel_tokens",
    ]
    for key in stat_keys:
        avg_v = stats_totals.get(key, 0.0) / max(args.iterations, 1)
        print(f"{key:<28} {avg_v}")

    return 0


def main() -> int:
    args = parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
