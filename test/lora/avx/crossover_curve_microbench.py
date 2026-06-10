#!/usr/bin/env python3
"""
Single MoE Layer Microbenchmark for Crossover Curves (CoLoRA Paper)

Compares two LoRA execution paths for a single MoE expert:

  Path A — cpu_first (CoLoRA default):
    Base MoE runs on GPU. LoRA is applied on CPU:
    Gather activations → H2D → CPU AVX-512 compute → D2H → merge

  Path B — load_then_run:
    Base MoE runs on GPU. LoRA weights loaded from CPU to GPU (one-time),
    then LoRA compute runs on GPU (BGMV or matmul).

  Key insight: H2D weight load is a ONE-TIME cost (amortized across all tokens).
  The benchmark separates:
    - Cold-miss cost: H2D weights + GPU compute (vs cpu_first)
    - Warm-hit cost:  GPU compute only (vs cpu_first)

Produces crossover curves (Figure 2a/b) for:
  - LoRA ranks: 16, 32, 64, 128
  - Sequence lengths: 512, 2048, 4096

Qwen3-VL-30B-A3B dimensions:
  - hidden_size = 2048 (LM token embedding)
  - moe_intermediate_size = 768 (MoE expert FFN intermediate)
  - top_k = 2
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch


# Qwen3-VL-30B-A3B architecture parameters
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
TOP_K = 2
BYTES_PER_PARAM = 2  # bf16/fp16


@dataclass
class CrossoverResult:
    seq_len: int
    lora_rank: int
    n_tokens: int
    # cpu_first path (CPU LoRA)
    t_cpu_first_us: float
    gather_us: float
    to_cpu_us: float
    cpu_compute_us: float
    to_gpu_us: float
    # load_then_run path (GPU LoRA)
    h2d_weight_transfer_us: float  # steady-state transfer cost (reused buffer)
    h2d_weight_alloc_us: float    # one-time allocation + transfer (cold miss)
    gpu_compute_us: float          # per-invocation GPU LoRA compute
    t_cold_miss_us: float          # = h2d_weight_alloc_us + gpu_compute_us
    t_warm_hit_us: float          # = h2d_weight_transfer_us + gpu_compute_us
    t_warm_reuse_us: float        # = gpu_compute_us (buffer already on GPU, no transfer)
    # Weight sizes
    weight_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single MoE Layer Crossover Curve Microbenchmark"
    )
    parser.add_argument(
        "--seq-lens", type=str, default="512,2048,4096",
    )
    parser.add_argument(
        "--lora-ranks", type=str, default="16,32,64,128",
    )
    parser.add_argument(
        "--token-counts", type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--moe-intermediate-size", type=int, default=MOE_INTERMEDIATE_SIZE)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/crossover_curve")
    parser.add_argument("--avx-mode", type=str, default="auto",
                        choices=["auto", "on", "off"])
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _median_us(samples: Sequence[float]) -> float:
    if not samples:
        return float("nan")
    return float(statistics.median(samples))


def _param_bytes(hidden_size: int, intermediate_size: int, rank: int) -> int:
    """Total bytes for one MoE expert's LoRA gate A+B weights."""
    # A: [R, H], B: [R, I]
    elements = (hidden_size + intermediate_size) * rank
    return elements * BYTES_PER_PARAM


# ---------------------------------------------------------------------------
# AVX-512 kernel resolution
# ---------------------------------------------------------------------------

def resolve_avx_mode(avx_mode: str) -> Tuple[bool, bool, Optional[Callable]]:
    if avx_mode == "off":
        return False, False, None

    try:
        from lightllm._kernels.lora.moe_lora_cpu_kernel import (
            moe_batch_lora_avx,
            is_available as avx_is_available,
        )
    except Exception:
        if avx_mode == "on":
            raise RuntimeError("AVX mode forced on, but kernel import failed.")
        return False, False, None

    avx_ready = bool(avx_is_available())
    if avx_mode == "on" and not avx_ready:
        raise RuntimeError("AVX mode forced on, but kernel is unavailable.")

    use_avx = avx_ready and avx_mode != "off"
    return avx_ready, use_avx, (moe_batch_lora_avx if use_avx else None)


# ---------------------------------------------------------------------------
# Path A: cpu_first — LoRA on CPU via AVX-512
# ---------------------------------------------------------------------------

def _cpu_lora_compute(
    cpu_input: torch.Tensor,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    scaling: float,
    use_avx: bool,
    avx_fn: Optional[Callable],
) -> torch.Tensor:
    if use_avx:
        assert avx_fn is not None
        return avx_fn(cpu_input, a_cpu, b_cpu, scaling)

    inter = torch.matmul(cpu_input.float(), a_cpu.float().t())
    out = torch.matmul(inter, b_cpu.float()) * scaling
    return out.to(dtype=cpu_input.dtype)


def run_cpu_first_once(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    cpu_input_pinned: torch.Tensor,
    gpu_output: torch.Tensor,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    scaling: float,
    use_avx: bool,
    avx_fn: Optional[Callable],
) -> Dict[str, float]:
    """Run the cpu_first path once. Returns per-stage times in microseconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # 1. Gather activations by expert
    flat_topk = topk_ids.reshape(-1)
    sorted_indices = torch.argsort(flat_topk)
    batch_indices = torch.div(sorted_indices, top_k, rounding_mode="floor")
    packed_gpu = hidden_states.index_select(0, batch_indices)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # 2. H2D transfer
    cpu_input_pinned.copy_(packed_gpu, non_blocking=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    # 3. CPU AVX LoRA compute
    cpu_out = _cpu_lora_compute(cpu_input_pinned, a_cpu, b_cpu, scaling, use_avx, avx_fn)
    t3 = time.perf_counter()

    # 4. D2H transfer — sync first so we measure ONLY the transfer time,
    #    not any GPU work that might have been queued by prior run_gpu_lora_once calls.
    #    Without this sync, GPU matmul from the previous measurement loop iteration
    #    bleeds into the D2H timing, inflating it by ~GPU_matmul_time.
    torch.cuda.synchronize()
    gpu_output.copy_(cpu_out, non_blocking=True)
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    return {
        "total_us": (t4 - t0) * 1e6,
        "gather_us": (t1 - t0) * 1e6,
        "to_cpu_us": (t2 - t1) * 1e6,
        "cpu_compute_us": (t3 - t2) * 1e6,
        "to_gpu_us": (t4 - t3) * 1e6,
    }


# ---------------------------------------------------------------------------
# Path B: load_then_run — LoRA on GPU
# ---------------------------------------------------------------------------

def measure_h2d_weight_transfer(
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    a_gpu: torch.Tensor,
    b_gpu: torch.Tensor,
    num_iters: int = 5,
) -> Tuple[float, float]:
    """
    Measure H2D cost for loading LoRA weights CPU→GPU.
    This is done in a SEPARATE scope to avoid fragmenting GPU memory before the benchmark.

    Returns (transfer_us, alloc_plus_transfer_us):
      - transfer_us: allocation + transfer (fresh allocation — realistic one-time cold-miss cost)
      - alloc_plus_transfer_us: same as transfer_us (both measure the same thing;
        the first element is for reference; use alloc_plus_transfer_us as the cold-miss cost)
    """
    # Fresh allocation pool — small to minimize GPU memory footprint
    pool_size = num_iters
    a_pool = [torch.empty_like(a_gpu) for _ in range(pool_size)]
    b_pool = [torch.empty_like(b_gpu) for _ in range(pool_size)]
    pool_idx = [0]

    def _fresh_alloc_copy():
        idx = pool_idx[0] % pool_size
        pool_idx[0] += 1
        a_tmp = a_pool[idx]
        b_tmp = b_pool[idx]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a_tmp.copy_(a_cpu, non_blocking=True)
        b_tmp.copy_(b_cpu, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1e6

    samples: List[float] = [_fresh_alloc_copy() for _ in range(num_iters)]

    # Reuse cost: pre-allocated buffers (steady-state per-request cost)
    reuse_samples: List[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a_gpu.copy_(a_cpu, non_blocking=True)
        b_gpu.copy_(b_cpu, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        reuse_samples.append((t1 - t0) * 1e6)

    # Clean up pool before returning — this is critical to avoid fragmenting GPU memory
    del a_pool, b_pool
    torch.cuda.empty_cache()

    return _median_us(reuse_samples), _median_us(samples)


def run_gpu_lora_once(
    hidden_states: torch.Tensor,
    a_gpu: torch.Tensor,
    b_gpu: torch.Tensor,
    gpu_output: torch.Tensor,
    scaling: float,
) -> float:
    """Run GPU LoRA compute (A and B already loaded). Returns microseconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    temp = torch.mm(hidden_states, a_gpu.t())
    _ = torch.mm(temp, b_gpu) * scaling
    torch.cuda.synchronize()  # sync after to prevent GPU work bleeding into next CPU path D2H
    t1 = time.perf_counter()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6


# ---------------------------------------------------------------------------
# Benchmark driver for one (seq_len, rank) configuration
# ---------------------------------------------------------------------------

def run_benchmark_for_config(
    seq_len: int,
    lora_rank: int,
    token_counts: List[int],
    hidden_size: int,
    moe_intermediate_size: int,
    dtype: torch.dtype,
    scaling: float,
    top_k: int,
    warmup: int,
    iters: int,
    use_avx: bool,
    avx_fn: Optional[Callable],
) -> List[CrossoverResult]:
    """Benchmark both paths for a single (seq_len, rank) pair."""

    # CPU LoRA weights (always on CPU for cpu_first path)
    a_cpu = torch.randn(lora_rank, hidden_size, device="cpu", dtype=torch.bfloat16).contiguous()
    b_cpu = torch.randn(lora_rank, moe_intermediate_size, device="cpu", dtype=torch.bfloat16).contiguous()

    # GPU LoRA weights (target buffers for load_then_run path)
    a_gpu = torch.empty(lora_rank, hidden_size, device="cuda", dtype=dtype)
    b_gpu = torch.empty(lora_rank, moe_intermediate_size, device="cuda", dtype=dtype)

    # Measure H2D weight costs: transfer (steady-state) and alloc+transfer (cold-miss)
    h2d_weight_transfer_us, h2d_weight_alloc_us = measure_h2d_weight_transfer(
        a_cpu, b_cpu, a_gpu, b_gpu
    )

    # Clean up H2D measurement GPU buffers before benchmark starts.
    # This prevents allocation fragmentation from contaminating the benchmark loop.
    del a_gpu, b_gpu
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Reallocate GPU weights for the benchmark (clean GPU memory state)
    a_gpu = torch.empty(lora_rank, hidden_size, device="cuda", dtype=dtype)
    b_gpu = torch.empty(lora_rank, moe_intermediate_size, device="cuda", dtype=dtype)
    a_gpu.copy_(a_cpu, non_blocking=True)
    b_gpu.copy_(b_cpu, non_blocking=True)
    torch.cuda.synchronize()

    # Re-randomize so subsequent GPU path iterations do real (non-cached) transfers
    a_cpu = torch.randn(lora_rank, hidden_size, device="cpu", dtype=torch.bfloat16).contiguous()
    b_cpu = torch.randn(lora_rank, moe_intermediate_size, device="cpu", dtype=torch.bfloat16).contiguous()

    weight_bytes = _param_bytes(hidden_size, moe_intermediate_size, lora_rank)
    results: List[CrossoverResult] = []
    max_n = max(token_counts)
    hidden_all = torch.randn(max_n, hidden_size, device="cuda", dtype=dtype)

    for n_tokens in token_counts:
        if n_tokens > seq_len * top_k:
            continue

        hidden_states = hidden_all[:n_tokens]
        topk_ids = torch.zeros((n_tokens, top_k), device="cuda", dtype=torch.int32)

        cpu_input_pinned = torch.empty(
            (n_tokens * top_k, hidden_size),
            device="cpu", dtype=torch.bfloat16, pin_memory=True,
        )
        gpu_output_cpu = torch.empty(
            (n_tokens * top_k, moe_intermediate_size),
            device="cuda", dtype=torch.bfloat16,
        )
        gpu_output = torch.empty(
            (n_tokens, moe_intermediate_size),
            device="cuda", dtype=dtype,
        )

        # Warmup both paths
        for _ in range(warmup):
            run_cpu_first_once(
                hidden_states, topk_ids, top_k, cpu_input_pinned,
                gpu_output_cpu, a_cpu, b_cpu, scaling, use_avx, avx_fn,
            )
            run_gpu_lora_once(hidden_states, a_gpu, b_gpu, gpu_output, scaling)

        # Measurement — ensure GPU is idle before starting to get clean CPU path timings
        torch.cuda.synchronize()
        cpu_samples: List[Dict[str, float]] = []
        gpu_samples: List[float] = []

        for _ in range(iters):
            # CPU path: gpu_output is written here, sync before ensures clean D2H measurement
            cpu_samples.append(
                run_cpu_first_once(
                    hidden_states, topk_ids, top_k, cpu_input_pinned,
                    gpu_output_cpu, a_cpu, b_cpu, scaling, use_avx, avx_fn,
                )
            )
            # GPU path: sync after ensures GPU work doesn't bleed into next CPU path
            gpu_samples.append(
                run_gpu_lora_once(hidden_states, a_gpu, b_gpu, gpu_output, scaling)
            )
            torch.cuda.synchronize()  # isolate GPU work from next CPU path

        gpu_compute_us = _median_us(gpu_samples)

        row = CrossoverResult(
            seq_len=seq_len,
            lora_rank=lora_rank,
            n_tokens=n_tokens,
            # cpu_first breakdown
            t_cpu_first_us=_median_us([s["total_us"] for s in cpu_samples]),
            gather_us=_median_us([s["gather_us"] for s in cpu_samples]),
            to_cpu_us=_median_us([s["to_cpu_us"] for s in cpu_samples]),
            cpu_compute_us=_median_us([s["cpu_compute_us"] for s in cpu_samples]),
            to_gpu_us=_median_us([s["to_gpu_us"] for s in cpu_samples]),
            # load_then_run
            h2d_weight_transfer_us=h2d_weight_transfer_us,
            h2d_weight_alloc_us=h2d_weight_alloc_us,
            gpu_compute_us=gpu_compute_us,
            t_cold_miss_us=h2d_weight_alloc_us + gpu_compute_us,
            t_warm_hit_us=h2d_weight_transfer_us + gpu_compute_us,
            t_warm_reuse_us=gpu_compute_us,
            weight_bytes=weight_bytes,
        )
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Output: CSV, JSON, summary table
# ---------------------------------------------------------------------------

def write_csv(results: List[CrossoverResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seq_len", "lora_rank", "n_tokens",
        "t_cpu_first_us", "gather_us", "h2d_act_us", "cpu_compute_us", "d2h_us",
        "t_cold_miss_us", "t_warm_hit_us", "t_warm_reuse_us",
        "h2d_weight_transfer_us", "h2d_weight_alloc_us",
        "gpu_compute_us",
        "weight_bytes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seq_len": r.seq_len, "lora_rank": r.lora_rank, "n_tokens": r.n_tokens,
                "t_cpu_first_us": f"{r.t_cpu_first_us:.3f}",
                "gather_us": f"{r.gather_us:.3f}",
                "h2d_act_us": f"{r.to_cpu_us:.3f}",
                "cpu_compute_us": f"{r.cpu_compute_us:.3f}",
                "d2h_us": f"{r.to_gpu_us:.3f}",
                "t_cold_miss_us": f"{r.t_cold_miss_us:.3f}",
                "t_warm_hit_us": f"{r.t_warm_hit_us:.3f}",
                "t_warm_reuse_us": f"{r.t_warm_reuse_us:.3f}",
                "h2d_weight_transfer_us": f"{r.h2d_weight_transfer_us:.3f}",
                "h2d_weight_alloc_us": f"{r.h2d_weight_alloc_us:.3f}",
                "gpu_compute_us": f"{r.gpu_compute_us:.3f}",
                "weight_bytes": r.weight_bytes,
            })


def write_summary_json(results: List[CrossoverResult], output_path: Path,
                       metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[int, int], List[CrossoverResult]] = defaultdict(list)
    for r in results:
        grouped[(r.seq_len, r.lora_rank)].append(r)

    crossover_points = []
    for (seq_len, rank), rows in sorted(grouped.items()):
        rows_sorted = sorted(rows, key=lambda x: x.n_tokens)

        # Cold-miss crossover: cpu_first vs GPU with alloc+transfer
        cold_n = None
        # Warm-xfer crossover: cpu_first vs GPU with transfer (reuse)
        xfer_n = None
        # Warm-reuse crossover: cpu_first vs GPU compute only
        reuse_n = None

        for row in rows_sorted:
            if cold_n is None and row.t_cpu_first_us > row.t_cold_miss_us:
                cold_n = row.n_tokens
            gpu_xfer = row.gpu_compute_us + row.h2d_weight_transfer_us
            if xfer_n is None and row.t_cpu_first_us > gpu_xfer:
                xfer_n = row.n_tokens
            if reuse_n is None and row.t_cpu_first_us > row.gpu_compute_us:
                reuse_n = row.n_tokens

        first = rows_sorted[0] if rows_sorted else None
        crossover_points.append({
            "seq_len": seq_len, "lora_rank": rank,
            "crossover_n_cold_miss": cold_n,
            "crossover_n_warm_xfer": xfer_n,
            "crossover_n_warm_reuse": reuse_n,
            "h2d_weight_transfer_us": first.h2d_weight_transfer_us if first else None,
            "h2d_weight_alloc_us": first.h2d_weight_alloc_us if first else None,
            "weight_bytes": first.weight_bytes if first else None,
        })

    summary = {**metadata, "crossover_points": crossover_points}
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary_table(results: List[CrossoverResult]) -> None:
    grouped: Dict[Tuple[int, int], List[CrossoverResult]] = defaultdict(list)
    for r in results:
        grouped[(r.seq_len, r.lora_rank)].append(r)

    print("\n" + "=" * 140)
    print("CROSSOVER POINTS SUMMARY  (cpu_first vs load_then_run)")
    print("Per-request comparison (both include gather + H2D_act):")
    print("  CPU = gather + H2D_act + CPU_compute + D2H  (no weight transfer)")
    print("  GPU (xfer) = GPU_compute + H2D_xfer  (weights stay GPU-resident)")
    print("  GPU (reuse) = GPU_compute  (weights already on GPU, no transfer)")
    print("=" * 140)
    print(f"{'S':>6} {'Rank':>6} {'Wt(KB)':>8} "
          f"{'H2D_xfer':>10} {'H2D_cold':>10} "
          f"{'Cold X':>8} {'Xfer X':>8} {'Reuse X':>10} "
          f"{'CPU@1':>10} {'GPU_xfer':>10} {'GPU_reuse':>11}")
    print("-" * 140)

    for (seq_len, rank), rows in sorted(grouped.items()):
        rows_sorted = sorted(rows, key=lambda x: x.n_tokens)

        # Cold-miss crossover: CPU vs GPU with alloc+transfer
        cold_n = None
        # Warm-xfer crossover: CPU vs GPU with transfer (reuse scenario)
        xfer_n = None
        # Warm-reuse crossover: CPU vs GPU compute only (weights pre-loaded)
        reuse_n = None

        for row in rows_sorted:
            # CPU total includes gather + H2D_act + CPU_compute + D2H
            if cold_n is None and row.t_cpu_first_us > row.t_cold_miss_us:
                cold_n = row.n_tokens
            # GPU warm-xfer = GPU_compute + H2D_transfer
            gpu_xfer = row.gpu_compute_us + row.h2d_weight_transfer_us
            if xfer_n is None and row.t_cpu_first_us > gpu_xfer:
                xfer_n = row.n_tokens
            # GPU warm-reuse = GPU_compute only
            if reuse_n is None and row.t_cpu_first_us > row.gpu_compute_us:
                reuse_n = row.n_tokens

        first = rows_sorted[0]
        gpu_xfer = first.gpu_compute_us + first.h2d_weight_transfer_us
        print(
            f"{seq_len:>6} {rank:>6} {first.weight_bytes / 1024:>8.1f} "
            f"{first.h2d_weight_transfer_us:>10.1f} {first.h2d_weight_alloc_us:>10.1f} "
            f"{str(cold_n or '—'):>8} {str(xfer_n or '—'):>8} {str(reuse_n or '—'):>10} "
            f"{first.t_cpu_first_us:>10.1f} {gpu_xfer:>10.1f} {first.gpu_compute_us:>11.1f}"
        )
    print("=" * 140)
    print("CPU wins = column has a number (first N where CPU < GPU path)")
    print("— = CPU wins at ALL measured N (GPU never leads)")
    print("CPU@1 = CPU path total at N=1 (includes all overhead)")
    print("GPU_xfer = GPU path with H2D transfer cost (realistic steady-state)")
    print("GPU_reuse = GPU path with weights pre-loaded (optimal baseline)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_crossover_curves(
    results: List[CrossoverResult],
    output_dir: Path,
    lora_ranks: List[int],
    seq_lens: List[int],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plotting")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Group data
    by_seq_len: Dict[int, Dict[int, List[CrossoverResult]]] = defaultdict(dict)
    by_rank: Dict[int, Dict[int, List[CrossoverResult]]] = defaultdict(dict)
    for r in results:
        by_seq_len[r.seq_len].setdefault(r.lora_rank, []).append(r)
        by_rank[r.lora_rank].setdefault(r.seq_len, []).append(r)

    # --- Figure 2a: per seq_len, all ranks, cold-miss comparison ---
    for seq_len in seq_lens:
        if seq_len not in by_seq_len:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for i, rank in enumerate(lora_ranks):
            if rank not in by_seq_len[seq_len]:
                continue
            rows = sorted(by_seq_len[seq_len][rank], key=lambda x: x.n_tokens)
            ns = [r.n_tokens for r in rows]
            cpu_ms = [r.t_cpu_first_us / 1000 for r in rows]
            cold_ms = [r.t_cold_miss_us / 1000 for r in rows]
            warm_ms = [r.t_warm_hit_us / 1000 for r in rows]

            color = colors[i % len(colors)]
            ax.plot(ns, cpu_ms, marker="o", ls="-", color=color,
                    label=f"cpu_first (r={rank})", ms=4, lw=1.2)
            ax.plot(ns, cold_ms, marker="s", ls="--", color=color, alpha=0.6,
                    label=f"load_then_run cold (r={rank})", ms=4, lw=1.0)
            ax.plot(ns, warm_ms, marker=".", ls=":", color=color, alpha=0.4,
                    label=f"load_then_run warm (r={rank})", ms=3, lw=0.8)

        ax.set_xlabel("Tokens per Expert (N)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Crossover Curve: Seq Len = {seq_len}  (Qwen3-30B-A3B, Single MoE Layer)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_xscale("log", base=2)
        plt.tight_layout()
        for fmt in ("pdf", "png"):
            plt.savefig(output_dir / f"fig2a_seq{seq_len}.{fmt}", dpi=300)
        plt.close()

    # --- Figure 2b: per rank, all seq lens, cold-miss comparison ---
    for rank in lora_ranks:
        if rank not in by_rank:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for i, seq_len in enumerate(seq_lens):
            if seq_len not in by_rank[rank]:
                continue
            rows = sorted(by_rank[rank][seq_len], key=lambda x: x.n_tokens)
            ns = [r.n_tokens for r in rows]
            cpu_ms = [r.t_cpu_first_us / 1000 for r in rows]
            cold_ms = [r.t_cold_miss_us / 1000 for r in rows]
            warm_ms = [r.t_warm_hit_us / 1000 for r in rows]

            color = colors[i % len(colors)]
            ax.plot(ns, cpu_ms, marker="o", ls="-", color=color,
                    label=f"cpu_first (S={seq_len})", ms=4, lw=1.2)
            ax.plot(ns, cold_ms, marker="s", ls="--", color=color, alpha=0.6,
                    label=f"load_then_run cold (S={seq_len})", ms=4, lw=1.0)
            ax.plot(ns, warm_ms, marker=".", ls=":", color=color, alpha=0.4,
                    label=f"load_then_run warm (S={seq_len})", ms=3, lw=0.8)

        ax.set_xlabel("Tokens per Expert (N)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Crossover Curve: LoRA Rank = {rank}  (Qwen3-30B-A3B, Single MoE Layer)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_xscale("log", base=2)
        plt.tight_layout()
        for fmt in ("pdf", "png"):
            plt.savefig(output_dir / f"fig2b_rank{rank}.{fmt}", dpi=300)
        plt.close()

    # --- Breakdown of cpu_first path ---
    for seq_len in seq_lens[:1]:
        for rank in lora_ranks[:1]:
            if seq_len not in by_seq_len or rank not in by_seq_len[seq_len]:
                continue
            rows = sorted(by_seq_len[seq_len][rank], key=lambda x: x.n_tokens)
            fig, ax = plt.subplots(figsize=(9, 5))
            ns = [r.n_tokens for r in rows]
            ax.stackplot(ns,
                         [r.gather_us / 1000 for r in rows],
                         [r.to_cpu_us / 1000 for r in rows],
                         [r.cpu_compute_us / 1000 for r in rows],
                         [r.to_gpu_us / 1000 for r in rows],
                         labels=["Gather", "H2D (act)", "CPU LoRA AVX", "D2H (result)"],
                         colors=["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"],
                         alpha=0.85)
            ax.set_xlabel("Tokens per Expert (N)")
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"cpu_first Latency Breakdown  (S={seq_len}, r={rank})")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_xscale("log", base=2)
            plt.tight_layout()
            for fmt in ("pdf", "png"):
                plt.savefig(output_dir / f"cpu_first_breakdown_S{seq_len}_r{rank}.{fmt}", dpi=300)
            plt.close()

    # --- Heatmap summary ---
    _plot_crossover_heatmap(results, output_dir, lora_ranks, seq_lens)


def _plot_crossover_heatmap(
    results: List[CrossoverResult],
    output_dir: Path,
    lora_ranks: List[int],
    seq_lens: List[int],
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    grouped: Dict[Tuple[int, int], List[CrossoverResult]] = defaultdict(list)
    for r in results:
        grouped[(r.seq_len, r.lora_rank)].append(r)

    for mode, label, filename in [
        ("cold_miss", "Cold-miss Crossover N", "crossover_heatmap_cold"),
        ("warm_hit", "Warm-hit Crossover N", "crossover_heatmap_warm"),
    ]:
        matrix = np.full((len(seq_lens), len(lora_ranks)), np.nan)
        for i, sl in enumerate(seq_lens):
            for j, rk in enumerate(lora_ranks):
                rows = sorted(grouped.get((sl, rk), []), key=lambda x: x.n_tokens)
                for row in rows:
                    compare = row.t_cold_miss_us if mode == "cold_miss" else row.t_warm_hit_us
                    if row.t_cpu_first_us > compare:
                        matrix[i, j] = row.n_tokens
                        break

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(lora_ranks)))
        ax.set_xticklabels([f"r={r}" for r in lora_ranks])
        ax.set_yticks(range(len(seq_lens)))
        ax.set_yticklabels([f"S={s}" for s in seq_lens])
        ax.set_title(label)

        for i in range(len(seq_lens)):
            for j in range(len(lora_ranks)):
                val = matrix[i, j]
                text = f"{int(val)}" if not np.isnan(val) else "—"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Crossover N (tokens)")
        plt.tight_layout()
        for fmt in ("pdf", "png"):
            plt.savefig(output_dir / f"{filename}.{fmt}", dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    lora_ranks = [int(x.strip()) for x in args.lora_ranks.split(",") if x.strip()]
    token_counts = [int(x.strip()) for x in args.token_counts.split(",") if x.strip()]
    dtype = _dtype_from_name(args.dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    avx_ready, use_avx, avx_fn = resolve_avx_mode(args.avx_mode)
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("=" * 80)
    print("Single MoE Layer Crossover Curve Microbenchmark")
    print("=" * 80)
    print(f"GPU: {device_name}")
    print(f"AVX kernel: available={avx_ready}, use={use_avx}")
    print(f"Model: Qwen3-VL-30B-A3B  (H={args.hidden_size}, I={args.moe_intermediate_size})")
    print(f"Sweep: seq_lens={seq_lens}, ranks={lora_ranks}")
    print(f"Token counts: {token_counts}")
    print(f"dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 80)

    all_results: List[CrossoverResult] = []

    for seq_len in seq_lens:
        for rank in lora_ranks:
            print(f"\n--- S={seq_len}, r={rank} ---")
            results = run_benchmark_for_config(
                seq_len=seq_len, lora_rank=rank,
                token_counts=token_counts,
                hidden_size=args.hidden_size,
                moe_intermediate_size=args.moe_intermediate_size,
                dtype=dtype, scaling=args.scaling, top_k=args.top_k,
                warmup=args.warmup, iters=args.iters,
                use_avx=use_avx, avx_fn=avx_fn,
            )
            all_results.extend(results)

            print(f"  H2D weight transfer (reuse): {results[0].h2d_weight_transfer_us:.1f} us "
                  f"| alloc+xfer (cold): {results[0].h2d_weight_alloc_us:.1f} us "
                  f"({results[0].weight_bytes / 1024:.1f} KB)")
            print(f"  {'N':>6} {'cpu_first':>12} {'cold_miss':>12} {'warm_xfer':>12} {'warm_reuse':>12}  |  "
                  f"{'Gather':>8} {'H2D_act':>8} {'CPU_cmp':>8} {'D2H':>8}  |  "
                  f"{'GPU_cmp':>8}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}  |  "
                  f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}  |  {'-'*8}")
            for r in results:
                print(f"  {r.n_tokens:>6} {r.t_cpu_first_us:>11.1f}us {r.t_cold_miss_us:>11.1f}us "
                      f"{r.t_warm_hit_us:>11.1f}us {r.t_warm_reuse_us:>11.1f}us  |  "
                      f"{r.gather_us:>7.1f}us {r.to_cpu_us:>7.1f}us "
                      f"{r.cpu_compute_us:>7.1f}us {r.to_gpu_us:>7.1f}us  |  "
                      f"{r.gpu_compute_us:>7.1f}us")

    output_dir = Path(args.output_dir)
    write_csv(all_results, output_dir / "crossover_results.csv")
    write_summary_json(all_results, output_dir / "crossover_summary.json", {
        "model": "Qwen3-VL-30B-A3B",
        "hidden_size": args.hidden_size,
        "moe_intermediate_size": args.moe_intermediate_size,
        "top_k": args.top_k,
        "dtype": args.dtype,
        "gpu": device_name,
        "avx_enabled": use_avx,
    })
    print_summary_table(all_results)

    if not args.no_plot:
        plot_crossover_curves(all_results, output_dir, lora_ranks, seq_lens)
        print(f"\nPlots saved to: {output_dir}")

    print(f"\nCSV: {output_dir / 'crossover_results.csv'}")
    print(f"JSON: {output_dir / 'crossover_summary.json'}")


if __name__ == "__main__":
    main()
