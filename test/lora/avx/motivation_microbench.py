#!/usr/bin/env python3
"""
CoLoRA Motivation Study: Single MoE Layer Microbenchmark

Compares two LoRA miss-recovery paths for decode phase (seq_len=1):

  Path A (GPU-transfer): Transfer LoRA weights CPU→GPU, compute merge on GPU
  Path B (CPU-compute):   Transfer activation GPU→CPU, compute merge on CPU, result→GPU

Sweeps LoRA rank and num_miss_loras to find the crossover point.

Qwen3-VL-30B-A3B MoE layer dimensions:
  - hidden_dim = 2048
  - intermediate_dim = 768
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch


HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 768
BYTES_PER_PARAM = 2  # bf16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoLoRA Motivation Study Microbenchmark")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--intermediate-dim", type=int, default=INTERMEDIATE_DIM)
    p.add_argument("--ranks", type=str, default="16,32,64,128")
    p.add_argument("--num-miss-list", type=str, default="1,2,4,8,16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--output", type=str, default="results/motivation_study.csv")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def _median_us(samples: Sequence[float]) -> float:
    return float(statistics.median(samples))


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def pregenerate_weights(
    ranks: List[int],
    num_miss_list: List[int],
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
) -> dict:
    """Pre-generate all LoRA weights in CPU pinned memory.

    Returns dict keyed by rank, each containing:
      - a_cpu: [max_miss, rank, hidden_dim] pinned bf16
      - b_cpu: [max_miss, rank, intermediate_dim] pinned bf16
    """
    max_miss = max(num_miss_list)
    weights = {}
    for rank in ranks:
        a = torch.randn(max_miss, rank, hidden_dim, device="cpu", dtype=dtype).pin_memory()
        b = torch.randn(max_miss, rank, intermediate_dim, device="cpu", dtype=dtype).pin_memory()
        weights[rank] = {"a": a, "b": b}
    return weights


def measure_path_a(
    hidden_states: torch.Tensor,
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    gpu_out: torch.Tensor,
    scaling: float,
) -> Tuple[float, float, float]:
    """Path A: GPU-transfer.

    Steps:
      1. H2D: transfer all LoRA weights CPU→GPU (batched, one transfer)
      2. GPU compute: for each adapter, compute activation @ A^T @ B^T

    Returns (h2d_us, gpu_compute_us, total_us).
    """
    num_miss = weights_a.shape[0]
    rank = weights_a.shape[1]
    hidden_dim = weights_a.shape[2]
    intermediate_dim = weights_b.shape[2]

    a_gpu = torch.empty_like(weights_a, device="cuda")
    b_gpu = torch.empty_like(weights_b, device="cuda")
    temp = torch.empty(num_miss, 1, rank, device="cuda", dtype=weights_a.dtype)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Step 1: H2D transfer (all weights batched)
    a_gpu.copy_(weights_a, non_blocking=True)
    b_gpu.copy_(weights_b, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Step 2: GPU compute — for each miss adapter: act @ A^T @ B^T
    # activation: [1, H], A: [R, H] → inter: [1, R], B: [R, I] → out: [1, I]
    # Batched: act [1, H] @ A^T [num_miss, H, R] → [num_miss, 1, R]
    # Then: [num_miss, 1, R] @ B [num_miss, R, I] → [num_miss, 1, I]
    a_t = weights_a.transpose(1, 2).to(device="cuda")  # [num_miss, H, R]
    temp = torch.bmm(hidden_states.unsqueeze(0).expand(num_miss, -1, -1), a_t)
    result = torch.bmm(temp, b_gpu) * scaling  # [num_miss, 1, I]
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    h2d_us = (t1 - t0) * 1e6
    gpu_compute_us = (t2 - t1) * 1e6
    total_us = (t2 - t0) * 1e6

    del a_gpu, b_gpu, a_t, temp, result
    return h2d_us, gpu_compute_us, total_us


def measure_path_b(
    hidden_states: torch.Tensor,
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    cpu_act: torch.Tensor,
    gpu_out: torch.Tensor,
    scaling: float,
) -> Tuple[float, float, float, float]:
    """Path B: CPU-compute.

    Steps:
      1. D2H: transfer activation GPU→CPU (just hidden_dim floats)
      2. CPU compute: for each adapter, act @ A^T @ B^T
      3. H2D: transfer results CPU→GPU

    Returns (d2h_us, cpu_compute_us, h2d_us, total_us).
    """
    num_miss = weights_a.shape[0]
    rank = weights_a.shape[1]
    intermediate_dim = weights_b.shape[2]

    # Pre-allocate CPU output buffer
    cpu_out = torch.empty(num_miss, 1, intermediate_dim, device="cpu", dtype=weights_a.dtype).pin_memory()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Step 1: D2H transfer (activation only)
    cpu_act.copy_(hidden_states, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Step 2: CPU compute
    # activation: [1, H], A: [num_miss, R, H] → inter: [num_miss, 1, R]
    # B: [num_miss, R, I] → out: [num_miss, 1, I]
    act_f32 = cpu_act.float()  # [1, H]
    a_f32 = weights_a.float()  # [num_miss, R, H]
    b_f32 = weights_b.float()  # [num_miss, R, I]
    inter = torch.bmm(act_f32.unsqueeze(0).expand(num_miss, -1, -1),
                      a_f32.transpose(1, 2))  # [num_miss, 1, R]
    result = torch.bmm(inter, b_f32) * scaling  # [num_miss, 1, I]
    cpu_out.copy_(result.to(dtype=weights_a.dtype))
    t2 = time.perf_counter()

    # Step 3: H2D transfer (results)
    torch.cuda.synchronize()
    gpu_out.copy_(cpu_out, non_blocking=True)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    d2h_us = (t1 - t0) * 1e6
    cpu_compute_us = (t2 - t1) * 1e6
    h2d_us = (t3 - t2) * 1e6
    total_us = (t3 - t0) * 1e6

    return d2h_us, cpu_compute_us, h2d_us, total_us


def run_benchmark(
    hidden_dim: int,
    intermediate_dim: int,
    ranks: List[int],
    num_miss_list: List[int],
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    device: str,
) -> List[dict]:
    """Run all benchmark configurations."""
    max_miss = max(num_miss_list)
    scaling = 1.0

    # Pre-generate weights
    all_weights = pregenerate_weights(ranks, num_miss_list, hidden_dim, intermediate_dim, dtype)

    # Pre-allocate buffers
    activation_gpu = torch.randn(1, hidden_dim, device=device, dtype=dtype)
    gpu_out_a = torch.empty(max_miss, 1, intermediate_dim, device=device, dtype=dtype)
    gpu_out_b = torch.empty(max_miss, 1, intermediate_dim, device=device, dtype=dtype)
    cpu_act_pinned = torch.empty(1, hidden_dim, device="cpu", dtype=dtype).pin_memory()

    results: List[dict] = []

    for rank in ranks:
        w = all_weights[rank]
        for num_miss in num_miss_list:
            a_cpu = w["a"][:num_miss]  # [num_miss, R, H]
            b_cpu = w["b"][:num_miss]  # [num_miss, R, I]

            # --- Path A warmup ---
            for _ in range(warmup):
                measure_path_a(activation_gpu, a_cpu, b_cpu,
                               gpu_out_a[:num_miss], scaling)
                torch.cuda.synchronize()

            # --- Path A measurement ---
            a_samples = []
            for _ in range(iters):
                h2d, gpu_c, total = measure_path_a(
                    activation_gpu, a_cpu, b_cpu,
                    gpu_out_a[:num_miss], scaling)
                a_samples.append((h2d, gpu_c, total))
                torch.cuda.synchronize()

            h2d_a = _median_us([s[0] for s in a_samples])
            gpu_c_a = _median_us([s[1] for s in a_samples])
            total_a = _median_us([s[2] for s in a_samples])

            # --- Path B warmup ---
            for _ in range(warmup):
                measure_path_b(activation_gpu, a_cpu, b_cpu,
                               cpu_act_pinned, gpu_out_b[:num_miss], scaling)
                torch.cuda.synchronize()

            # --- Path B measurement ---
            b_samples = []
            for _ in range(iters):
                d2h, cpu_c, h2d, total = measure_path_b(
                    activation_gpu, a_cpu, b_cpu,
                    cpu_act_pinned, gpu_out_b[:num_miss], scaling)
                b_samples.append((d2h, cpu_c, h2d, total))
                torch.cuda.synchronize()

            d2h_b = _median_us([s[0] for s in b_samples])
            cpu_c_b = _median_us([s[1] for s in b_samples])
            h2d_b = _median_us([s[2] for s in b_samples])
            total_b = _median_us([s[3] for s in b_samples])

            # Path A row
            results.append({
                "rank": rank,
                "num_miss": num_miss,
                "path": "A",
                "h2d_ms": f"{h2d_a / 1000:.6f}",
                "d2h_ms": "",
                "gpu_compute_ms": f"{gpu_c_a / 1000:.6f}",
                "cpu_compute_ms": "",
                "total_ms": f"{total_a / 1000:.6f}",
            })
            # Path B row
            results.append({
                "rank": rank,
                "num_miss": num_miss,
                "path": "B",
                "h2d_ms": f"{h2d_b / 1000:.6f}",
                "d2h_ms": f"{d2h_b / 1000:.6f}",
                "gpu_compute_ms": "",
                "cpu_compute_ms": f"{cpu_c_b / 1000:.6f}",
                "total_ms": f"{total_b / 1000:.6f}",
            })

            winner = "CPU" if total_b < total_a else "GPU"
            speedup = max(total_a, total_b) / min(total_a, total_b)
            print(f"  r={rank:>3} miss={num_miss:>2} | "
                  f"PathA: H2D={h2d_a:>8.1f}us GPU={gpu_c_a:>8.1f}us total={total_a:>8.1f}us | "
                  f"PathB: D2H={d2h_b:>8.1f}us CPU={cpu_c_b:>8.1f}us H2D={h2d_b:>8.1f}us total={total_b:>8.1f}us | "
                  f"{winner} wins ({speedup:.2f}x)")

    return results


def write_csv(results: List[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "num_miss", "path", "h2d_ms", "d2h_ms",
                  "gpu_compute_ms", "cpu_compute_ms", "total_ms"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults written to: {output_path}")


def print_summary(results: List[dict]) -> None:
    """Print crossover table."""
    from collections import defaultdict
    by_config: dict = defaultdict(dict)
    for r in results:
        by_config[(r["rank"], r["num_miss"])][r["path"]] = float(r["total_ms"])

    ranks = sorted(set(r["rank"] for r in results))
    misses = sorted(set(r["num_miss"] for r in results))

    print("\n" + "=" * 90)
    print("CROSSOVER TABLE (Path B CPU vs Path A GPU)")
    print("  Cell: speedup of winner. B = CPU wins, A = GPU wins.")
    print("=" * 90)
    header = f"{'Rank\\Miss':>10}"
    for m in misses:
        header += f"{m:>12}"
    print(header)
    print("-" * 90)

    for rank in ranks:
        line = f"{rank:>10}"
        for num_miss in misses:
            key = (rank, num_miss)
            if key in by_config and "A" in by_config[key] and "B" in by_config[key]:
                a_time = by_config[key]["A"]
                b_time = by_config[key]["B"]
                speedup = max(a_time, b_time) / min(a_time, b_time)
                winner = "B" if b_time < a_time else "A"
                line += f"  {winner}:{speedup:.1f}x  "
            else:
                line += f"{'N/A':>12}"
        print(line)
    print("=" * 90)
    print("B = CPU-compute (Path B) wins. A = GPU-transfer (Path A) wins.")


def plot_results(results: List[dict], output_dir: str) -> None:
    """Generate crossover heatmap and latency breakdown plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    by_config: dict = defaultdict(dict)
    for r in results:
        by_config[(r["rank"], r["num_miss"])][r["path"]] = {
            "total_ms": float(r["total_ms"]),
            "h2d_ms": float(r["h2d_ms"]) if r["h2d_ms"] else 0,
            "d2h_ms": float(r["d2h_ms"]) if r["d2h_ms"] else 0,
            "gpu_compute_ms": float(r["gpu_compute_ms"]) if r["gpu_compute_ms"] else 0,
            "cpu_compute_ms": float(r["cpu_compute_ms"]) if r["cpu_compute_ms"] else 0,
        }

    ranks = sorted(set(r["rank"] for r in results))
    misses = sorted(set(r["num_miss"] for r in results))

    # Figure 2a: Crossover Heatmap
    matrix_speedup = np.zeros((len(ranks), len(misses)))
    matrix_winner = np.zeros((len(ranks), len(misses)))  # 1 = CPU wins, -1 = GPU wins

    for i, rank in enumerate(ranks):
        for j, num_miss in enumerate(misses):
            key = (rank, num_miss)
            if key in by_config and "A" in by_config[key] and "B" in by_config[key]:
                a_time = by_config[key]["A"]["total_ms"]
                b_time = by_config[key]["B"]["total_ms"]
                speedup = max(a_time, b_time) / min(a_time, b_time)
                matrix_speedup[i, j] = speedup
                matrix_winner[i, j] = 1 if b_time < a_time else -1

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Use RdBu_r: red = GPU wins, blue = CPU wins
    masked = np.ma.masked_where(matrix_winner == 0, matrix_winner)
    im = ax.imshow(masked, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    for i in range(len(ranks)):
        for j in range(len(misses)):
            if matrix_winner[i, j] != 0:
                text = f"{matrix_speedup[i, j]:.1f}x"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(matrix_winner[i, j]) > 0 else "black")

    ax.set_xticks(range(len(misses)))
    ax.set_xticklabels([str(m) for m in misses])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"r={r}" for r in ranks])
    ax.set_xlabel("num_miss_loras")
    ax.set_ylabel("LoRA Rank")
    ax.set_title("Figure 2a: Crossover Heatmap\n(Blue = CPU wins, Red = GPU wins)")
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 1])
    cbar.ax.set_yticklabels(["GPU wins", "CPU wins"])
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        plt.savefig(out / f"fig2a_crossover_heatmap.{fmt}", dpi=300)
    plt.close()

    # Figure 2b: Latency Breakdown (selected configs)
    selected = [(32, 1), (32, 4), (32, 16)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (sel_rank, sel_miss) in enumerate(selected):
        ax = axes[idx]
        key = (sel_rank, sel_miss)
        if key not in by_config:
            continue

        path_a = by_config[key]["A"]
        path_b = by_config[key]["B"]

        # Path A: H2D + GPU compute
        a_components = [path_a["h2d_ms"], path_a["gpu_compute_ms"]]
        # Path B: D2H + CPU compute + H2D
        b_components = [path_b["d2h_ms"], path_b["cpu_compute_ms"], path_b["h2d_ms"]]

        x = [0, 1]
        width = 0.35

        bottom = 0
        colors_a = ["#ff9999", "#ff4444"]
        for comp, color in zip(a_components, colors_a):
            ax.bar(x[0], comp, width, bottom=bottom, color=color, alpha=0.85)
            bottom += comp

        bottom = 0
        colors_b = ["#9999ff", "#4444ff", "#6666cc"]
        labels_b = ["D2H (act)", "CPU compute", "H2D (result)"]
        for comp, color, label in zip(b_components, colors_b, labels_b):
            ax.bar(x[1], comp, width, bottom=bottom, color=color, alpha=0.85, label=label)
            bottom += comp

        ax.set_xticks(x)
        ax.set_xticklabels(["Path A\n(GPU)", "Path B\n(CPU)"])
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"r={sel_rank}, miss={sel_miss}")

    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 2b: Latency Breakdown", fontsize=14)
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        plt.savefig(out / f"fig2b_latency_breakdown.{fmt}", dpi=300)
    plt.close()

    # Figure 2c: Per-LoRA Scaling
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    line_styles = {"A": "--", "B": "-"}

    for i, rank in enumerate(ranks):
        a_times = []
        b_times = []
        miss_vals = []
        for num_miss in misses:
            key = (rank, num_miss)
            if key in by_config:
                a_times.append(by_config[key]["A"]["total_ms"])
                b_times.append(by_config[key]["B"]["total_ms"])
                miss_vals.append(num_miss)

        color = colors[i % len(colors)]
        ax.plot(miss_vals, a_times, marker="s", ls="--", color=color,
                label=f"Path A (GPU), r={rank}", ms=5, lw=1.5, alpha=0.7)
        ax.plot(miss_vals, b_times, marker="o", ls="-", color=color,
                label=f"Path B (CPU), r={rank}", ms=5, lw=1.5)

    ax.set_xlabel("num_miss_loras")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Figure 2c: Per-LoRA Scaling (decode, seq_len=1)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        plt.savefig(out / f"fig2c_scaling.{fmt}", dpi=300)
    plt.close()

    print(f"Plots saved to: {out}")


def main() -> None:
    args = parse_args()

    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    num_miss_list = [int(x.strip()) for x in args.num_miss_list.split(",") if x.strip()]
    dtype = _dtype(args.dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("=" * 80)
    print("CoLoRA Motivation Study: Single MoE Layer Microbenchmark")
    print("=" * 80)
    print(f"GPU: {device_name}")
    print(f"Model: hidden_dim={args.hidden_dim}, intermediate_dim={args.intermediate_dim}")
    print(f"Sweep: ranks={ranks}, num_miss={num_miss_list}")
    print(f"dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 80)

    results = run_benchmark(
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        ranks=ranks,
        num_miss_list=num_miss_list,
        warmup=args.warmup,
        iters=args.iters,
        dtype=dtype,
        device=args.device,
    )

    write_csv(results, args.output)
    print_summary(results)

    if not args.no_plot:
        plot_results(results, str(Path(args.output).parent))

    print(f"\nTotal rows: {len(results)} (expected: {len(ranks) * len(num_miss_list) * 2})")


if __name__ == "__main__":
    main()
