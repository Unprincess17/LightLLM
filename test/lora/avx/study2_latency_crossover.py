#!/usr/bin/env python3
"""
Study 2: Latency Crossover Threshold (Micro-benchmarking)

Goal:
    Find the token threshold N where:
        T_CPU_Stream > T_GPU_Stream

Definitions:
    T_GPU_Stream:
        Base MoE layer execution on GPU (gate/up/down GEMMs + activation).

    T_CPU_Stream:
        Gather/packer + transfer to CPU + CPU LoRA compute + transfer to GPU.

Notes:
    - This script is designed to be run under Nsight Systems (nsys) so that
      NVTX ranges and numeric timing output line up.
    - The "coalescing packer" path is approximated with:
        argsort(topk_ids) -> index_select(hidden_states)
      for a single routed expert.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch

from lightllm.utils.nvtx_utils import NvtxAnnotate


@dataclass
class ResultRow:
    n_tokens: int
    t_gpu_stream_us: float
    t_cpu_stream_us: float
    gather_us: float
    to_cpu_us: float
    cpu_compute_us: float
    to_gpu_us: float


def parse_token_counts(text: str) -> List[int]:
    values: List[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        n = int(raw)
        if n <= 0:
            raise ValueError(f"Token count must be > 0, got {n}")
        values.append(n)

    if not values:
        raise ValueError("No valid token counts provided")

    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            raise ValueError(
                "Token counts must be strictly increasing (monotonic). "
                f"Got: {values}"
            )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study 2 micro-benchmark for CPU/GPU stream latency crossover"
    )
    parser.add_argument("--token-counts", type=str, default="1,8,32,64,128,256")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="test/lora/avx/study2_latency_crossover.csv")
    parser.add_argument(
        "--avx-mode",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="auto=use AVX if available, on=require AVX, off=force torch CPU matmul",
    )
    return parser.parse_args()


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


def _run_gpu_stream_once(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    nvtx_tag: str,
) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with NvtxAnnotate(f"Study2/N={nvtx_tag}/GPU_Stream"):
        gate = torch.mm(hidden_states, w1.t())
        up = torch.mm(hidden_states, w3.t())
        act = torch.nn.functional.silu(gate) * up
        _ = torch.mm(act, w2.t())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6


def _cpu_lora_compute(
    cpu_input: torch.Tensor,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    scaling: float,
    use_avx: bool,
    avx_batch_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] | None,
) -> torch.Tensor:
    if use_avx:
        if avx_batch_fn is None:
            raise RuntimeError("AVX compute was requested, but AVX function is not available.")
        return avx_batch_fn(cpu_input, a_cpu, b_cpu, scaling)

    # Reference fallback on CPU.
    inter = torch.matmul(cpu_input.float(), a_cpu.float().t())
    out = torch.matmul(inter, b_cpu.float()) * scaling
    return out.to(dtype=cpu_input.dtype)


def _run_cpu_stream_once(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    cpu_input_pinned: torch.Tensor,
    gpu_output: torch.Tensor,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    scaling: float,
    use_avx: bool,
    avx_batch_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] | None,
    nvtx_tag: str,
) -> Dict[str, float]:
    """
    Returns per-stage times in microseconds.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with NvtxAnnotate(f"Study2/N={nvtx_tag}/CPU_Stream"):
        with NvtxAnnotate(f"Study2/N={nvtx_tag}/Gather"):
            flat_topk = topk_ids.reshape(-1)
            sorted_token_indices = torch.argsort(flat_topk)
            batch_indices = torch.div(sorted_token_indices, top_k, rounding_mode="floor")
            packed_gpu = hidden_states.index_select(0, batch_indices)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with NvtxAnnotate(f"Study2/N={nvtx_tag}/Transfer_ToCPU"):
            cpu_input_pinned.copy_(packed_gpu, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        with NvtxAnnotate(f"Study2/N={nvtx_tag}/CPU_AVX_Compute"):
            cpu_out = _cpu_lora_compute(cpu_input_pinned, a_cpu, b_cpu, scaling, use_avx, avx_batch_fn)
        t3 = time.perf_counter()

        with NvtxAnnotate(f"Study2/N={nvtx_tag}/Transfer_ToGPU"):
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


def _resolve_avx_mode(
    avx_mode: str,
) -> tuple[bool, bool, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] | None]:
    """
    Returns:
        (avx_kernel_available, use_avx, avx_batch_fn)
    """
    if avx_mode == "off":
        return False, False, None

    try:
        from lightllm._kernels.lora.lora_cpu_kernel import batch_lora_avx, is_available as avx_is_available
    except Exception:
        if avx_mode == "on":
            raise RuntimeError("AVX mode was forced on, but AVX kernel module failed to import.")
        return False, False, None

    avx_ready = bool(avx_is_available())
    if avx_mode == "on" and not avx_ready:
        raise RuntimeError("AVX mode was forced on, but AVX kernel is unavailable.")

    use_avx = avx_ready and avx_mode != "off"
    return avx_ready, use_avx, (batch_lora_avx if use_avx else None)


def _find_threshold(rows: Sequence[ResultRow]) -> int | None:
    for row in rows:
        if row.t_cpu_stream_us > row.t_gpu_stream_us:
            return row.n_tokens
    return None


def _write_csv(rows: Sequence[ResultRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_tokens",
                "t_gpu_stream_us",
                "t_cpu_stream_us",
                "gather_us",
                "transfer_to_cpu_us",
                "cpu_avx_compute_us",
                "transfer_to_gpu_us",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "n_tokens": row.n_tokens,
                    "t_gpu_stream_us": f"{row.t_gpu_stream_us:.3f}",
                    "t_cpu_stream_us": f"{row.t_cpu_stream_us:.3f}",
                    "gather_us": f"{row.gather_us:.3f}",
                    "transfer_to_cpu_us": f"{row.to_cpu_us:.3f}",
                    "cpu_avx_compute_us": f"{row.cpu_compute_us:.3f}",
                    "transfer_to_gpu_us": f"{row.to_gpu_us:.3f}",
                }
            )


def main() -> None:
    args = parse_args()
    token_counts = parse_token_counts(args.token_counts)
    dtype = _dtype_from_name(args.dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    avx_ready, use_avx, avx_batch_fn = _resolve_avx_mode(args.avx_mode)
    if use_avx and dtype != torch.bfloat16:
        raise ValueError("AVX path requires bfloat16 dtype.")

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("==============================================")
    print("Study 2 Latency Crossover Micro-benchmark")
    print("==============================================")
    print(f"GPU: {device_name}")
    print(f"Token counts: {token_counts}")
    print(f"hidden_size={args.hidden_size}, intermediate_size={args.intermediate_size}, lora_rank={args.lora_rank}")
    print(f"dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}, cpu_threads={torch.get_num_threads()}")
    print(f"AVX kernel available={avx_ready}, use_avx={use_avx}")
    print("==============================================")

    max_n = max(token_counts)

    with torch.no_grad():
        hidden_all = torch.randn(max_n, args.hidden_size, device="cuda", dtype=dtype)

        # Base MoE GPU weights (single expert projection path).
        w1 = torch.randn(args.intermediate_size, args.hidden_size, device="cuda", dtype=dtype)
        w3 = torch.randn(args.intermediate_size, args.hidden_size, device="cuda", dtype=dtype)
        w2 = torch.randn(args.hidden_size, args.intermediate_size, device="cuda", dtype=dtype)

        # CPU LoRA weights for AVX path.
        a_cpu = torch.randn(args.lora_rank, args.hidden_size, device="cpu", dtype=torch.bfloat16).contiguous()
        b_cpu = torch.randn(args.lora_rank, args.hidden_size, device="cpu", dtype=torch.bfloat16).contiguous()

        results: List[ResultRow] = []
        for n_tokens in token_counts:
            nvtx_tag = str(n_tokens)
            hidden_states = hidden_all[:n_tokens]
            topk_ids = torch.zeros((n_tokens, args.top_k), device="cuda", dtype=torch.int32)

            cpu_input_pinned = torch.empty(
                (n_tokens * args.top_k, args.hidden_size),
                device="cpu",
                dtype=torch.bfloat16,
                pin_memory=True,
            )
            gpu_output = torch.empty(
                (n_tokens * args.top_k, args.hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            )

            # Warmup
            for _ in range(args.warmup):
                _run_gpu_stream_once(hidden_states, w1, w3, w2, nvtx_tag)
                _run_cpu_stream_once(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    top_k=args.top_k,
                    cpu_input_pinned=cpu_input_pinned,
                    gpu_output=gpu_output,
                    a_cpu=a_cpu,
                    b_cpu=b_cpu,
                    scaling=args.scaling,
                    use_avx=use_avx,
                    avx_batch_fn=avx_batch_fn,
                    nvtx_tag=nvtx_tag,
                )

            gpu_samples: List[float] = []
            cpu_total_samples: List[float] = []
            gather_samples: List[float] = []
            to_cpu_samples: List[float] = []
            cpu_compute_samples: List[float] = []
            to_gpu_samples: List[float] = []

            for _ in range(args.iters):
                gpu_samples.append(_run_gpu_stream_once(hidden_states, w1, w3, w2, nvtx_tag))

                cpu_parts = _run_cpu_stream_once(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    top_k=args.top_k,
                    cpu_input_pinned=cpu_input_pinned,
                    gpu_output=gpu_output,
                    a_cpu=a_cpu,
                    b_cpu=b_cpu,
                    scaling=args.scaling,
                    use_avx=use_avx,
                    avx_batch_fn=avx_batch_fn,
                    nvtx_tag=nvtx_tag,
                )
                cpu_total_samples.append(cpu_parts["total_us"])
                gather_samples.append(cpu_parts["gather_us"])
                to_cpu_samples.append(cpu_parts["to_cpu_us"])
                cpu_compute_samples.append(cpu_parts["cpu_compute_us"])
                to_gpu_samples.append(cpu_parts["to_gpu_us"])

            row = ResultRow(
                n_tokens=n_tokens,
                t_gpu_stream_us=_median_us(gpu_samples),
                t_cpu_stream_us=_median_us(cpu_total_samples),
                gather_us=_median_us(gather_samples),
                to_cpu_us=_median_us(to_cpu_samples),
                cpu_compute_us=_median_us(cpu_compute_samples),
                to_gpu_us=_median_us(to_gpu_samples),
            )
            results.append(row)

    print()
    print(
        f"{'N':>6} {'T_GPU_Stream(us)':>18} {'T_CPU_Stream(us)':>18} "
        f"{'Gather(us)':>12} {'ToCPU(us)':>12} {'CPU_AVX(us)':>12} {'ToGPU(us)':>12}"
    )
    print("-" * 96)
    for r in results:
        print(
            f"{r.n_tokens:>6d} "
            f"{r.t_gpu_stream_us:>18.3f} "
            f"{r.t_cpu_stream_us:>18.3f} "
            f"{r.gather_us:>12.3f} "
            f"{r.to_cpu_us:>12.3f} "
            f"{r.cpu_compute_us:>12.3f} "
            f"{r.to_gpu_us:>12.3f}"
        )

    threshold = _find_threshold(results)
    if threshold is None:
        print("\nN_threshold: not found in tested range (CPU stream stayed <= GPU stream).")
    else:
        print(f"\nN_threshold: {threshold} (first N where T_CPU_Stream > T_GPU_Stream)")

    output_csv = Path(args.output_csv)
    _write_csv(results, output_csv)
    print(f"CSV written: {output_csv}")


if __name__ == "__main__":
    main()
