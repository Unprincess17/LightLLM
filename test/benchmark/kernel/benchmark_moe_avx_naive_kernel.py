#!/usr/bin/env python3
"""
Micro-benchmark: MoE CPU LoRA AVX kernels vs PyTorch bf16 matmul (naive reference).

Up/down AVX paths use a float-accumulate reference implementation in C++ (correctness-first);
PyTorch ``matmul`` is often faster on stage-2 for medium/large shapes, so speedup can be below 1x.

Run from the repository root::

    python test/benchmark/kernel/benchmark_moe_avx_naive_kernel.py
    python test/benchmark/kernel/benchmark_moe_avx_naive_kernel.py --phase gate --n 64 --h 4096 --r 32 --iters 500

Requires the MoE AVX extension to be available (same as live e2e ``cpu_kernel_mode: avx``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo root on path when executed as a file
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from lightllm._kernels.lora.moe_lora_cpu_kernel import (
    is_available as moe_lora_avx_is_available,
    moe_batch_lora_gate_avx,
    moe_batch_lora_up_avx,
    moe_batch_lora_down_avx,
)


def _naive_gate(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, A.transpose(0, 1))


def _naive_stage2(inter: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    return torch.matmul(inter, B) * scaling


def _bench_one(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("gate", "up", "down", "pipeline", "all"),
        default="all",
        help="pipeline = gate then up (matches strict MoE CPU path); "
        "fused moe_batch_lora_avx is a separate code path (not benchmarked here).",
    )
    p.add_argument("--n", type=int, default=32, help="batch / token count")
    p.add_argument("--h", type=int, default=4096, help="hidden size")
    p.add_argument("--r", type=int, default=32, help="LoRA rank")
    p.add_argument("--scaling", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    if not moe_lora_avx_is_available():
        print("MoE LoRA AVX kernel is not available on this host; exiting.", file=sys.stderr)
        sys.exit(2)

    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    n, h, r = args.n, args.h, args.r
    scaling = args.scaling

    def run_phase(name: str) -> None:
        if name == "gate":
            x = torch.randn(n, h, dtype=torch.bfloat16, generator=g)
            A = torch.randn(r, h, dtype=torch.bfloat16, generator=g)

            def avx():
                return moe_batch_lora_gate_avx(x, A, scaling=1.0)

            def naive():
                return _naive_gate(x, A)

            y_a = avx()
            y_n = naive()
            rel = (y_a.float() - y_n.float()).abs().max().item() / max(y_n.float().abs().max().item(), 1e-6)
        elif name in ("up", "down"):
            inter = torch.randn(n, r, dtype=torch.bfloat16, generator=g)
            B = torch.randn(r, h, dtype=torch.bfloat16, generator=g)
            avx_fn = moe_batch_lora_up_avx if name == "up" else moe_batch_lora_down_avx

            def avx():
                return avx_fn(inter, B, scaling=scaling)

            def naive():
                return _naive_stage2(inter, B, scaling)

            y_a = avx()
            y_n = naive()
            rel = (y_a.float() - y_n.float()).abs().max().item() / max(y_n.float().abs().max().item(), 1e-6)
        else:  # pipeline (gate + up, same as strict MoE CPU LoRA)
            x = torch.randn(n, h, dtype=torch.bfloat16, generator=g)
            A = torch.randn(r, h, dtype=torch.bfloat16, generator=g)
            B = torch.randn(r, h, dtype=torch.bfloat16, generator=g)

            def avx():
                inter = moe_batch_lora_gate_avx(x, A, scaling=1.0)
                return moe_batch_lora_up_avx(inter, B, scaling=scaling)

            def naive():
                inter = _naive_gate(x, A)
                return _naive_stage2(inter, B, scaling)

            y_a = avx()
            y_n = naive()
            rel = (y_a.float() - y_n.float()).abs().max().item() / max(y_n.float().abs().max().item(), 1e-6)

        ms_avx = _bench_one(avx, args.warmup, args.iters)
        ms_naive = _bench_one(naive, args.warmup, args.iters)
        print(
            f"{name:5s}  n={n} h={h} r={r}  "
            f"AVX {ms_avx:.4f} ms/iter  naive {ms_naive:.4f} ms/iter  "
            f"speedup {ms_naive / ms_avx:.2f}x  max_rel_err {rel:.4f}"
        )

    phases = (
        ["gate", "up", "down", "pipeline"]
        if args.phase == "all"
        else [args.phase]
    )
    print(f"warmup={args.warmup} iters={args.iters} scaling={scaling}")
    for ph in phases:
        run_phase(ph)


if __name__ == "__main__":
    main()
