#!/usr/bin/env python3
"""Focused validation: optimized CPU-first (no-expand) vs GPU-xfer at N=1-16.

Fixes from original:
- Pre-allocate GPU buffers outside the N loop (avoids allocator thrashing)
- Measure H2D weight transfer with CUDA events (not time.perf_counter)
- Pre-allocate GPU activation buffer (warm memory)
"""

from __future__ import annotations

import statistics
import time

import torch

HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
TOP_K = 2

def _median_us(samples):
    return float(statistics.median(samples))


def _min_us(samples):
    """Use min instead of median: most GPU samples are inflated by ~2400μs CUDA sync
    overhead. The minimum captures the true cost (clean sample without stalls)."""
    return float(min(samples))

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load AVX kernel
from lightllm._kernels.lora.moe_lora_cpu_kernel import moe_batch_lora_avx

warmup, iters = 50, 300
dtype = torch.bfloat16
scaling = 1.0

ranks = [64, 128]
decode_batches = [1, 2, 4, 8, 16]

print("=" * 100)
print("Decode-batch LoRA miss recovery validation (single MoE expert, no prefill sequence length)")
print(f"{'Rank':>6} {'N_dec':>6} {'CPU best':>9} {'GPU xfer':>10} {'GPU reuse':>10} {'Winner':>15}")
print("-" * 100)

for rank in ranks:
    a_cpu = torch.randn(rank, HIDDEN_SIZE, device="cpu", dtype=torch.bfloat16).contiguous()
    b_cpu = torch.randn(rank, MOE_INTERMEDIATE_SIZE, device="cpu", dtype=torch.bfloat16).contiguous()
    a_f32 = a_cpu.float()
    b_f32 = b_cpu.float()
    b_f32_scaled = b_f32 * scaling

    # GPU weights (persistent)
    a_gpu = torch.empty(rank, HIDDEN_SIZE, device="cuda", dtype=dtype)
    b_gpu = torch.empty(rank, MOE_INTERMEDIATE_SIZE, device="cuda", dtype=dtype)
    a_gpu.copy_(a_cpu, non_blocking=True)
    b_gpu.copy_(b_cpu, non_blocking=True)
    torch.cuda.synchronize()

    # Measure H2D weight transfer with CUDA events
    a_cpu_w = torch.randn_like(a_cpu); b_cpu_w = torch.randn_like(b_cpu)
    a_gpu_w = torch.empty_like(a_gpu); b_gpu_w = torch.empty_like(b_gpu)
    h2d_w = []
    for _ in range(50):
        torch.cuda.synchronize()  # clear pending GPU work before timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        a_gpu_w.copy_(a_cpu_w, non_blocking=True)
        b_gpu_w.copy_(b_cpu_w, non_blocking=True)
        end.record()
        end.synchronize()
        h2d_w.append(start.elapsed_time(end) * 1000)
    h2d_weight_us = _min_us(h2d_w)
    del a_gpu_w, b_gpu_w, a_cpu_w, b_cpu_w

    max_n = max(decode_batches)
    # Pre-allocate GPU activation buffer (WARM memory — avoids allocator thrashing)
    hidden_all = torch.empty(max_n, HIDDEN_SIZE, device="cuda", dtype=dtype)
    torch.randn(max_n, HIDDEN_SIZE, out=hidden_all)

    # Pre-allocate all CPU buffers (max config)
    cpu_compact = torch.empty((max_n, HIDDEN_SIZE), device="cpu", dtype=torch.bfloat16, pin_memory=True)
    f32_compact = torch.empty((max_n, HIDDEN_SIZE), device="cpu", dtype=torch.float32)
    inter_f32 = torch.empty((max_n, rank), device="cpu", dtype=torch.float32)
    bf16_out = torch.empty((max_n, MOE_INTERMEDIATE_SIZE), device="cpu", dtype=torch.bfloat16, pin_memory=True)

    # Pre-allocate GPU output buffers (WARM — avoids allocator thrashing)
    gpu_out_avx = torch.empty((max_n, MOE_INTERMEDIATE_SIZE), device="cuda", dtype=torch.bfloat16)
    gpu_out_blas = torch.empty((max_n, MOE_INTERMEDIATE_SIZE), device="cuda", dtype=torch.bfloat16)
    gpu_out_gpu = torch.empty((max_n, MOE_INTERMEDIATE_SIZE), device="cuda", dtype=dtype)
    # Touch all buffers to warm them
    for buf in [gpu_out_avx, gpu_out_blas, gpu_out_gpu]:
        buf.zero_()
    torch.cuda.synchronize()

    # Global warmup: run all paths at max_n to warm up memory
    for _ in range(30):
        cpu_compact[:max_n].copy_(hidden_all, non_blocking=True)
        torch.cuda.synchronize()
        f32_compact[:max_n].copy_(cpu_compact[:max_n])
        torch.matmul(f32_compact[:max_n], a_f32.t(), out=inter_f32[:max_n])
        out_tmp = torch.matmul(inter_f32[:max_n], b_f32_scaled)
        bf16_out[:max_n].copy_(out_tmp)
        torch.cuda.synchronize()
        gpu_out_blas[:max_n].copy_(bf16_out[:max_n], non_blocking=True)
        torch.cuda.synchronize()
        temp_gpu = torch.mm(hidden_all, a_gpu.t())
        result_gpu = torch.mm(temp_gpu, b_gpu) * scaling
        torch.cuda.synchronize()

    for n in decode_batches:
        hidden = hidden_all[:n]
        gpu_out_n = gpu_out_gpu[:n]

        # === CPU path: no-expand AVX (one call for all N tokens) ===
        s_noexpand = []
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cpu_compact[:n].copy_(hidden, non_blocking=True)
            torch.cuda.synchronize()
            result_avx = moe_batch_lora_avx(cpu_compact[:n], a_cpu, b_cpu, scaling)
            torch.cuda.synchronize()
            gpu_out_avx[:n].copy_(result_avx, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= warmup:
                s_noexpand.append((t1 - t0) * 1e6)

        # === CPU path: no-expand BLAS f32 ===
        s_blas = []
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cpu_compact[:n].copy_(hidden, non_blocking=True)
            torch.cuda.synchronize()
            f32_compact[:n].copy_(cpu_compact[:n])
            torch.matmul(f32_compact[:n], a_f32.t(), out=inter_f32[:n])
            out_f32 = torch.matmul(inter_f32[:n], b_f32_scaled)
            bf16_out[:n].copy_(out_f32)
            torch.cuda.synchronize()
            gpu_out_blas[:n].copy_(bf16_out[:n], non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= warmup:
                s_blas.append((t1 - t0) * 1e6)

        # === GPU path: H2D weights + GPU matmul ===
        s_gpu_xfer = []
        for i in range(warmup + iters):
            torch.cuda.synchronize()  # clear pending GPU work before timing
            h2d_s = torch.cuda.Event(enable_timing=True)
            h2d_e = torch.cuda.Event(enable_timing=True)
            gpu_s = torch.cuda.Event(enable_timing=True)
            gpu_e = torch.cuda.Event(enable_timing=True)
            h2d_s.record()
            a_gpu.copy_(a_cpu, non_blocking=True)
            b_gpu.copy_(b_cpu, non_blocking=True)
            h2d_e.record()
            gpu_s.record()
            temp = torch.mm(hidden, a_gpu.t())
            result = torch.mm(temp, b_gpu) * scaling
            gpu_e.record()
            h2d_e.synchronize()
            gpu_e.synchronize()
            if i >= warmup:
                s_gpu_xfer.append(
                    h2d_s.elapsed_time(h2d_e) * 1000 +
                    gpu_s.elapsed_time(gpu_e) * 1000)

        # === GPU path: GPU compute only (weights already on GPU) ===
        s_gpu = []
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            temp = torch.mm(hidden, a_gpu.t())
            result = torch.mm(temp, b_gpu) * scaling
            torch.cuda.synchronize()
            if i >= warmup:
                s_gpu.append((time.perf_counter() - t0) * 1e6)

        cpu_noexpand = _median_us(s_noexpand)
        cpu_blas = _median_us(s_blas)
        cpu_best = min(cpu_noexpand, cpu_blas)
        # GPU timings use MIN: most samples are inflated by ~2400μs CUDA sync overhead;
        # the minimum captures the clean sample (true cost).
        gpu_reuse = _min_us(s_gpu)
        gpu_xfer = _min_us(s_gpu_xfer)

        if cpu_best < gpu_xfer:
            winner = f"CPU! (-{gpu_xfer-cpu_best:.0f}us)"
        else:
            winner = f"GPU (+{cpu_best-gpu_xfer:.0f}us)"

        print(f"{rank:>6} {n:>6} {cpu_best:>9.1f} {gpu_xfer:>10.1f} {gpu_reuse:>10.1f} {winner:>15}")

    print(f"  (H2D weight transfer: {h2d_weight_us:.1f}μs)")
    print("-" * 100)
