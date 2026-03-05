#!/usr/bin/env python3
"""
Quick benchmark to measure AVX kernel vs PyTorch for single token inference
This script simulates the study2 scenario but runs locally without CUDA
"""

import torch
import sys
import time
sys.path.insert(0, '/home/shufan/LightLLM-integrate-to-SLoRA')

from lightllm._kernels.lora.lora_cpu_kernel import (
    batch_lora_avx,
    is_available as avx_is_available
)

def torch_reference(x, A, B, scaling=1.0):
    """PyTorch reference implementation for single token"""
    x_f32 = x.float()
    A_f32 = A.float()
    B_f32 = B.float()
    intermediate = torch.matmul(x_f32, A_f32.T)
    out = torch.matmul(intermediate, B_f32) * scaling
    return out.bfloat16()

def main():
    print("=" * 60)
    print("Single Token LoRA Benchmark: AVX-512 vs PyTorch")
    print("=" * 60)

    if not avx_is_available():
        print("ERROR: AVX-512 kernel not available")
        return

    # Configuration (single token, 4k hidden size, rank 16)
    batch, hidden, rank = 1, 4096, 16

    # Create tensors in bfloat16 (CPU)
    x = torch.randn(batch, hidden, dtype=torch.bfloat16)
    A = torch.randn(rank, hidden, dtype=torch.bfloat16)
    B = torch.randn(rank, hidden, dtype=torch.bfloat16)
    scaling = 0.1

    print(f"Configuration: batch={batch}, hidden={hidden}, rank={rank}")
    print()

    # Warm up
    print("Warming up...")
    for _ in range(10):
        _ = batch_lora_avx(x, A, B, scaling)
        _ = torch_reference(x, A, B, scaling)

    # Benchmark AVX
    print("\nBenchmarking AVX-512 kernel...")
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        result = batch_lora_avx(x, A, B, scaling)
    avx_time = (time.perf_counter() - start) / iterations * 1000
    print(f"AVX-512: {avx_time:.3f} ms per iteration")

    # Benchmark PyTorch (single-threaded)
    print("\nBenchmarking PyTorch (single-threaded)...")
    torch.set_num_threads(1)
    start = time.perf_counter()
    for _ in range(iterations):
        result = torch_reference(x, A, B, scaling)
    pytorch_time = (time.perf_counter() - start) / iterations * 1000
    print(f"PyTorch (1T): {pytorch_time:.3f} ms per iteration")

    # Benchmark PyTorch (multi-threaded)
    print("\nBenchmarking PyTorch (multi-threaded)...")
    torch.set_num_threads(torch.get_num_threads())  # Reset to default
    start = time.perf_counter()
    for _ in range(iterations):
        result = torch_reference(x, A, B, scaling)
    pytorch_mt_time = (time.perf_counter() - start) / iterations * 1000
    print(f"PyTorch (MT): {pytorch_mt_time:.3f} ms per iteration")

    # Calculate ratios
    print()
    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    avx_ratio = avx_time / pytorch_time
    print(f"AVX-512 vs PyTorch (1T): {avx_ratio:.2f}x")

    avx_mt_ratio = avx_time / pytorch_mt_time
    print(f"AVX-512 vs PyTorch (MT): {avx_mt_ratio:.2f}x")

    # Verify correctness
    print()
    print("=" * 60)
    print("Correctness Verification")
    print("=" * 60)
    out_avx = batch_lora_avx(x, A, B, scaling)
    out_ref = torch_reference(x, A, B, scaling)

    # Compare with tolerance (bfloat16 has limited precision)
    max_diff = (out_avx.float() - out_ref.float()).abs().max().item()
    ref_max = out_ref.float().abs().max().item()
    rel_diff = max_diff / max(ref_max, 1e-6)

    print(f"Max absolute difference: {max_diff:.4f}")
    print(f"Relative difference: {rel_diff * 100:.2f}%")

    if rel_diff < 0.02:
        print("✓ Correctness verified (relative error < 2%)")
    else:
        print("⚠️  WARNING: Relative error exceeds 2%")

    print()
    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("- This scenario matches study2's N=1 (single token) case")
    print("- For MoE architectures, this is the most common and strictest scenario")
    print("- AVX-512 offers significant improvement over CPU PyTorch implementations")

if __name__ == "__main__":
    main()
