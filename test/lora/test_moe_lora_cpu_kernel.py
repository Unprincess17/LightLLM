#!/usr/bin/env python3
"""
Benchmark script for MoE-specific AVX-512 LoRA kernel

This script benchmarks the performance of the new MoE-specific AVX-512 kernel
across various token counts and expert sizes, characteristic of MoE architectures.
It includes:

1. Performance comparison across different kernel sizes
2. Evaluation of dynamic kernel selection
3. Irregular workload scheduling efficiency
4. Gate/Up/Down phase specific performance
5. Comparison with original AVX kernel

Created for LightLLM project
"""

import torch
import sys
import time
sys.path.insert(0, '/home/shufan/LightLLM-integrate-to-SLoRA')

from lightllm._kernels.lora.moe_lora_cpu_kernel import (
    is_available as moe_is_available,
    moe_batch_lora_avx,
    moe_batch_lora_gate_avx,
    moe_batch_lora_up_avx,
    moe_batch_lora_down_avx,
)

from lightllm._kernels.lora.lora_cpu_kernel import (
    is_available as original_is_available,
    batch_lora_avx,
)

def torch_reference(x, A, B, scaling=1.0):
    """PyTorch reference implementation for comparison"""
    x_f32 = x.float()
    A_f32 = A.float()
    B_f32 = B.float()

    intermediate = torch.matmul(x_f32, A_f32.T)
    out = torch.matmul(intermediate, B_f32) * scaling

    return out.bfloat16()

def torch_gate_reference(x, A, scaling=1.0):
    """PyTorch reference for gate projection"""
    x_f32 = x.float()
    A_f32 = A.float()
    return (torch.matmul(x_f32, A_f32.T) * scaling).bfloat16()

def torch_updown_reference(x, B, scaling=1.0):
    """PyTorch reference for up/down projection"""
    x_f32 = x.float()
    B_f32 = B.float()
    return (torch.matmul(x_f32, B_f32) * scaling).bfloat16()

def run_benchmark():
    print("=" * 60)
    print("MoE-Specific AVX-512 LoRA Kernel Benchmark")
    print("=" * 60)

    if not moe_is_available():
        print("ERROR: MoE LoRA kernel not available")
        return

    if not original_is_available():
        print("ERROR: Original AVX kernel not available")
        return

    print(f"MoE kernel available: ✓")
    print(f"Original AVX kernel available: ✓")
    print()

    # Configuration for testing
    hidden_sizes = [2048, 4096]
    ranks = [16, 32, 64]
    token_counts = [1, 2, 4, 8, 16, 32, 64, 128]  # MoE expert sizes
    scaling = 0.1
    iterations = 1000

    print("=" * 60)
    print("Full Kernel Performance (Dynamic Selection)")
    print("=" * 60)

    print(f"{'Token Count':<12} {'Hidden Size':<10} {'Rank':<6} {'MoE Kernel':<12} {'Original AVX':<12} {'PyTorch':<12} {'Speedup (MoE/PyTorch)':<15}")
    print("-" * 90)

    all_results = []

    for hidden in hidden_sizes:
        for rank in ranks:
            for tokens in token_counts:
                # Create test data
                x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
                A = torch.randn(rank, hidden, dtype=torch.bfloat16)
                B = torch.randn(rank, hidden, dtype=torch.bfloat16)

                # Warm up
                _ = moe_batch_lora_avx(x, A, B, scaling)
                _ = batch_lora_avx(x, A, B, scaling)
                _ = torch_reference(x, A, B, scaling)

                # Benchmark MoE kernel
                start = time.perf_counter()
                for _ in range(iterations):
                    result_moe = moe_batch_lora_avx(x, A, B, scaling)
                time_moe = (time.perf_counter() - start) / iterations * 1000  # ms

                # Benchmark original AVX kernel
                start = time.perf_counter()
                for _ in range(iterations):
                    result_original = batch_lora_avx(x, A, B, scaling)
                time_original = (time.perf_counter() - start) / iterations * 1000  # ms

                # Benchmark PyTorch
                start = time.perf_counter()
                for _ in range(iterations):
                    result_ref = torch_reference(x, A, B, scaling)
                time_ref = (time.perf_counter() - start) / iterations * 1000  # ms

                # Verify correctness
                max_diff = (result_moe.float() - result_ref.float()).abs().max().item()
                ref_max = result_ref.float().abs().max().item()
                rel_diff = max_diff / max(ref_max, 1e-6)

                speedup = time_ref / time_moe

                all_results.append({
                    'tokens': tokens,
                    'hidden': hidden,
                    'rank': rank,
                    'time_moe': time_moe,
                    'time_original': time_original,
                    'time_pytorch': time_ref,
                    'speedup': speedup,
                    'rel_diff': rel_diff
                })

                print(f"{tokens:<12} {hidden:<10} {rank:<6} {time_moe:<12.3f} {time_original:<12.3f} {time_ref:<12.3f} {speedup:<15.2f}")

    print()
    print("=" * 60)
    print("Phase-Specific Kernel Performance")
    print("=" * 60)

    print(f"{'Phase':<10} {'Token Count':<12} {'Hidden Size':<10} {'Rank':<6} {'Time (ms)':<12} {'Speedup (vs PyTorch)':<15}")
    print("-" * 75)

    for phase, moe_func, ref_func in [
        ("Gate", moe_batch_lora_gate_avx, torch_gate_reference),
        ("Up", moe_batch_lora_up_avx, torch_updown_reference),
        ("Down", moe_batch_lora_down_avx, torch_updown_reference),
    ]:
        for hidden in hidden_sizes:
            for rank in ranks:
                for tokens in [1, 4, 16, 64]:  # Representative sizes
                    x = torch.randn(tokens, hidden if phase == "Gate" else rank,
                                  dtype=torch.bfloat16)
                    A = torch.randn(rank, hidden, dtype=torch.bfloat16)
                    B = torch.randn(rank, hidden, dtype=torch.bfloat16)

                    # Warm up
                    if phase == "Gate":
                        _ = moe_func(x, A, scaling)
                        _ = ref_func(x, A, scaling)
                    else:
                        _ = moe_func(x, B, scaling)
                        _ = ref_func(x, B, scaling)

                    # Benchmark
                    start = time.perf_counter()
                    for _ in range(iterations):
                        if phase == "Gate":
                            _ = moe_func(x, A, scaling)
                        else:
                            _ = moe_func(x, B, scaling)
                    time_moe = (time.perf_counter() - start) / iterations * 1000  # ms

                    start = time.perf_counter()
                    for _ in range(iterations):
                        if phase == "Gate":
                            _ = ref_func(x, A, scaling)
                        else:
                            _ = ref_func(x, B, scaling)
                    time_ref = (time.perf_counter() - start) / iterations * 1000  # ms

                    speedup = time_ref / time_moe

                    print(f"{phase:<10} {tokens:<12} {hidden:<10} {rank:<6} {time_moe:<12.3f} {speedup:<15.2f}")

    print()
    print("=" * 60)
    print("Dynamic Kernel Selection Performance")
    print("=" * 60)

    print(f"{'Token Count':<12} {'Optimal Kernel':<12} {'Time (ms)':<12}")
    print("-" * 45)

    for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x = torch.randn(tokens, 4096, dtype=torch.bfloat16)
        A = torch.randn(32, 4096, dtype=torch.bfloat16)
        B = torch.randn(32, 4096, dtype=torch.bfloat16)

        # Warm up
        _ = moe_batch_lora_avx(x, A, B, scaling)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = moe_batch_lora_avx(x, A, B, scaling)
        time_moe = (time.perf_counter() - start) / iterations * 1000  # ms

        # Determine optimal kernel type based on token count
        if tokens <= 2:
            optimal = "Tiny"
        elif tokens <= 8:
            optimal = "Small"
        elif tokens <= 32:
            optimal = "Medium"
        else:
            optimal = "Large"

        print(f"{tokens:<12} {optimal:<12} {time_moe:<12.3f}")

    print()
    print("=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    all_correct = True
    for hidden in hidden_sizes:
        for rank in ranks:
            for tokens in token_counts:
                x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
                A = torch.randn(rank, hidden, dtype=torch.bfloat16)
                B = torch.randn(rank, hidden, dtype=torch.bfloat16)

                result_moe = moe_batch_lora_avx(x, A, B, scaling)
                result_ref = torch_reference(x, A, B, scaling)

                max_diff = (result_moe.float() - result_ref.float()).abs().max().item()
                ref_max = result_ref.float().abs().max().item()
                rel_diff = max_diff / max(ref_max, 1e-6)

                if rel_diff > 0.02:  # 2% tolerance for bf16
                    print(f"❌ Error: Tokens={tokens}, Hidden={hidden}, Rank={rank} - "
                          f"Rel diff: {rel_diff * 100:.2f}%")
                    all_correct = False
                else:
                    print(f"✅ OK: Tokens={tokens}, Hidden={hidden}, Rank={rank} - "
                          f"Rel diff: {rel_diff * 100:.2f}%")

    if all_correct:
        print("\nAll results are within acceptable tolerance (2%)")
    else:
        print("\nSome results exceed acceptable tolerance")

    print()
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    if all_results:
        avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
        min_speedup = min(r['speedup'] for r in all_results)
        max_speedup = max(r['speedup'] for r in all_results)
        avg_diff = sum(r['rel_diff'] for r in all_results) / len(all_results)

        print(f"Average speedup over PyTorch: {avg_speedup:.2f}x")
        print(f"Minimum speedup: {min_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Average relative difference: {avg_diff * 100:.2f}%")

        # Find best performing configuration
        best_config = max(all_results, key=lambda x: x['speedup'])
        print(f"\nBest configuration:")
        print(f"Tokens: {best_config['tokens']}, Hidden: {best_config['hidden']}, "
              f"Rank: {best_config['rank']}")
        print(f"Speedup: {best_config['speedup']:.2f}x, Time: {best_config['time_moe']:.3f}ms")

if __name__ == "__main__":
    run_benchmark()
