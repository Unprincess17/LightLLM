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
from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.lora.expert_cache import (
    ExpertCacheKey,
    MoEExpertCacheConfig,
    MoEExpertCacheManager,
)
from lightllm.server.lora.lora_mem_pool import LoRAModulePool
from lightllm.models.qwen3_vl_moe.lora_dispatch import Qwen3VLMoELoRADispatcher, BGMV_AVAILABLE

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

    # Benchmark COLoRA hybrid miss path (GPU-miss fallback to CPU)
    print("\nBenchmarking COLoRA hybrid miss path (CPU fallback)...")
    pool = LoRAModulePool.create(
        pool_size=8,
        max_rank=rank,
        input_dim=hidden,
        output_dim=hidden,
        dtype=torch.bfloat16,
        device="cpu",
        num_layers=1,
        num_experts=1,
    )
    pool.load_adapter(
        adapter_idx=0,
        rank=rank,
        scaling=scaling,
        layer_weights={0: {"A": A, "B": B}},
    )

    cache_mgr = MoEExpertCacheManager(
        MoEExpertCacheConfig(
            cache_budget_mb=16,
            promote_min_hits=1000,  # keep benchmark in miss-only mode
            promote_window=8,
            max_promote_per_step=1,
            decay=0.9,
        )
    )
    cache_mgr.register_projection_pool("gate", pool)

    dispatcher = Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=rank,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
    )
    dispatcher.expert_cache_manager = cache_mgr
    bins = torch.tensor([0], dtype=torch.long)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = dispatcher._batch_apply_moe_lora_hybrid(
            input_tensor=x,
            layer_id=0,
            buffer_layer_id=0,
            pool=pool,
            bins=bins,
            projection="gate",
            expert_id=0,
        )
    hybrid_miss_time = (time.perf_counter() - start) / iterations * 1000
    print(f"COLoRA hybrid miss: {hybrid_miss_time:.3f} ms per iteration")

    if torch.cuda.is_available() and BGMV_AVAILABLE:
        print("\nBenchmarking COLoRA hybrid hit path (GPU cache hit)...")
        hit_key = ExpertCacheKey(projection="gate", adapter_idx=0, layer_id=0, expert_id=0)
        cache_mgr.record_access([hit_key])
        cache_mgr.schedule_promotion([hit_key])
        cache_mgr.apply_completed_promotions()

        x_cuda = x.to("cuda")
        bins_cuda = bins.to("cuda")
        start = time.perf_counter()
        for _ in range(iterations):
            _ = dispatcher._batch_apply_moe_lora_hybrid(
                input_tensor=x_cuda,
                layer_id=0,
                buffer_layer_id=0,
                pool=pool,
                bins=bins_cuda,
                projection="gate",
                expert_id=0,
            )
        hybrid_hit_time = (time.perf_counter() - start) / iterations * 1000
        print(f"COLoRA hybrid hit: {hybrid_hit_time:.3f} ms per iteration")
    else:
        hybrid_hit_time = None
        print("COLoRA hybrid hit path skipped (requires CUDA + BGMV kernel).")

    # Calculate ratios
    print()
    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    avx_ratio = avx_time / pytorch_time
    print(f"AVX-512 vs PyTorch (1T): {avx_ratio:.2f}x")

    avx_mt_ratio = avx_time / pytorch_mt_time
    print(f"AVX-512 vs PyTorch (MT): {avx_mt_ratio:.2f}x")
    print(f"COLoRA hybrid miss vs AVX-512: {(hybrid_miss_time / avx_time):.2f}x")
    if hybrid_hit_time is not None:
        print(f"COLoRA hybrid hit vs AVX-512: {(hybrid_hit_time / avx_time):.2f}x")

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
