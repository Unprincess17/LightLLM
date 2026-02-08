"""
Unit tests for AVX-512 BF16 LoRA Kernel
Tests correctness, performance, and edge cases.

Usage:
    python test_lora_avx.py
    python test_lora_avx.py --benchmark    # Run performance benchmark
    python test_lora_avx.py --verbose     # Detailed output
"""

import torch
import time
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lightllm._kernels.lora.lora_cpu_kernel import (
    batch_lora_avx,
    lora_down_avx,
    lora_up_avx,
    is_available
)


def torch_reference(x, A, B, scaling=1.0):
    """PyTorch reference implementation for correctness comparison."""
    x_f32 = x.float()
    A_f32 = A.float()
    B_f32 = B.float()

    # x @ A.T: [B, H] @ [R, H].T = [B, R]
    intermediate = torch.matmul(x_f32, A_f32.T)
    # intermediate @ B: [B, R] @ [R, H] = [B, H]
    out = torch.matmul(intermediate, B_f32) * scaling
    return out.bfloat16()


def test_correctness(x, A, B, scaling, name, rtol=0.01):
    """Test AVX kernel output against PyTorch reference.

    Uses relative tolerance (rtol) since bfloat16 has limited precision.
    For bfloat16 (7-bit mantissa), ~1% relative error is expected.
    """
    # AVX kernel output
    out_avx = batch_lora_avx(x, A, B, scaling)

    # PyTorch reference
    out_ref = torch_reference(x, A, B, scaling)

    # Compare using relative tolerance
    # For bfloat16, we expect some numerical differences
    abs_diff = (out_avx.float() - out_ref.float()).abs()
    max_diff = abs_diff.max().item()

    # Use reference values to compute relative error
    # Avoid division by zero
    ref_abs = out_ref.float().abs()
    ref_max = max(ref_abs.max().item(), 1e-6)
    rel_diff = max_diff / ref_max

    # bfloat16 has 7-bit mantissa, ~1.5% relative precision
    # Allow some headroom for accumulated errors
    passed = rel_diff < 0.02  # 2% relative tolerance

    return {
        'name': name,
        'passed': passed,
        'max_diff': max_diff,
        'rel_diff': rel_diff,
        'rtol': rtol
    }


def run_correctness_tests(verbose=False):
    """Run all correctness tests."""
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    if not is_available():
        print("SKIP: AVX kernel not available")
        return False

    results = []

    # Test 1: Standard shapes
    print("\n[Test 1] Standard shapes (B=4, H=4096, R=16)")
    x = torch.randn(4, 4096, dtype=torch.bfloat16)
    A = torch.randn(16, 4096, dtype=torch.bfloat16)
    B = torch.randn(16, 4096, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 0.1, "standard"))

    # Test 2: Small rank
    print("[Test 2] Small rank (R=4)")
    x = torch.randn(4, 4096, dtype=torch.bfloat16)
    A = torch.randn(4, 4096, dtype=torch.bfloat16)
    B = torch.randn(4, 4096, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 1.0, "small_rank"))

    # Test 3: Large rank
    print("[Test 3] Large rank (R=64)")
    x = torch.randn(2, 4096, dtype=torch.bfloat16)
    A = torch.randn(64, 4096, dtype=torch.bfloat16)
    B = torch.randn(64, 4096, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 0.5, "large_rank"))

    # Test 4: Single batch
    print("[Test 4] Single batch (B=1)")
    x = torch.randn(1, 4096, dtype=torch.bfloat16)
    A = torch.randn(16, 4096, dtype=torch.bfloat16)
    B = torch.randn(16, 4096, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 1.0, "single_batch"))

    # Test 5: Large batch
    print("[Test 5] Large batch (B=32)")
    x = torch.randn(32, 4096, dtype=torch.bfloat16)
    A = torch.randn(16, 4096, dtype=torch.bfloat16)
    B = torch.randn(16, 4096, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 0.1, "large_batch"))

    # Test 6: Scaling factor
    print("[Test 6] Different scaling factors")
    x = torch.randn(4, 2048, dtype=torch.bfloat16)
    A = torch.randn(8, 2048, dtype=torch.bfloat16)
    B = torch.randn(8, 2048, dtype=torch.bfloat16)
    for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
        results.append(test_correctness(x, A, B, scale, f"scaling_{scale}"))

    # Test 7: Non-contiguous input
    print("[Test 7] Non-contiguous input")
    x_full = torch.randn(8, 4096, dtype=torch.bfloat16)
    x = x_full[::2, :]  # Strided, non-contiguous
    A = torch.randn(16, 4096, dtype=torch.bfloat16)
    B = torch.randn(16, 4096, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 1.0, "noncontiguous_input"))

    # Test 8: Hidden dim not multiple of 32 (edge case)
    print("[Test 8] Hidden dim not multiple of 32 (H=4000)")
    x = torch.randn(4, 4000, dtype=torch.bfloat16)
    A = torch.randn(16, 4000, dtype=torch.bfloat16)
    B = torch.randn(16, 4000, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 1.0, "non_multiple_h"))

    # Test 9: Small hidden dim
    print("[Test 9] Small hidden dim (H=256)")
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    A = torch.randn(8, 256, dtype=torch.bfloat16)
    B = torch.randn(8, 256, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 1.0, "small_hidden"))

    # Test 10: Zero values
    print("[Test 10] Zero input values")
    x = torch.zeros(4, 2048, dtype=torch.bfloat16)
    A = torch.randn(8, 2048, dtype=torch.bfloat16)
    B = torch.randn(8, 2048, dtype=torch.bfloat16)
    results.append(test_correctness(x, A, B, 1.0, "zero_input"))

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if verbose:
        print("\nDetailed Results:")
        for r in results:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"  {r['name']}: {status} (rel_diff={r['rel_diff']*100:.2f}%)")

    return passed == total


def run_performance_benchmark(verbose=False):
    """Run performance benchmarks comparing AVX vs PyTorch.

    For fair comparison, we compare against:
    1. Single-threaded PyTorch (same resource usage)
    2. Multi-threaded PyTorch (absolute throughput)
    3. PCIe transfer time (Shadow Pipelining budget)
    """
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    if not is_available():
        print("SKIP: AVX kernel not available")
        return

    # Store original thread count
    import torch
    original_threads = torch.get_num_threads()

    # Configuration
    configs = [
        # (batch, hidden, rank, description)
        (1, 4096, 16, "B=1, H=4096, R=16"),
        (4, 4096, 16, "B=4, H=4096, R=16"),
        (8, 4096, 16, "B=8, H=4096, R=16"),
        (16, 4096, 16, "B=16, H=4096, R=16"),
        (32, 4096, 16, "B=32, H=4096, R=16"),
        (4, 4096, 8, "B=4, H=4096, R=8"),
        (4, 4096, 32, "B=4, H=4096, R=32"),
        (4, 2048, 16, "B=4, H=2048, R=16"),
        (4, 8192, 16, "B=4, H=8192, R=16"),
    ]

    print(f"Original PyTorch threads: {original_threads}")
    print("\n" + "=" * 60)
    print("Comparison 1: Single-threaded PyTorch (Fair Comparison)")
    print("=" * 60)

    # Set PyTorch to single-threaded for fair comparison
    torch.set_num_threads(1)

    print(f"{'Config':<25} {'AVX (ms)':<12} {'PyTorch-1T (ms)':<18} {'Ratio':<10} {'Status':<15}")
    print("-" * 80)

    single_thread_results = []

    for batch, hidden, rank, desc in configs:
        # Create tensors
        x = torch.randn(batch, hidden, dtype=torch.bfloat16)
        A = torch.randn(rank, hidden, dtype=torch.bfloat16)
        B = torch.randn(rank, hidden, dtype=torch.bfloat16)

        # Warm up AVX
        for _ in range(10):
            _ = batch_lora_avx(x, A, B, 0.1)

        # Benchmark AVX
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            out_avx = batch_lora_avx(x, A, B, 0.1)
        avx_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark PyTorch (single-threaded)
        x_f32 = x.float()
        A_f32 = A.float()
        B_f32 = B.float()

        start = time.perf_counter()
        for _ in range(iterations):
            intermediate = torch.matmul(x_f32, A_f32.T)
            out_pt = torch.matmul(intermediate, B_f32) * 0.1
        pytorch_time = (time.perf_counter() - start) / iterations * 1000

        ratio = avx_time / pytorch_time if pytorch_time > 0 else float('inf')
        status = "FASTER" if ratio < 1.0 else "SLOWER"

        print(f"{desc:<25} {avx_time:<12.3f} {pytorch_time:<18.3f} {ratio:<10.2f}x {status:<15}")

        single_thread_results.append((desc, avx_time, pytorch_time))

    # Restore original threads for multi-threaded comparison
    torch.set_num_threads(original_threads)

    print("\n" + "=" * 60)
    print("Comparison 2: Multi-threaded PyTorch (Absolute Throughput)")
    print("=" * 60)

    print(f"{'Config':<25} {'AVX (ms)':<12} {'PyTorch-MT (ms)':<18} {'Ratio':<10}")
    print("-" * 65)

    for batch, hidden, rank, desc in configs:
        # Create tensors
        x = torch.randn(batch, hidden, dtype=torch.bfloat16)
        A = torch.randn(rank, hidden, dtype=torch.bfloat16)
        B = torch.randn(rank, hidden, dtype=torch.bfloat16)

        # Warm up
        for _ in range(10):
            _ = batch_lora_avx(x, A, B, 0.1)

        # Benchmark AVX
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            out_avx = batch_lora_avx(x, A, B, 0.1)
        avx_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark PyTorch (multi-threaded)
        x_f32 = x.float()
        A_f32 = A.float()
        B_f32 = B.float()

        start = time.perf_counter()
        for _ in range(iterations):
            intermediate = torch.matmul(x_f32, A_f32.T)
            out_pt = torch.matmul(intermediate, B_f32) * 0.1
        pytorch_time = (time.perf_counter() - start) / iterations * 1000

        ratio = avx_time / pytorch_time if pytorch_time > 0 else float('inf')

        print(f"{desc:<25} {avx_time:<12.3f} {pytorch_time:<18.3f} {ratio:<10.2f}x")

    print("\n" + "=" * 60)
    print("Shadow Pipelining Analysis")
    print("=" * 60)

    # PCIe Gen4 x16 bandwidth: ~16 GB/s theoretical, ~12-14 GB/s practical
    # BF16 = 2 bytes per element
    pcie_bandwidth_gbs = 14.0  # GB/s (practical)

    print(f"PCIe Gen4 x16 Bandwidth: ~{pcie_bandwidth_gbs} GB/s")
    print(f"{'Config':<25} {'Data Size':<12} {'PCIe Time':<15} {'AVX Time':<12} {'Status':<15}")
    print("-" * 80)

    for batch, hidden, rank, desc in configs:
        # Data size: input + output BF16 tensors
        data_bytes = (batch * hidden * 2) * 2  # input + output, 2 bytes each
        pcie_time_us = (data_bytes / (pcie_bandwidth_gbs * 1e9)) * 1e6

        # Find AVX time from results
        avx_time = next((r[1] for r in single_thread_results if r[0] == desc), 0)
        avx_time_us = avx_time * 1000

        # Check if AVX can hide behind PCIe
        status = "CAN HIDE" if avx_time_us < pcie_time_us else "TOO SLOW"

        print(f"{desc:<25} {data_bytes/1024:<10.1f} KB {pcie_time_us:<15.1f} us {avx_time_us:<12.1f} us {status:<15}")

    print("\n" + "-" * 65)
    print("Interpretation:")
    print("  - 'CAN HIDE': AVX time < PCIe transfer time (good for Shadow Pipelining)")
    print("  - 'TOO SLOW': AVX time > PCIe transfer time (needs optimization)")
    print("  - Single-threaded comparison shows true AVX efficiency")


def run_individual_kernel_tests(verbose=False):
    """Test lora_down and lora_up separately."""
    print("\n" + "=" * 60)
    print("Individual Kernel Tests (lora_down, lora_up)")
    print("=" * 60)

    if not is_available():
        print("SKIP: AVX kernel not available")
        return False

    batch, hidden, rank = 4, 4096, 16

    # Test lora_down
    print("\n[Test] lora_down: x @ A.T")
    x = torch.randn(batch, hidden, dtype=torch.bfloat16)
    A = torch.randn(rank, hidden, dtype=torch.bfloat16)

    out_avx = lora_down_avx(x, A)
    out_ref = torch.matmul(x.float(), A.float().T).bfloat16()

    abs_diff = (out_avx.float() - out_ref.float()).abs()
    ref_max = out_ref.float().abs().max().item()
    rel_diff = abs_diff.max().item() / max(ref_max, 1e-6)
    passed = rel_diff < 0.02  # 2% relative tolerance
    print(f"  Result: {'PASS' if passed else 'FAIL'} (rel_diff={rel_diff*100:.2f}%)")

    # Test lora_up
    print("\n[Test] lora_up: x @ B")
    x = torch.randn(batch, rank, dtype=torch.bfloat16)
    B = torch.randn(rank, hidden, dtype=torch.bfloat16)

    out_avx = lora_up_avx(x, B)
    out_ref = torch.matmul(x.float(), B.float()).bfloat16()

    abs_diff = (out_avx.float() - out_ref.float()).abs()
    ref_max = out_ref.float().abs().max().item()
    rel_diff = abs_diff.max().item() / max(ref_max, 1e-6)
    passed = rel_diff < 0.02  # 2% relative tolerance
    print(f"  Result: {'PASS' if passed else 'FAIL'} (rel_diff={rel_diff*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test AVX-512 BF16 LoRA Kernel")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AVX-512 BF16 LoRA Kernel Tests")
    print("=" * 60)
    print(f"Kernel available: {is_available()}")

    if not is_available():
        print("\nERROR: AVX kernel not available!")
        print("Make sure you're running on a Sapphire Rapids or newer CPU.")
        sys.exit(1)

    # Run correctness tests
    success = run_correctness_tests(args.verbose)

    # Run individual kernel tests
    run_individual_kernel_tests(args.verbose)

    if args.benchmark:
        run_performance_benchmark(args.verbose)

    if not success:
        print("\nERROR: Some tests failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
