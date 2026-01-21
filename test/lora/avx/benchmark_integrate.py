#!/usr/bin/env python3
"""
COLoRA Full Suite Benchmark
---------------------------
This script compares two MoE-LoRA Serving architectures:
1. GPU-Centric (Baseline): Swap weights in via PCIe -> Compute on GPU.
   Cost = Weight Transfer (H2D) + GPU Compute
2. CPU-Centric (Ours): Swap activations out via PCIe -> Compute on CPU -> Swap results back.
   Cost = Activation Round-Trip (D2H + H2D) + CPU Compute

Usage:
   python3 benchmark_full_suite.py --batch-size 1 --hidden-size 4096 --lora-rank 16
"""

import argparse
import time
import torch
from torch.utils.cpp_extension import load_inline

# ==========================================
# 1. C++ Source for CPU AVX-512 Compute
# ==========================================
cpu_source = """
#include <torch/extension.h>
#include <immintrin.h>
#include <vector>
#include <omp.h>

// Optimized Down-Projection (x @ A)
void lora_down_kernel(float* input, float* weight, float* output, int batch_size, int hidden_dim, int rank) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float* x_ptr = input + b * hidden_dim;
        float* out_ptr = output + b * rank;
        
        __m512 acc = _mm512_setzero_ps();
        for (int k = 0; k < hidden_dim; k++) {
            __m512 vec_x = _mm512_set1_ps(x_ptr[k]);
            __m512 vec_w = _mm512_loadu_ps(weight + k * rank);
            acc = _mm512_fmadd_ps(vec_x, vec_w, acc);
        }
        _mm512_storeu_ps(out_ptr, acc);
    }
}

// Optimized Up-Projection (temp @ B)
void lora_up_kernel(float* input, float* weight, float* output, int batch_size, int hidden_dim, int rank) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float* in_ptr = input + b * rank;
        float* out_ptr = output + b * hidden_dim;

        for (int j = 0; j < hidden_dim; j += 16) {
             __m512 acc = _mm512_setzero_ps();
             for (int k = 0; k < rank; k++) {
                 __m512 vec_val = _mm512_set1_ps(in_ptr[k]);
                 __m512 vec_w = _mm512_loadu_ps(weight + k * hidden_dim + j);
                 acc = _mm512_fmadd_ps(vec_val, vec_w, acc);
             }
             _mm512_storeu_ps(out_ptr + j, acc);
        }
    }
}

// PyTorch Binding
torch::Tensor lora_avx_forward(torch::Tensor input, torch::Tensor A, torch::Tensor B) {
    int batch_size = input.size(0);
    int hidden_dim = input.size(1);
    int rank = A.size(1);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(torch::kCPU);
    auto temp = torch::zeros({batch_size, rank}, options);
    auto output = torch::zeros({batch_size, hidden_dim}, options);

    lora_down_kernel(input.data_ptr<float>(), A.data_ptr<float>(), temp.data_ptr<float>(), batch_size, hidden_dim, rank);
    lora_up_kernel(temp.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), batch_size, hidden_dim, rank);

    return output;
}
"""

# ==========================================
# 2. CUDA Source for PCIe Benchmarking
# ==========================================
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// 1. Measure One-Way Transfer (For Weights: H2D)
float measure_oneway_transfer(int payload_bytes, int iterations) {
    void* h_data; void* d_data;
    cudaMallocHost(&h_data, payload_bytes);
    cudaMalloc(&d_data, payload_bytes);
    memset(h_data, 1, payload_bytes);
    
    // Warmup
    for(int i=0; i<10; i++) cudaMemcpy(d_data, h_data, payload_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) {
        cudaMemcpy(d_data, h_data, payload_bytes, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    cudaFreeHost(h_data); cudaFree(d_data);
    return (total_ms / iterations) * 1000; // return us
}

// 2. Measure Round-Trip Transfer (For Activations: D2H + H2D)
float measure_roundtrip_transfer(int payload_bytes, int iterations) {
    void* h_data; void* d_data;
    cudaMallocHost(&h_data, payload_bytes);
    cudaMalloc(&d_data, payload_bytes);
    memset(h_data, 1, payload_bytes);
    
    // Warmup
    for(int i=0; i<10; i++) {
        cudaMemcpy(h_data, d_data, payload_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_data, h_data, payload_bytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) {
        // Simulate Activation Offload Ping-Pong
        cudaMemcpy(h_data, d_data, payload_bytes, cudaMemcpyDeviceToHost); // D2H
        cudaMemcpy(d_data, h_data, payload_bytes, cudaMemcpyHostToDevice); // H2D
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    cudaFreeHost(h_data); cudaFree(d_data);
    return (total_ms / iterations) * 1000; // return us
}
"""

def compile_modules():
    print(">>> [JIT] Compiling AVX-512 CPU Kernel...")
    cpu_ops = load_inline(
        name='lora_cpu_ops',
        cpp_sources=[cpu_source],
        functions=['lora_avx_forward'],
        extra_cflags=['-O3', '-march=native', '-fopenmp'],
        extra_ldflags=['-fopenmp'],
        with_cuda=False
    )
    
    print(">>> [JIT] Compiling CUDA PCIe Benchmark...")
    try:
        cuda_ops = load_inline(
            name='lora_cuda_ops',
            cpp_sources=[cuda_source],
            functions=['measure_oneway_transfer', 'measure_roundtrip_transfer'],
            with_cuda=True,
            extra_cuda_cflags=['-O3']
        )
    except Exception as e:
        print(f"CUDA Compilation Failed: {e}")
        cuda_ops = None
        
    return cpu_ops, cuda_ops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=2000)
    args = parser.parse_args()

    cpu_ops, cuda_ops = compile_modules()
    if cuda_ops is None: return

    print(f"\n===============================================================")
    print(f"COLoRA End-to-End Benchmark (Rank={args.lora_rank}, Batch={args.batch_size}, Hidden={args.hidden_size})")
    print(f"===============================================================")

    # -----------------------------------------------------------
    # 1. CPU Compute Benchmark (AVX-512)
    # -----------------------------------------------------------
    print("\n[Phase 1] Benchmarking CPU Compute (AVX-512)...")
    dtype = torch.float32
    x = torch.randn(args.batch_size, args.hidden_size, dtype=dtype)
    A = torch.randn(args.hidden_size, args.lora_rank, dtype=dtype)
    B = torch.randn(args.lora_rank, args.hidden_size, dtype=dtype)
    
    # Warmup
    for _ in range(20): _ = cpu_ops.lora_avx_forward(x, A, B)
    
    start = time.time()
    for _ in range(args.num_iters):
        _ = cpu_ops.lora_avx_forward(x, A, B)
    t_cpu_compute = ((time.time() - start) / args.num_iters) * 1e6 # us
    print(f"  -> CPU Compute Latency: \033[1;36m{t_cpu_compute:.2f} us\033[0m")

    # -----------------------------------------------------------
    # 2. GPU Baseline (Weights H2D + GPU Compute)
    # -----------------------------------------------------------
    print("\n[Phase 2] Benchmarking GPU Baseline (Load Weights + Compute)...")
    
    # A. Weight Transfer Time
    # Size = (Hidden*Rank + Rank*Hidden) * 2 bytes (BF16 simulation)
    # Note: We use 2 bytes to simulate real world BF16 transfer cost
    weight_payload = (args.hidden_size * args.lora_rank * 2) * 2 
    t_gpu_load = cuda_ops.measure_oneway_transfer(weight_payload, args.num_iters)
    
    # B. GPU Compute Time
    x_gpu = x.cuda(); A_gpu = A.cuda(); B_gpu = B.cuda()
    # Warmup
    for _ in range(20): 
        _ = torch.matmul(torch.matmul(x_gpu, A_gpu), B_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(args.num_iters):
        # Using PyTorch overhead as proxy for real Kernel Launch overhead
        _ = torch.matmul(torch.matmul(x_gpu, A_gpu), B_gpu)
    torch.cuda.synchronize()
    t_gpu_compute = ((time.time() - start) / args.num_iters) * 1e6 # us
    
    t_total_gpu = t_gpu_load + t_gpu_compute
    print(f"  -> Weight Transfer (H2D): {t_gpu_load:.2f} us ({weight_payload/1024:.1f} KB)")
    print(f"  -> GPU Compute + Launch:  {t_gpu_compute:.2f} us")
    print(f"  -> Total GPU Baseline:    \033[1;31m{t_total_gpu:.2f} us\033[0m")

    # -----------------------------------------------------------
    # 3. CPU Approach (Activations Round-Trip + CPU Compute)
    # -----------------------------------------------------------
    print("\n[Phase 3] Benchmarking COLoRA Approach (Activation Round-Trip + Compute)...")
    
    # Activation Payload = Batch * Hidden * 2 bytes (BF16)
    act_payload = args.batch_size * args.hidden_size * 2
    t_roundtrip = cuda_ops.measure_roundtrip_transfer(act_payload, args.num_iters)
    
    t_total_cpu_sync = t_roundtrip + t_cpu_compute
    
    print(f"  -> Activation Round-Trip: {t_roundtrip:.2f} us ({act_payload/1024:.1f} KB)")
    print(f"  -> CPU Compute:           {t_cpu_compute:.2f} us")
    print(f"  -> Total CPU (Sync):      \033[1;32m{t_total_cpu_sync:.2f} us\033[0m")

    # -----------------------------------------------------------
    # 4. Final Comparison
    # -----------------------------------------------------------
    print(f"\n===============================================================")
    print(f"FINAL RESULT SUMMARY")
    print(f"===============================================================")
    print(f"{'Metric':<25} | {'GPU Baseline':<15} | {'COLoRA (CPU)':<15}")
    print(f"--------------------------+-----------------+----------------")
    print(f"{'Data Movement':<25} | {t_gpu_load:<10.2f} us   | {t_roundtrip:<10.2f} us")
    print(f"{'Computation':<25} | {t_gpu_compute:<10.2f} us   | {t_cpu_compute:<10.2f} us")
    print(f"--------------------------+-----------------+----------------")
    print(f"{'End-to-End Latency':<25} | {t_total_gpu:<10.2f} us   | {t_total_cpu_sync:<10.2f} us")
    print(f"===============================================================")
    
    speedup = t_total_gpu / t_total_cpu_sync
    if t_total_cpu_sync < t_total_gpu:
        print(f"VICTORY: CPU is \033[1;32m{speedup:.2f}x FASTER\033[0m than GPU!")
    else:
        print(f"RESULT: CPU is slower. (Break-even point reached)")
        
    print(f"CSV_LINE: {args.batch_size},{t_total_gpu:.2f},{t_total_cpu_sync:.2f}")

if __name__ == "__main__":
    main()
