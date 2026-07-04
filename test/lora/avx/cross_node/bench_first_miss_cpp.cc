// bench_first_miss_cpp.cc
// C++ compute-only benchmark: measures per-miss first/rest GPU time ratio
// for the same LoRA compute as the Python path (x @ A.T @ B per miss).
//
// Purpose: determine if the ~1.79x first-miss tax is Python/PyTorch dispatch
// or GPU-side (cuBLAS init, allocator, cache). If C++ shows ~1.0x, it's Python.
// If ~1.74x, it's GPU-side.
//
// Build: nvcc -O2 -o bench_first_miss_cpp bench_first_miss_cpp.cc -lcublas
// Run:   ./bench_first_miss_cpp [nm] [rank] [hidden] [intermediate] [iters]

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#define CHECK_CUDA(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    int nm = (argc > 1) ? atoi(argv[1]) : 8;
    int rank = (argc > 2) ? atoi(argv[2]) : 64;
    int hidden = (argc > 3) ? atoi(argv[3]) : 2048;
    int intermediate = (argc > 4) ? atoi(argv[4]) : 2048;
    int iters = (argc > 5) ? atoi(argv[5]) : 50;

    printf("=== C++ Compute-Only First-Miss Benchmark ===\n");
    printf("NM=%d rank=%d hidden=%d intermediate=%d iters=%d\n",
           nm, rank, hidden, intermediate, iters);

    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate activation (1 x hidden, float32)
    float* d_act;
    CHECK_CUDA(cudaMalloc(&d_act, hidden * sizeof(float)));

    // Allocate per-miss weights and intermediates
    std::vector<float*> d_A(nm), d_B(nm), d_inter(nm), d_out(nm);
    for (int i = 0; i < nm; i++) {
        CHECK_CUDA(cudaMalloc(&d_A[i], rank * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], rank * intermediate * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_inter[i], rank * sizeof(float)));      // [1, rank]
        CHECK_CUDA(cudaMalloc(&d_out[i], intermediate * sizeof(float))); // [1, intermediate]
    }

    // Initialize with random data (same as Python torch.randn)
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, d_act, hidden);
    for (int i = 0; i < nm; i++) {
        curandGenerateUniform(gen, d_A[i], rank * hidden);
        curandGenerateUniform(gen, d_B[i], rank * intermediate);
    }

    // Warmup (1 iteration, not timed)
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < nm; i++) {
        // inter = act @ A.T  ->  [1, rank] = [1, hidden] @ [hidden, rank]
        // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
        // op(A) = act [1, hidden] (no transpose), op(B) = A.T [hidden, rank] (transpose)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  rank, 1, hidden,
                                  &alpha,
                                  d_A[i], rank,        // A [rank, hidden] in col-major
                                  d_act, hidden,        // act [hidden, 1] in col-major
                                  &beta,
                                  d_inter[i], rank));   // out [rank, 1]
        // out = inter @ B  ->  [1, intermediate] = [1, rank] @ [rank, intermediate]
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  intermediate, 1, rank,
                                  &alpha,
                                  d_B[i], intermediate,  // B [intermediate, rank] in col-major
                                  d_inter[i], rank,       // inter [rank, 1]
                                  &beta,
                                  d_out[i], intermediate));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations
    std::vector<double> first_gpu_us(iters), rest_gpu_us(iters * (nm - 1));
    cudaEvent_t ev_start, ev_end;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_end));

    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < nm; i++) {
            CHECK_CUDA(cudaEventRecord(ev_start));
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                      rank, 1, hidden, &alpha,
                                      d_A[i], rank, d_act, hidden, &beta,
                                      d_inter[i], rank));
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      intermediate, 1, rank, &alpha,
                                      d_B[i], intermediate, d_inter[i], rank, &beta,
                                      d_out[i], intermediate));
            CHECK_CUDA(cudaEventRecord(ev_end));
            CHECK_CUDA(cudaEventSynchronize(ev_end));

            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_end));
            double us = ms * 1000.0;
            if (i == 0)
                first_gpu_us[iter] = us;
            else
                rest_gpu_us[iter * (nm - 1) + (i - 1)] = us;
        }
    }

    // Compute statistics
    double first_sum = 0, rest_sum = 0;
    for (double v : first_gpu_us) first_sum += v;
    for (double v : rest_gpu_us) rest_sum += v;
    double first_mean = first_sum / iters;
    double rest_mean = rest_sum / (iters * (nm - 1));

    // Median
    std::sort(first_gpu_us.begin(), first_gpu_us.end());
    std::sort(rest_gpu_us.begin(), rest_gpu_us.end());
    double first_med = first_gpu_us[iters / 2];
    double rest_med = rest_gpu_us[rest_gpu_us.size() / 2];

    printf("\nResults (per-miss GPU time):\n");
    printf("  first_miss GPU: mean=%.1fus  median=%.1fus\n", first_mean, first_med);
    printf("  rest_miss  GPU: mean=%.1fus  median=%.1fus\n", rest_mean, rest_med);
    printf("  first/rest ratio: mean=%.2fx  median=%.2fx\n",
           first_mean / rest_mean, first_med / rest_med);

    // Per-iteration first-miss breakdown (first 5 iterations)
    printf("\nFirst 5 iterations:\n");
    for (int iter = 0; iter < 5 && iter < iters; iter++) {
        printf("  iter %d: first=%.1fus", iter, first_gpu_us[iter]);
        if (nm > 1) {
            double r = 0;
            for (int i = 1; i < nm; i++)
                r += rest_gpu_us[iter * (nm - 1) + (i - 1)];
            r /= (nm - 1);
            printf("  rest_avg=%.1fus  ratio=%.2fx", r, first_gpu_us[iter] / r);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_end));
    curandDestroyGenerator(gen);
    cublasDestroy(handle);
    for (int i = 0; i < nm; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_inter[i]);
        cudaFree(d_out[i]);
    }
    cudaFree(d_act);

    return 0;
}
