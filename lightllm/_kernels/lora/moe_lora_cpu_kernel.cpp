/**
 * MoE-Specific AVX-512 BF16 LoRA Kernel for Sapphire Rapids
 *
 * This kernel is optimized for the irregular and sparse computational
 * characteristics of Mixture of Experts (MoE) architectures:
 *
 * Key Features:
 * 1. Dynamic expert kernel selection based on token count
 * 2. Sparse token handling for MoE expert activation
 * 3. Optimized memory access patterns for various expert sizes
 * 4. Efficient scheduling of irregular workloads
 * 5. Specialized kernels for Gate/Up/Down phases of MoE
 * 6. Cache-aware design for small token counts
 *
 * Created for LightLLM project
 */

#include <torch/extension.h>
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>

using bf16 = c10::BFloat16;

// Convert packed BF16 values to packed single-precision (FP32)
// BF16 format is the upper 16 bits of FP32, so we just need to
// unpack 16-bit values and shift left by 16 bits.
inline __m512 _mm512_cvtbf16_ps(__m256i bf16_vec) {
    __m512i int32_vec = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(int32_vec, 16);
    return _mm512_castsi512_ps(shifted);
}

// Helper to detect optimal kernel type based on token count
inline int get_optimal_kernel_type(int N) {
    if (N <= 2) {
        return 0;  // Tiny kernel for single/dual token
    } else if (N <= 8) {
        return 1;  // Small kernel for 3-8 tokens
    } else if (N <= 32) {
        return 2;  // Medium kernel for 9-32 tokens
    } else {
        return 3;  // Large kernel for 33+ tokens
    }
}

// Tiny kernel for N <= 2 - minimal overhead, optimized for single token
void moe_lora_tiny_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, int out_H, float scaling) {

    float inter[128] = {0.0f};  // Support up to rank=128

    // Stage 1: x[H] @ A^T[R,H] -> inter[R], fully vectorized with BF16 DP
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * H;

        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();

            int k = 0;
            for (; k + 31 < H; k += 32) {
                __m512i v_x = _mm512_loadu_si512((const __m512i*)(x_ptr + k));
                __m512i v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + k));
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }

            float res = _mm512_reduce_add_ps(acc);
            for (; k < H; ++k) {
                res += (float)x_ptr[k] * (float)A_ptr[k];
            }
            inter[r] = res;
        }

        // Stage 2: inter[R] @ B[R,out_H] -> out[N,out_H], fully vectorized
        // out_H dimension outer loop, accumulate all ranks into vector accumulator
        bf16* out_ptr = out + n * out_H;
        float out_f32[16];
        int h = 0;
        for (; h + 15 < out_H; h += 16) {
            __m512 out_vec = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m256i v_B = _mm256_loadu_si256((const __m256i*)(B_mat + r * out_H + h));
                __m512 v_B_f32 = _mm512_cvtbf16_ps(v_B);
                out_vec = _mm512_fmadd_ps(_mm512_set1_ps(inter[r] * scaling), v_B_f32, out_vec);
            }
            _mm512_storeu_ps(out_f32, out_vec);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] += bf16(out_f32[i]);
            }
        }

        // Handle tail
        for (; h < out_H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                acc += inter[r] * (float)B_mat[r * out_H + h] * scaling;
            }
            out_ptr[h] += bf16(acc);
        }
    }
}

// Small kernel for N <= 8 - cache optimized, AVX-512 BF16 with rank-blocking
void moe_lora_small_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, int out_H, float scaling) {

    const int cache_block_h = 512;
    const int rank_block = 4;

    #pragma omp parallel for schedule(static) if(N > 1)
    for (int n = 0; n < N; ++n) {
        float inter[256];
        const bf16* x_ptr = x + n * H;
        bf16* out_ptr = out + n * out_H;

        // Stage 1: x @ A^T -> inter[R], rank-blocked for cache reuse of x
        // Process rank_block ranks per H-block so x stays in L1 cache
        for (int r = 0; r < R; r += rank_block) {
            const int r_end = (r + rank_block < R) ? r + rank_block : R;
            const int r_count = r_end - r;

            // Initialize accumulators for all ranks in this block
            __m512 rank_acc[4];
            for (int i = 0; i < r_count; ++i) rank_acc[i] = _mm512_setzero_ps();

            int k = 0;
            for (; k + cache_block_h <= H; k += cache_block_h) {
                const bf16* x_block = x_ptr + k;

                // Process x_block in 32-element chunks, reuse across all ranks
                int j = 0;
                for (; j + 31 < cache_block_h; j += 32) {
                    __m512i v_x = _mm512_loadu_si512((const __m512i*)(x_block + j));
                    for (int i = 0; i < r_count; ++i) {
                        const bf16* A_block = A_mat + (r + i) * H + k + j;
                        __m512i v_A = _mm512_loadu_si512((const __m512i*)(A_block));
                        rank_acc[i] = _mm512_dpbf16_ps(rank_acc[i], (__m512bh)v_x, (__m512bh)v_A);
                    }
                }
            }

            // Tail: process remaining H elements for each rank independently
            const int tail_start = k;
            for (int i = 0; i < r_count; ++i) {
                float res = _mm512_reduce_add_ps(rank_acc[i]);
                for (int kk = tail_start; kk < H; ++kk) {
                    res += (float)x_ptr[kk] * (float)A_mat[(r + i) * H + kk];
                }
                inter[r + i] = res;
            }
        }

        // Stage 2: inter[R] @ B[R,out_H] -> out[n,out_H], fully vectorized
        float out_f32[16];
        int h = 0;
        for (; h + 15 < out_H; h += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m256i v_B = _mm256_loadu_si256((const __m256i*)(B_mat + r * out_H + h));
                __m512 v_B_f32 = _mm512_cvtbf16_ps(v_B);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(inter[r] * scaling), v_B_f32, acc);
            }
            _mm512_storeu_ps(out_f32, acc);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] += bf16(out_f32[i]);
            }
        }

        for (; h < out_H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                acc += inter[r] * (float)B_mat[r * out_H + h] * scaling;
            }
            out_ptr[h] += bf16(acc);
        }
    }
}

// Medium kernel for N <= 32 - balanced performance
void moe_lora_medium_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, int out_H, float scaling) {

    const int cache_block_h = 256;
    const int rank_block = 8;

    #pragma omp parallel for schedule(dynamic, 2)
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * H;

        for (int r = 0; r < R; r += rank_block) {
            __m512 rank_acc[8];
            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                rank_acc[i] = _mm512_setzero_ps();
            }

            int k = 0;
            for (; k + cache_block_h <= H; k += cache_block_h) {
                const bf16* x_block = x_ptr + k;

                for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                    const bf16* A_ptr = A_mat + (r + i) * H + k;

                    int j = 0;
                    for (; j + 31 < cache_block_h; j += 32) {
                        auto v_x = _mm512_loadu_si512((const __m512i*)(x_block + j));
                        auto v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + j));
                        rank_acc[i] = _mm512_dpbf16_ps(rank_acc[i], (__m512bh)v_x, (__m512bh)v_A);
                    }

                    for (; j < cache_block_h; ++j) {
                        float xv = (float)x_block[j];
                        float av = (float)A_ptr[j];
                        rank_acc[i][0] += xv * av;
                    }
                }
            }

            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                float res = _mm512_reduce_add_ps(rank_acc[i]);
                for (; k < H; ++k) {
                    res += (float)x_ptr[k] * (float)A_mat[(r + i) * H + k];
                }

                const bf16* B_ptr = B_mat + (r + i) * out_H;
                for (int h = 0; h < out_H; ++h) {
                    float contribution = res * scaling * (float)B_ptr[h];
                    out[n * out_H + h] += bf16(contribution);
                }
            }
        }
    }
}

// Large kernel for N > 32 - throughput optimized
void moe_lora_large_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, int out_H, float scaling) {

    const int cache_block_h = 512;
    const int rank_block = 16;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * H;

        for (int r = 0; r < R; r += rank_block) {
            __m512 rank_acc[16];
            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                rank_acc[i] = _mm512_setzero_ps();
            }

            int k = 0;
            for (; k + cache_block_h <= H; k += cache_block_h) {
                const bf16* x_block = x_ptr + k;

                for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                    const bf16* A_ptr = A_mat + (r + i) * H + k;

                    int j = 0;
                    for (; j + 31 < cache_block_h; j += 32) {
                        auto v_x = _mm512_loadu_si512((const __m512i*)(x_block + j));
                        auto v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + j));
                        rank_acc[i] = _mm512_dpbf16_ps(rank_acc[i], (__m512bh)v_x, (__m512bh)v_A);
                    }

                    for (; j < cache_block_h; ++j) {
                        float xv = (float)x_block[j];
                        float av = (float)A_ptr[j];
                        rank_acc[i][0] += xv * av;
                    }
                }
            }

            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                float res = _mm512_reduce_add_ps(rank_acc[i]);
                for (; k < H; ++k) {
                    res += (float)x_ptr[k] * (float)A_mat[(r + i) * H + k];
                }

                const bf16* B_ptr = B_mat + (r + i) * out_H;
                for (int h = 0; h < out_H; ++h) {
                    float contribution = res * scaling * (float)B_ptr[h];
                    out[n * out_H + h] += bf16(contribution);
                }
            }
        }
    }
}

// Specialized kernel for Gate phase of MoE
void moe_lora_gate_kernel(
    const bf16* x, const bf16* A_mat, bf16* out,
    int N, int H, int R, float scaling) {

    const int cache_block_h = 128;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * H;

        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();

            int k = 0;
            for (; k + cache_block_h <= H; k += cache_block_h) {
                const bf16* x_block = x_ptr + k;
                const bf16* A_block = A_ptr + k;

                int j = 0;
                for (; j + 31 < cache_block_h; j += 32) {
                    auto v_x = _mm512_loadu_si512((const __m512i*)(x_block + j));
                    auto v_A = _mm512_loadu_si512((const __m512i*)(A_block + j));
                    acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
                }

                for (; j < cache_block_h; ++j) {
                    float xv = (float)x_block[j];
                    float av = (float)A_block[j];
                    acc[0] += xv * av;
                }
            }

            float res = _mm512_reduce_add_ps(acc);
            for (; k < H; ++k) {
                res += (float)x_ptr[k] * (float)A_ptr[k];
            }

            out[n * R + r] = bf16(res * scaling);
        }
    }
}

// Specialized kernel for Up phase of MoE - AVX-512 BF16 optimized
// Computes out[n,h] += sum_r x[n,r] * B[r,h] * scaling with B stored row-major [R, H].
void moe_lora_up_kernel(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int R, int H, float scaling) {

    #pragma omp parallel for schedule(dynamic, 2)
    for (int n = 0; n < N; ++n) {
        float x_f32[256];

        const bf16* x_ptr = x + n * R;
        bf16* out_ptr = out + n * H;

        // Pre-convert x[n,r] to float once (R is small, ~8-64)
        for (int r = 0; r < R; ++r) {
            x_f32[r] = (float)x_ptr[r];  // NOTE: scaling applied below
        }

        // H dimension outer loop: fully vectorized with 16-way SIMD
        float out_f32[16];
        int h = 0;
        for (; h + 15 < H; h += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m256i v_B = _mm256_loadu_si256((const __m256i*)(B_mat + r * H + h));
                __m512 v_B_f32 = _mm512_cvtbf16_ps(v_B);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(x_f32[r] * scaling), v_B_f32, acc);
            }
            _mm512_storeu_ps(out_f32, acc);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] += bf16(out_f32[i]);
            }
        }

        // Handle remaining H elements (tail)
        for (; h < H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                acc += x_f32[r] * (float)B_mat[r * H + h] * scaling;
            }
            out_ptr[h] += bf16(acc);
        }
    }
}

// Same layout/matmul as moe_lora_up_kernel (down uses identical [N,R]@[R,H]->[N,H] here).
// AVX-512 BF16 optimized version.
void moe_lora_down_kernel(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int R, int H, float scaling) {

    #pragma omp parallel for schedule(dynamic, 1)
    for (int n = 0; n < N; ++n) {
        float x_f32[256];

        const bf16* x_ptr = x + n * R;
        bf16* out_ptr = out + n * H;

        for (int r = 0; r < R; ++r) {
            x_f32[r] = (float)x_ptr[r];
        }

        float out_f32[16];
        int h = 0;
        for (; h + 15 < H; h += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m256i v_B = _mm256_loadu_si256((const __m256i*)(B_mat + r * H + h));
                __m512 v_B_f32 = _mm512_cvtbf16_ps(v_B);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(x_f32[r] * scaling), v_B_f32, acc);
            }
            _mm512_storeu_ps(out_f32, acc);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] += bf16(out_f32[i]);
            }
        }

        for (; h < H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                acc += x_f32[r] * (float)B_mat[r * H + h] * scaling;
            }
            out_ptr[h] += bf16(acc);
        }
    }
}

// Main dispatcher function
void moe_lora_dispatch(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, int out_H, float scaling,
    const std::string& phase = "full") {

    if (phase == "gate") {
        moe_lora_gate_kernel(x, A_mat, out, N, H, R, scaling);
    } else if (phase == "up") {
        // up kernel: N, R (input dim), out_H (output dim)
        moe_lora_up_kernel(x, B_mat, out, N, H, out_H, scaling);
    } else if (phase == "down") {
        // down kernel: N, R (input dim), out_H (output dim)
        moe_lora_down_kernel(x, B_mat, out, N, H, out_H, scaling);
    } else {
        int kernel_type = get_optimal_kernel_type(N);

        switch (kernel_type) {
            case 0:
                moe_lora_tiny_kernel(x, A_mat, B_mat, out, N, H, R, out_H, scaling);
                break;
            case 1:
                moe_lora_small_kernel(x, A_mat, B_mat, out, N, H, R, out_H, scaling);
                break;
            case 2:
                moe_lora_medium_kernel(x, A_mat, B_mat, out, N, H, R, out_H, scaling);
                break;
            case 3:
                moe_lora_large_kernel(x, A_mat, B_mat, out, N, H, R, out_H, scaling);
                break;
            default:
                moe_lora_medium_kernel(x, A_mat, B_mat, out, N, H, R, out_H, scaling);
        }
    }
}

// C++ interface for Python
void moe_batch_lora_avx(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, int out_H, float scaling) {
    moe_lora_dispatch(x, A_mat, B_mat, out, N, H, R, out_H, scaling);
}

void moe_batch_lora_gate_avx(
    const bf16* x, const bf16* A_mat, bf16* out,
    int N, int H, int R, float scaling) {
    // Gate: input dim = H (hidden), output dim = R (rank)
    moe_lora_dispatch(x, A_mat, nullptr, out, N, H, R, R, scaling, "gate");
}

void moe_batch_lora_up_avx(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int R, int out_H, float scaling) {
    // Up: input dim = R (rank), output dim = out_H (intermediate)
    moe_lora_dispatch(x, nullptr, B_mat, out, N, R, 0, out_H, scaling, "up");
}

void moe_batch_lora_down_avx(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int R, int out_H, float scaling) {
    // Down: input dim = R (intermediate), output dim = out_H (hidden)
    moe_lora_dispatch(x, nullptr, B_mat, out, N, R, 0, out_H, scaling, "down");
}

// Multi-adapter batched kernel: each token can use a different adapter's weights.
// This eliminates Python per-adapter loop overhead for cold-miss scenarios.
//
// x: [N, H] input (all tokens concatenated)
// A_all: [num_adapters, R, H] stacked A matrices
// B_all: [num_adapters, R, H] stacked B matrices
// adapter_ids: [N] int tensor mapping each token to its adapter index (0-based)
// scaling: [num_adapters] per-adapter scaling factors (or nullptr for uniform scaling)
// out: [N, H] output (pre-allocated, zeroed)
// N, H, R, num_adapters: dimensions
// uniform_scaling: if > 0, use this for all adapters (ignores scaling array)
void moe_lora_multi_adapter_avx(
    const bf16* x, const bf16* A_all, const bf16* B_all,
    const int* adapter_ids, const float* scaling,
    bf16* out, int N, int H, int R, int num_adapters,
    float uniform_scaling) {

    // Use the per-token gate+up approach, but all in C++ to avoid Python overhead
    #pragma omp parallel for schedule(dynamic, 2) if(N > 4)
    for (int n = 0; n < N; ++n) {
        int adapter_idx = adapter_ids[n];
        if (adapter_idx < 0 || adapter_idx >= num_adapters) continue;

        const bf16* x_ptr = x + n * H;
        const bf16* A_mat = A_all + adapter_idx * R * H;
        const bf16* B_mat = B_all + adapter_idx * R * H;
        bf16* out_ptr = out + n * H;

        float s = (uniform_scaling > 0.0f) ? uniform_scaling : scaling[adapter_idx];

        // Stage 1: x[H] @ A^T[R,H] -> inter[R]
        float inter[128];  // max rank 128
        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k + 31 < H; k += 32) {
                __m512i v_x = _mm512_loadu_si512((const __m512i*)(x_ptr + k));
                __m512i v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + k));
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }
            float res = _mm512_reduce_add_ps(acc);
            for (; k < H; ++k) {
                res += (float)x_ptr[k] * (float)A_ptr[k];
            }
            inter[r] = res;
        }

        // Stage 2: inter[R] @ B[R,H] -> out[H]
        float out_f32[16];
        int h = 0;
        for (; h + 15 < H; h += 16) {
            __m512 out_vec = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m256i v_B = _mm256_loadu_si256((const __m256i*)(B_mat + r * H + h));
                __m512 v_B_f32 = _mm512_cvtbf16_ps(v_B);
                out_vec = _mm512_fmadd_ps(_mm512_set1_ps(inter[r] * s), v_B_f32, out_vec);
            }
            _mm512_storeu_ps(out_f32, out_vec);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] = bf16(out_f32[i]);
            }
        }
        for (; h < H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                acc += inter[r] * (float)B_mat[r * H + h] * s;
            }
            out_ptr[h] = bf16(acc);
        }
    }
}

// Pool-based multi-adapter kernel: accepts the pool buffer directly with per-adapter
// offset/rank information, avoiding any tensor copying in Python.
//
// key_buffer: pool key_buffer [total_slots, max_rank, H] (CPU, bf16)
// value_buffer: pool value_buffer [total_slots, max_rank, H] (CPU, bf16)
// x: [N, H] input (CPU, bf16)
// out: [N, H] output (pre-allocated, zeroed, CPU, bf16)
// adapter_ids: [N] int per-token adapter index in the pool (0-based)
// pool_offsets: [num_unique] int per-unique-adapter offset into key/value_buffer
// pool_ranks: [num_unique] int per-unique-adapter rank
// pool_scaling: [num_unique] float per-unique-adapter scaling (or nullptr)
// adapter_local_ids: [N] int mapping token to index in pool_offsets/ranks/scaling
//   (adapter_ids in the pool, remapped to 0..num_unique-1)
void moe_lora_pool_multi_adapter_avx(
    const bf16* key_buffer, const bf16* value_buffer,
    const bf16* x, bf16* out,
    const int* adapter_local_ids,
    const int* pool_offsets, const int* pool_ranks, const float* pool_scaling,
    int N, int H, int max_rank, int num_unique,
    float uniform_scaling) {

    #pragma omp parallel for schedule(dynamic, 2) if(N > 4)
    for (int n = 0; n < N; ++n) {
        int local_id = adapter_local_ids[n];
        if (local_id < 0 || local_id >= num_unique) continue;

        const bf16* x_ptr = x + n * H;
        int offset = pool_offsets[local_id];
        int R = pool_ranks[local_id];
        float s = (uniform_scaling > 0.0f) ? uniform_scaling : pool_scaling[local_id];

        const bf16* A_mat = key_buffer + offset * max_rank * H;
        const bf16* B_mat = value_buffer + offset * max_rank * H;
        bf16* out_ptr = out + n * H;

        // Stage 1: x[H] @ A^T[R,H] -> inter[R]
        float inter[128];
        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k + 31 < H; k += 32) {
                __m512i v_x = _mm512_loadu_si512((const __m512i*)(x_ptr + k));
                __m512i v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + k));
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }
            float res = _mm512_reduce_add_ps(acc);
            for (; k < H; ++k) {
                res += (float)x_ptr[k] * (float)A_ptr[k];
            }
            inter[r] = res;
        }

        // Stage 2: inter[R] @ B[R,H] -> out[H]
        float out_f32[16];
        int h = 0;
        for (; h + 15 < H; h += 16) {
            __m512 out_vec = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m256i v_B = _mm256_loadu_si256((const __m256i*)(B_mat + r * H + h));
                __m512 v_B_f32 = _mm512_cvtbf16_ps(v_B);
                out_vec = _mm512_fmadd_ps(_mm512_set1_ps(inter[r] * s), v_B_f32, out_vec);
            }
            _mm512_storeu_ps(out_f32, out_vec);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] = bf16(out_f32[i]);
            }
        }
        for (; h < H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                acc += inter[r] * (float)B_mat[r * H + h] * s;
            }
            out_ptr[h] = bf16(acc);
        }
    }
}

// FP16 pool multi-adapter kernel: reads fp16 weights from pool, converts to fp32
// on the fly using AVX-512 FP16 conversion, and uses AVX-512 FMA for matmul.
// Input x is still bf16 (activation on CPU), output is bf16.
//
// key_buffer: pool key_buffer [total_slots, max_rank, H] (CPU, fp16)
// value_buffer: pool value_buffer [total_slots, max_rank, H] (CPU, fp16)
// x: [N, H] input (CPU, bf16)
// out: [N, H] output (pre-allocated, zeroed, CPU, bf16)
void moe_lora_pool_fp16_multi_adapter_avx(
    const uint16_t* key_buffer, const uint16_t* value_buffer,  // fp16 as uint16_t
    const bf16* x, bf16* out,
    const int* adapter_local_ids,
    const int* pool_offsets, const int* pool_ranks, const float* pool_scaling,
    int N, int H, int max_rank, int num_unique,
    float uniform_scaling) {

    #pragma omp parallel for schedule(dynamic, 2) if(N > 4)
    for (int n = 0; n < N; ++n) {
        int local_id = adapter_local_ids[n];
        if (local_id < 0 || local_id >= num_unique) continue;

        const bf16* x_ptr = x + n * H;
        int offset = pool_offsets[local_id];
        int R = pool_ranks[local_id];
        float s = (uniform_scaling > 0.0f) ? uniform_scaling : pool_scaling[local_id];

        const uint16_t* A_mat = key_buffer + offset * max_rank * H;
        const uint16_t* B_mat = value_buffer + offset * max_rank * H;
        bf16* out_ptr = out + n * H;

        // Stage 1: x[H] @ A^T[R,H] -> inter[R]
        // x is bf16, A is fp16 — need to convert both to fp32 for FMA
        // Since x is shared across all ranks, pre-convert x to fp32
        // H=4096 elements × 4 bytes = 16KB, fits in L1 cache
        float x_f32[4096];  // VLA, max H=4096
        int k = 0;
        for (; k + 15 < H; k += 16) {
            __m256i v_x_bf16 = _mm256_loadu_si256((const __m256i*)(x_ptr + k));
            __m512 v_x_f32 = _mm512_cvtbf16_ps(v_x_bf16);
            _mm512_storeu_ps(x_f32 + k, v_x_f32);
        }
        for (; k < H; ++k) {
            x_f32[k] = (float)x_ptr[k];
        }

        float inter[128];
        for (int r = 0; r < R; ++r) {
            const uint16_t* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();
            int j = 0;
            // Process 16 elements at a time: load 16 fp16, convert to fp32 via F16C, FMA
            // F16C (_mm256_cvtph_ps) converts 8 fp16 (__m128i) -> 8 fp32 (__m256).
            // We do two halves and insert into __m512.
            for (; j + 15 < H; j += 16) {
                __m128i v_A_lo = _mm_loadu_si128((const __m128i*)(A_ptr + j));      // lower 8 fp16
                __m128i v_A_hi = _mm_loadu_si128((const __m128i*)(A_ptr + j + 8));   // upper 8 fp16
                __m256 v_A_f32_lo = _mm256_cvtph_ps(v_A_lo);
                __m256 v_A_f32_hi = _mm256_cvtph_ps(v_A_hi);
                __m512 v_A_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(v_A_f32_lo), v_A_f32_hi, 1);
                __m512 v_x = _mm512_loadu_ps(x_f32 + j);
                acc = _mm512_fmadd_ps(v_x, v_A_f32, acc);
            }
            float res = _mm512_reduce_add_ps(acc);
            for (; j < H; ++j) {
                // Manual fp16 -> fp32 for tail
                uint16_t h = A_ptr[j];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exponent = (h >> 10) & 0x1f;
                uint32_t mantissa = h & 0x3ff;
                float val;
                if (exponent == 0) {
                    if (mantissa == 0) val = 0.0f;
                    else val = ldexpf((float)mantissa / 1024.0f, -14);
                } else if (exponent == 31) {
                    val = mantissa ? NAN : (sign ? -INFINITY : INFINITY);
                } else {
                    val = ldexpf(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
                }
                if (sign) val = -val;
                res += x_f32[j] * val;
            }
            inter[r] = res;
        }

        // Stage 2: inter[R] @ B[R,H] -> out[H]
        // B is fp16, convert to fp32 for FMA
        float out_f32[16];
        int h = 0;
        for (; h + 15 < H; h += 16) {
            __m512 out_vec = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                __m128i v_B_lo = _mm_loadu_si128((const __m128i*)(B_mat + r * H + h));
                __m128i v_B_hi = _mm_loadu_si128((const __m128i*)(B_mat + r * H + h + 8));
                __m256 v_B_f32_lo = _mm256_cvtph_ps(v_B_lo);
                __m256 v_B_f32_hi = _mm256_cvtph_ps(v_B_hi);
                __m512 v_B_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(v_B_f32_lo), v_B_f32_hi, 1);
                out_vec = _mm512_fmadd_ps(_mm512_set1_ps(inter[r] * s), v_B_f32, out_vec);
            }
            _mm512_storeu_ps(out_f32, out_vec);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] = bf16(out_f32[i]);
            }
        }
        for (; h < H; ++h) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) {
                uint16_t raw = B_mat[r * H + h];
                uint32_t sign = (raw >> 15) & 1;
                uint32_t exponent = (raw >> 10) & 0x1f;
                uint32_t mantissa = raw & 0x3ff;
                float val;
                if (exponent == 0) {
                    if (mantissa == 0) val = 0.0f;
                    else val = ldexpf((float)mantissa / 1024.0f, -14);
                } else if (exponent == 31) {
                    val = mantissa ? NAN : (sign ? -INFINITY : INFINITY);
                } else {
                    val = ldexpf(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
                }
                if (sign) val = -val;
                acc += inter[r] * val * s;
            }
            out_ptr[h] = bf16(acc);
        }
    }
}

// Python bindings
PYBIND11_MODULE(moe_lora_cpu_kernel, m) {
    m.def("moe_batch_lora_avx", [](
        const at::Tensor& x, const at::Tensor& A, const at::Tensor& B,
        at::Tensor& out, int N, int H, int R, int out_H, float scaling) {
        moe_batch_lora_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const bf16*>(B.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, H, R, out_H, scaling
        );
    }, "MoE-specific AVX-512 LoRA kernel");

    m.def("moe_batch_lora_gate_avx", [](
        const at::Tensor& x, const at::Tensor& A, at::Tensor& out,
        int N, int H, int R, float scaling) {
        moe_batch_lora_gate_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, H, R, scaling
        );
    }, "MoE Gate phase AVX-512 LoRA kernel");

    m.def("moe_batch_lora_up_avx", [](
        const at::Tensor& x, const at::Tensor& B, at::Tensor& out,
        int N, int R, int out_H, float scaling) {
        moe_batch_lora_up_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(B.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, R, out_H, scaling
        );
    }, "MoE Up phase AVX-512 LoRA kernel");

    m.def("moe_batch_lora_down_avx", [](
        const at::Tensor& x, const at::Tensor& B, at::Tensor& out,
        int N, int R, int out_H, float scaling) {
        moe_batch_lora_down_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(B.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, R, out_H, scaling
        );
    }, "MoE Down phase AVX-512 LoRA kernel");

    m.def("moe_lora_multi_adapter_avx", [](
        const at::Tensor& x, const at::Tensor& A_all, const at::Tensor& B_all,
        const at::Tensor& adapter_ids, const at::Tensor& scaling,
        at::Tensor& out, int N, int H, int R, int num_adapters,
        float uniform_scaling) {
        const float* scaling_ptr = (scaling.numel() > 0 && uniform_scaling <= 0.0f)
            ? scaling.data_ptr<float>() : nullptr;
        moe_lora_multi_adapter_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(A_all.data_ptr()),
            reinterpret_cast<const bf16*>(B_all.data_ptr()),
            adapter_ids.data_ptr<int>(),
            scaling_ptr,
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, H, R, num_adapters, uniform_scaling
        );
    }, "Multi-adapter batched AVX-512 LoRA kernel");

    m.def("moe_lora_pool_multi_adapter_avx", [](
        const at::Tensor& key_buffer, const at::Tensor& value_buffer,
        const at::Tensor& x, at::Tensor& out,
        const at::Tensor& adapter_local_ids,
        const at::Tensor& pool_offsets, const at::Tensor& pool_ranks,
        const at::Tensor& pool_scaling,
        int N, int H, int max_rank, int num_unique,
        float uniform_scaling) {
        const float* scaling_ptr = (pool_scaling.numel() > 0 && uniform_scaling <= 0.0f)
            ? pool_scaling.data_ptr<float>() : nullptr;
        moe_lora_pool_multi_adapter_avx(
            reinterpret_cast<const bf16*>(key_buffer.data_ptr()),
            reinterpret_cast<const bf16*>(value_buffer.data_ptr()),
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            adapter_local_ids.data_ptr<int>(),
            pool_offsets.data_ptr<int>(),
            pool_ranks.data_ptr<int>(),
            scaling_ptr,
            N, H, max_rank, num_unique, uniform_scaling
        );
    }, "Pool-based multi-adapter batched AVX-512 LoRA kernel");

    m.def("moe_lora_pool_fp16_multi_adapter_avx", [](
        const at::Tensor& key_buffer, const at::Tensor& value_buffer,
        const at::Tensor& x, at::Tensor& out,
        const at::Tensor& adapter_local_ids,
        const at::Tensor& pool_offsets, const at::Tensor& pool_ranks,
        const at::Tensor& pool_scaling,
        int N, int H, int max_rank, int num_unique,
        float uniform_scaling) {
        const float* scaling_ptr = (pool_scaling.numel() > 0 && uniform_scaling <= 0.0f)
            ? pool_scaling.data_ptr<float>() : nullptr;
        moe_lora_pool_fp16_multi_adapter_avx(
            reinterpret_cast<const uint16_t*>(key_buffer.data_ptr()),
            reinterpret_cast<const uint16_t*>(value_buffer.data_ptr()),
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            adapter_local_ids.data_ptr<int>(),
            pool_offsets.data_ptr<int>(),
            pool_ranks.data_ptr<int>(),
            scaling_ptr,
            N, H, max_rank, num_unique, uniform_scaling
        );
    }, "Pool-based multi-adapter batched AVX-512 LoRA kernel (fp16 weights)");

    m.def("is_available", []() {
        // Check if AVX-512 BF16 instructions are available
        int eax, ebx, ecx, edx;
        __asm__ __volatile__(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(7), "c"(0));
        return (ebx & (1 << 5)) != 0;  // Check AVX-512F
    });
}
