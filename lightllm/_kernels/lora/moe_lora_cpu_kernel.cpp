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

// Tiny kernel for N <= 2 - minimal overhead
void moe_lora_tiny_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, float scaling) {

    // Use register blocking for minimal memory usage
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * H;

        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();

            int k = 0;
            for (; k + 31 < H; k += 32) {
                auto v_x = _mm512_loadu_si512((const __m512i*)(x_ptr + k));
                auto v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + k));
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }

            float res = _mm512_reduce_add_ps(acc);
            for (; k < H; ++k) {
                res += (float)x_ptr[k] * (float)A_ptr[k];
            }

            // Up projection for this rank
            const bf16* B_ptr = B_mat + r * H;
            __m512 up_acc = _mm512_setzero_ps();

            int h = 0;
            for (; h + 15 < H; h += 16) {
                auto v_B = _mm256_loadu_si256((const __m256i*)(B_ptr + h));
                up_acc = _mm512_fmadd_ps(_mm512_set1_ps(res * scaling),
                                        _mm512_cvtph_ps(v_B), up_acc);
            }

            float up_res_arr[16];
            _mm256_storeu_ps(up_res_arr, _mm512_castps512_ps256(up_acc));
            for (int i = 0; i < 16; ++i) {
                if (h + i < H) {
                    out[n * H + (h + i)] += bf16(up_res_arr[i]);
                }
            }

            for (; h < H; ++h) {
                out[n * H + h] += bf16(res * scaling * (float)B_ptr[h]);
            }
        }
    }
}

// Small kernel for N <= 8 - cache optimized
void moe_lora_small_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, float scaling) {

    const int cache_block_h = 128;

    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * H;

        __m512 rank_acc[16];  // Rank blocking
        for (int r = 0; r < std::min(R, 16); ++r) {
            rank_acc[r] = _mm512_setzero_ps();
        }

        int k = 0;
        for (; k + cache_block_h <= H; k += cache_block_h) {
            const bf16* x_block = x_ptr + k;

            for (int r = 0; r < std::min(R, 16); ++r) {
                const bf16* A_ptr = A_mat + r * H + k;

                int j = 0;
                for (; j + 31 < cache_block_h; j += 32) {
                    auto v_x = _mm512_loadu_si512((const __m512i*)(x_block + j));
                    auto v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + j));
                    rank_acc[r] = _mm512_dpbf16_ps(rank_acc[r], (__m512bh)v_x, (__m512bh)v_A);
                }

                for (; j < cache_block_h; ++j) {
                    float xv = (float)x_block[j];
                    float av = (float)A_ptr[j];
                    rank_acc[r][0] += xv * av;
                }
            }
        }

        // Process remaining H
        for (int r = 0; r < std::min(R, 16); ++r) {
            float res = _mm512_reduce_add_ps(rank_acc[r]);
            for (; k < H; ++k) {
                res += (float)x_ptr[k] * (float)A_mat[r * H + k];
            }

            // Up projection
            const bf16* B_ptr = B_mat + r * H;
            for (int h = 0; h < H; ++h) {
                float contribution = res * scaling * (float)B_ptr[h];
                out[n * H + h] += bf16(contribution);
            }
        }

        // Handle remaining ranks
        for (int r = 16; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();

            int kh = 0;
            for (; kh + 31 < H; kh += 32) {
                auto v_x = _mm512_loadu_si512((const __m512i*)(x_ptr + kh));
                auto v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + kh));
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }

            float res = _mm512_reduce_add_ps(acc);
            for (; kh < H; ++kh) {
                res += (float)x_ptr[kh] * (float)A_ptr[kh];
            }

            const bf16* B_ptr = B_mat + r * H;
            for (int h = 0; h < H; ++h) {
                float contribution = res * scaling * (float)B_ptr[h];
                out[n * H + h] += bf16(contribution);
            }
        }
    }
}

// Medium kernel for N <= 32 - balanced performance
void moe_lora_medium_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, float scaling) {

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

                const bf16* B_ptr = B_mat + (r + i) * H;
                for (int h = 0; h < H; ++h) {
                    float contribution = res * scaling * (float)B_ptr[h];
                    out[n * H + h] += bf16(contribution);
                }
            }
        }
    }
}

// Large kernel for N > 32 - throughput optimized
void moe_lora_large_kernel(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, float scaling) {

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

                const bf16* B_ptr = B_mat + (r + i) * H;
                for (int h = 0; h < H; ++h) {
                    float contribution = res * scaling * (float)B_ptr[h];
                    out[n * H + h] += bf16(contribution);
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

// Specialized kernel for Up phase of MoE
void moe_lora_up_kernel(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int R, int H, float scaling) {

    const int cache_block_h = 256;
    const int rank_block = 8;

    #pragma omp parallel for schedule(dynamic, 2)
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * R;

        for (int r = 0; r < R; r += rank_block) {
            __m512 rank_acc[8];
            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                rank_acc[i] = _mm512_setzero_ps();
            }

            int h = 0;
            for (; h + cache_block_h <= H; h += cache_block_h) {
                for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                    float x_val = (float)x_ptr[r + i];
                    if (x_val == 0.0f) continue;

                    const bf16* B_ptr = B_mat + (r + i) * H + h;

                    int j = 0;
                    for (; j + 15 < cache_block_h; j += 16) {
                        auto v_B = _mm256_loadu_si256((const __m256i*)(B_ptr + j));
                        rank_acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(x_val * scaling),
                                                    _mm512_cvtph_ps(v_B), rank_acc[i]);
                    }

                    for (; j < cache_block_h; ++j) {
                        float b_val = (float)B_ptr[j];
                        rank_acc[i][0] += x_val * scaling * b_val;
                    }
                }

                for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                    float acc_arr[16];
                    _mm256_storeu_ps(acc_arr, _mm512_castps512_ps256(rank_acc[i]));
                    for (int j = 0; j < 16; ++j) {
                        if (h + j < H) {
                            out[n * H + (h + j)] += bf16(acc_arr[j]);
                        }
                    }
                }
            }

            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                float x_val = (float)x_ptr[r + i];
                if (x_val == 0.0f) continue;

                const bf16* B_ptr = B_mat + (r + i) * H + h;
                for (; h < H; ++h) {
                    float b_val = (float)B_ptr[h];
                    out[n * H + h] += bf16(x_val * scaling * b_val);
                }
            }
        }
    }
}

// Specialized kernel for Down phase of MoE
void moe_lora_down_kernel(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int R, int H, float scaling) {

    const int cache_block_h = 512;
    const int rank_block = 16;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int n = 0; n < N; ++n) {
        const bf16* x_ptr = x + n * R;

        for (int r = 0; r < R; r += rank_block) {
            __m512 rank_acc[16];
            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                rank_acc[i] = _mm512_setzero_ps();
            }

            int h = 0;
            for (; h + cache_block_h <= H; h += cache_block_h) {
                for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                    float x_val = (float)x_ptr[r + i];
                    if (x_val == 0.0f) continue;

                    const bf16* B_ptr = B_mat + (r + i) * H + h;

                    int j = 0;
                    for (; j + 15 < cache_block_h; j += 16) {
                        auto v_B = _mm256_loadu_si256((const __m256i*)(B_ptr + j));
                        rank_acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(x_val * scaling),
                                                    _mm512_cvtph_ps(v_B), rank_acc[i]);
                    }

                    for (; j < cache_block_h; ++j) {
                        float b_val = (float)B_ptr[j];
                        rank_acc[i][0] += x_val * scaling * b_val;
                    }
                }

                for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                    float acc_arr[16];
                    _mm256_storeu_ps(acc_arr, _mm512_castps512_ps256(rank_acc[i]));
                    for (int j = 0; j < 16; ++j) {
                        if (h + j < H) {
                            out[n * H + (h + j)] += bf16(acc_arr[j]);
                        }
                    }
                }
            }

            for (int i = 0; i < rank_block && (r + i) < R; ++i) {
                float x_val = (float)x_ptr[r + i];
                if (x_val == 0.0f) continue;

                const bf16* B_ptr = B_mat + (r + i) * H + h;
                for (; h < H; ++h) {
                    float b_val = (float)B_ptr[h];
                    out[n * H + h] += bf16(x_val * scaling * b_val);
                }
            }
        }
    }
}

// Main dispatcher function
void moe_lora_dispatch(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, float scaling,
    const std::string& phase = "full") {

    if (phase == "gate") {
        moe_lora_gate_kernel(x, A_mat, out, N, H, R, scaling);
    } else if (phase == "up") {
        moe_lora_up_kernel(x, B_mat, out, N, H, R, scaling);
    } else if (phase == "down") {
        moe_lora_down_kernel(x, B_mat, out, N, H, R, scaling);
    } else {
        int kernel_type = get_optimal_kernel_type(N);

        switch (kernel_type) {
            case 0:
                moe_lora_tiny_kernel(x, A_mat, B_mat, out, N, H, R, scaling);
                break;
            case 1:
                moe_lora_small_kernel(x, A_mat, B_mat, out, N, H, R, scaling);
                break;
            case 2:
                moe_lora_medium_kernel(x, A_mat, B_mat, out, N, H, R, scaling);
                break;
            case 3:
                moe_lora_large_kernel(x, A_mat, B_mat, out, N, H, R, scaling);
                break;
            default:
                moe_lora_medium_kernel(x, A_mat, B_mat, out, N, H, R, scaling);
        }
    }
}

// C++ interface for Python
void moe_batch_lora_avx(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* out, int N, int H, int R, float scaling) {
    moe_lora_dispatch(x, A_mat, B_mat, out, N, H, R, scaling);
}

void moe_batch_lora_gate_avx(
    const bf16* x, const bf16* A_mat, bf16* out,
    int N, int H, int R, float scaling) {
    moe_lora_dispatch(x, A_mat, nullptr, out, N, H, R, scaling, "gate");
}

void moe_batch_lora_up_avx(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int H, int R, float scaling) {
    moe_lora_dispatch(x, nullptr, B_mat, out, N, H, R, scaling, "up");
}

void moe_batch_lora_down_avx(
    const bf16* x, const bf16* B_mat, bf16* out,
    int N, int H, int R, float scaling) {
    moe_lora_dispatch(x, nullptr, B_mat, out, N, H, R, scaling, "down");
}

// Python bindings
PYBIND11_MODULE(moe_lora_cpu_kernel, m) {
    m.def("moe_batch_lora_avx", [](
        const at::Tensor& x, const at::Tensor& A, const at::Tensor& B,
        at::Tensor& out, int N, int H, int R, float scaling) {
        moe_batch_lora_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(A.data_ptr()),
            reinterpret_cast<const bf16*>(B.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, H, R, scaling
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
        int N, int H, int R, float scaling) {
        moe_batch_lora_up_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(B.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, H, R, scaling
        );
    }, "MoE Up phase AVX-512 LoRA kernel");

    m.def("moe_batch_lora_down_avx", [](
        const at::Tensor& x, const at::Tensor& B, at::Tensor& out,
        int N, int H, int R, float scaling) {
        moe_batch_lora_down_avx(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(B.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            N, H, R, scaling
        );
    }, "MoE Down phase AVX-512 LoRA kernel");

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
