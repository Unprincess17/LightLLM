// AVX-512 BF16 LoRA Kernel for Sapphire Rapids
// Uses native _mm512_dpbf16_ps instruction
// Highly optimized for small batches and single token inference in MoE architectures

#include <torch/extension.h>
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>
#include <algorithm>

using bf16 = c10::BFloat16;

// Helper to estimate optimal number of threads based on input size
inline int get_optimal_threads(int B, int H, int R) {
    const int max_threads = std::thread::hardware_concurrency();
    int task_size = B * R;

    if (B == 1) {
        return std::min(max_threads / 2, R); // For single token, use fewer threads but larger task chunks
    } else if (B <= 4) {
        return std::min(max_threads, B * R);
    } else {
        return max_threads;
    }
}

// Down Projection: x [B, H] @ A.T [R, H] -> Out [B, R]
// Optimized for single token and small batches with prefetch and cache blocking
void lora_down_avx512_bf16(
    const bf16* x, const bf16* A_mat, bf16* out,
    int B, int H, int R) {

    const int cache_block_h = 256; // Smaller cache blocks for better hit rate

    const int num_threads = get_optimal_threads(B, H, R);
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(dynamic, 2)
        for (int br = 0; br < B * R; ++br) {
            int b = br / R;
            int r = br % R;

            const bf16* x_ptr = x + b * H;
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();

            int k = 0;
            for (; k + cache_block_h <= H; k += cache_block_h) {
                const bf16* x_block = x_ptr + k;
                const bf16* A_block = A_ptr + k;

                // Prefetch next block
                if (k + 2 * cache_block_h <= H) {
                    _mm_prefetch((const char*)(x_ptr + k + 2 * cache_block_h), _MM_HINT_T1);
                    _mm_prefetch((const char*)(A_ptr + k + 2 * cache_block_h), _MM_HINT_T1);
                }

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
            out[b * R + r] = bf16(res);
        }
    }
}

// Up Projection: x [B, R] @ B [R, H] -> Out [B, H]
// Optimized for small batches with register blocking and ILP optimization
void lora_up_avx512_bf16(
    const bf16* x, const bf16* B_mat, bf16* out,
    int B, int R, int H) {

    const int reg_block_h = 16; // AVX-512 vector width for BF16
    const int rank_block = 4;   // Smaller rank blocks for better cache

    const int num_threads = get_optimal_threads(B, H, R);
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < B; ++b) {
            const bf16* x_ptr = x + b * R;
            bf16* out_ptr = out + b * H;

            int h = 0;
            for (; h + reg_block_h <= H; h += reg_block_h) {
                __m512 acc = _mm512_setzero_ps();

                for (int r = 0; r < R; r += rank_block) {
                    int r_end = std::min(r + rank_block, R);

                    for (int r_i = r; r_i < r_end; ++r_i) {
                        float x_val = (float)x_ptr[r_i];
                        if (x_val == 0.0f) continue;

                        const bf16* B_row = B_mat + r_i * H;
                        __m256i v_B_u16 = _mm256_loadu_si256((const __m256i*)(B_row + h));
                        __m512i v_B_u32 = _mm512_cvtepu16_epi32(v_B_u16);
                        __m512 v_Bf32 = _mm512_castsi512_ps(_mm512_slli_epi32(v_B_u32, 16));
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(x_val), v_Bf32, acc);
                    }
                }

                float acc_arr[16];
                _mm512_storeu_ps(acc_arr, acc);
                for (int i = 0; i < reg_block_h; ++i) {
                    out_ptr[h + i] = bf16(acc_arr[i]);
                }
            }

            for (; h < H; ++h) {
                float sum = 0.0f;
                for (int r = 0; r < R; ++r) {
                    sum += (float)x_ptr[r] * (float)B_mat[r * H + h];
                }
                out_ptr[h] = bf16(sum);
            }
        }
    }
}

// Combined LoRA computation
void batch_lora_avx512_bf16(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* temp, bf16* out,
    int B, int H_in, int R, int H_out, float scaling) {

    lora_down_avx512_bf16(x, A_mat, temp, B, H_in, R);
    lora_up_avx512_bf16(temp, B_mat, out, B, R, H_out);

    // Apply scaling
    if (scaling != 1.0f) {
        for (int i = 0; i < B * H_out; ++i) {
            out[i] = bf16((float)out[i] * scaling);
        }
    }
}

// PyTorch bindings
void lora_down_bindings(torch::Tensor x, torch::Tensor A, torch::Tensor out) {
    lora_down_avx512_bf16(
        (const bf16*)x.data_ptr(),
        (const bf16*)A.data_ptr(),
        (bf16*)out.data_ptr(),
        x.size(0), x.size(1), A.size(0));
}

void lora_up_bindings(torch::Tensor x, torch::Tensor B_tensor, torch::Tensor out) {
    lora_up_avx512_bf16(
        (const bf16*)x.data_ptr(),
        (const bf16*)B_tensor.data_ptr(),
        (bf16*)out.data_ptr(),
        x.size(0), x.size(1), B_tensor.size(1));
}

void batch_lora_bindings(torch::Tensor x, torch::Tensor A, torch::Tensor B_tensor,
                        torch::Tensor out, float scaling) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, hidden_in]");
    TORCH_CHECK(A.dim() == 2, "A must be 2D [rank, hidden_in]");
    TORCH_CHECK(B_tensor.dim() == 2, "B must be 2D [rank, hidden_out]");
    TORCH_CHECK(out.dim() == 2, "out must be 2D [batch, hidden_out]");

    const int batch = x.size(0);
    const int hidden_in = x.size(1);
    const int rank = A.size(0);
    const int hidden_out = B_tensor.size(1);

    TORCH_CHECK(A.size(1) == hidden_in, "A.shape[1] must match x.shape[1]");
    TORCH_CHECK(B_tensor.size(0) == rank, "B.shape[0] must match A.shape[0]");
    TORCH_CHECK(out.size(0) == batch, "out.shape[0] must match x.shape[0]");
    TORCH_CHECK(out.size(1) == hidden_out, "out.shape[1] must match B.shape[1]");

    // Reuse temporary buffer per-thread to avoid repeated heap allocations.
    thread_local std::vector<bf16> temp_buffer;
    size_t temp_size = static_cast<size_t>(batch) * static_cast<size_t>(rank);
    if (temp_buffer.size() < temp_size) {
        temp_buffer.resize(temp_size);
    }
    batch_lora_avx512_bf16(
        (const bf16*)x.data_ptr(),
        (const bf16*)A.data_ptr(),
        (const bf16*)B_tensor.data_ptr(),
        temp_buffer.data(),
        (bf16*)out.data_ptr(),
        batch, hidden_in, rank, hidden_out, scaling);
}
