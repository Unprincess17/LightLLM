// AVX-512 BF16 LoRA Kernel for Sapphire Rapids
// Uses native _mm512_dpbf16_ps instruction
// Optimized for cache-friendly memory access

#include <torch/extension.h>
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include <cstring>

using bf16 = c10::BFloat16;

// Down Projection: x [B, H] @ A.T [R, H] -> Out [B, R]
// Inner product pattern: contiguous access for both x and A rows
void lora_down_avx512_bf16(
    const bf16* x, const bf16* A_mat, bf16* out,
    int B, int H, int R) {

    for (int b = 0; b < B; ++b) {
        const bf16* x_ptr = x + b * H;

        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();

            // Process 32 BF16 elements per iteration (512 bits)
            int k = 0;
            for (; k + 31 < H; k += 32) {
                auto v_x = _mm512_loadu_si512((const __m512i*)(x_ptr + k));
                auto v_A = _mm512_loadu_si512((const __m512i*)(A_ptr + k));
                // DPBF16: Dot product BF16 pairs, accumulate to F32
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }

            // Horizontal reduce and store
            float res_f32 = _mm512_reduce_add_ps(acc);
            for (; k < H; ++k) {
                res_f32 += (float)x_ptr[k] * (float)A_ptr[k];
            }
            out[b * R + r] = bf16(res_f32);
        }
    }
}

// Up Projection: x [B, R] @ B [R, H] -> Out [B, H]
// CORRECTED: Iterate H on outside, accumulate R in register
// Writes to memory only ONCE per output element
void lora_up_avx512_bf16(
    const bf16* x, const bf16* B_mat, bf16* out,
    int B, int R, int H) {

    for (int b = 0; b < B; ++b) {
        const bf16* x_ptr = x + b * R;
        bf16* out_ptr = out + b * H;

        // Process in blocks of 16 (one AVX-512 register = 16 floats = 32 BF16s)
        int h = 0;
        for (; h + 15 < H; h += 16) {
            __m512 acc = _mm512_setzero_ps();

            // Accumulate over rank dimension
            for (int r = 0; r < R; ++r) {
                float x_val = (float)x_ptr[r];
                if (x_val == 0.0f) continue;

                const bf16* B_row = B_mat + r * H;
                // Load 16 BF16s as F32 via vcvtneps_pbh (if available) or manual
                // For Sapphire Rapids, use native BF16 support
                __m512i v_B = _mm512_loadu_si512((const __m512i*)(B_row + h));
                // Convert BF16 to F32 using DPBF16-style conversion
                // Shift upper 16 bits (BF16) to lower 16 bits, then zero-extend to F32
                // Actually, BF16 is in upper bits for DPBF16, let's use vmovnebhd
                __m512 v_Bf32 = _mm512_castsi512_ps(_mm512_srli_epi32(v_B, 16));
                acc = _mm512_fmadd_ps(_mm512_set1_ps(x_val), v_Bf32, acc);
            }

            // Store result
            float acc_arr[16];
            _mm512_storeu_ps(acc_arr, acc);
            for (int i = 0; i < 16; ++i) {
                out_ptr[h + i] = bf16(acc_arr[i]);
            }
        }

        // Handle remaining elements
        for (; h < H; ++h) {
            float sum = 0.0f;
            for (int r = 0; r < R; ++r) {
                sum += (float)x_ptr[r] * (float)B_mat[r * H + h];
            }
            out_ptr[h] = bf16(sum);
        }
    }
}

// Combined LoRA computation
void batch_lora_avx512_bf16(
    const bf16* x, const bf16* A_mat, const bf16* B_mat,
    bf16* temp, bf16* out,
    int B, int H, int R, float scaling) {

    lora_down_avx512_bf16(x, A_mat, temp, B, H, R);
    lora_up_avx512_bf16(temp, B_mat, out, B, R, H);

    // Apply scaling
    if (scaling != 1.0f) {
        for (int i = 0; i < B * H; ++i) {
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
    int B = x.size(0), H = x.size(1), R = A.size(0);
    std::vector<bf16> temp(B * R);
    batch_lora_avx512_bf16(
        (const bf16*)x.data_ptr(),
        (const bf16*)A.data_ptr(),
        (const bf16*)B_tensor.data_ptr(),
        temp.data(),
        (bf16*)out.data_ptr(),
        B, H, R, scaling);
}
