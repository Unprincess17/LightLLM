// AVX-512 BF16 LoRA Kernel for Sapphire Rapids (Intel Xeon Gold 5520+)
// Uses _mm512_dpbf16_ps for 2x throughput: BF16*BF16 + F32 -> F32 in one cycle

#include <torch/extension.h>
#include <immintrin.h>
#include <cstddef>
#include <vector>

// Use torch::BFloat16 type
using bf16 = c10::BFloat16;

// Down Projection: x [B, H] @ A.T [R, H] -> Out [B, R]
// A is stored as [R, H] Row-major (no transpose needed)
// x is stored as [B, H] Row-major
void lora_down_avx512_bf16(
    const bf16* x,      // [B, H] - contiguous
    const bf16* A_mat,  // [R, H] - contiguous (Row-major)
    bf16* out,          // [B, R] - output
    int B, int H, int R) {

    // Process each batch element
    for (int b = 0; b < B; ++b) {
        const bf16* x_ptr = x + b * H;

        // Process each rank
        for (int r = 0; r < R; ++r) {
            const bf16* A_ptr = A_mat + r * H;
            __m512 acc = _mm512_setzero_ps();  // F32 accumulator

            // Vectorized dot product: 32 BF16s (512 bits) per iteration
            int k = 0;
            for (; k + 31 < H; k += 32) {
                auto v_x = _mm512_loadu_si512((void*)(x_ptr + k));
                auto v_A = _mm512_loadu_si512((void*)(A_ptr + k));
                // DPBF16: Dot product BF16 pairs, accumulate to F32
                acc = _mm512_dpbf16_ps(acc, (__m512bh)v_x, (__m512bh)v_A);
            }

            // Horizontal reduce F32 accumulator to single scalar
            float res_f32 = _mm512_reduce_add_ps(acc);

            // Store as BF16
            out[b * R + r] = bf16(res_f32);
        }
    }
}

// Up Projection: x [B, R] @ B [R, H] -> Out [B, H]
void lora_up_avx512_bf16_opt(
    const bf16* x,       // [B, R] - contiguous
    const bf16* B_mat,   // [R, H] - contiguous (Row-major)
    bf16* out,          // [B, H] - output
    int B, int R, int H) {

    // Process each batch element
    for (int b = 0; b < B; ++b) {
        const bf16* x_ptr = x + b * R;

        // Process H in chunks of 16 (32 bytes = 16 BF16s)
        int h = 0;
        for (; h + 15 < H; h += 16) {
            // Initialize accumulators for each H position
            __m512 accs[16];
            for (int i = 0; i < 16; ++i) {
                accs[i] = _mm512_setzero_ps();
            }

            // Accumulate over ranks
            for (int r = 0; r < R; ++r) {
                float x_val = (float)x_ptr[r];
                __m512 v_x = _mm512_set1_ps(x_val);

                // Load 16 BF16s from B matrix
                const bf16* B_ptr = B_mat + r * H + h;

                // Process each of the 16 elements
                for (int hi = 0; hi < 16; ++hi) {
                    float b_val = (float)B_ptr[hi];
                    accs[hi] = _mm512_fmadd_ps(v_x, _mm512_set1_ps(b_val), accs[hi]);
                }
            }

            // Store results
            for (int hi = 0; hi < 16; ++hi) {
                float res = _mm512_cvtss_f32(accs[hi]);
                out[b * H + h + hi] = bf16(res);
            }
        }

        // Handle remaining H elements
        for (; h < H; ++h) {
            __m512 acc = _mm512_setzero_ps();
            for (int r = 0; r < R; ++r) {
                float x_val = (float)x_ptr[r];
                float b_val = (float)B_mat[r * H + h];
                acc = _mm512_fmadd_ps(_mm512_set1_ps(x_val), _mm512_set1_ps(b_val), acc);
            }
            out[b * H + h] = bf16(_mm512_cvtss_f32(acc));
        }
    }
}

// Combined LoRA: x @ A.T @ B * scaling
void batch_lora_avx512_bf16(
    const bf16* x,       // [B, H]
    const bf16* A_mat,   // [R, H]
    const bf16* B_mat,   // [R, H]
    bf16* temp,          // [B, R] - intermediate storage
    bf16* out,           // [B, H]
    int B, int H, int R,
    float scaling) {

    // Step 1: Down projection x @ A.T -> temp [B, R]
    lora_down_avx512_bf16(x, A_mat, temp, B, H, R);

    // Step 2: Up projection temp @ B -> out [B, H]
    lora_up_avx512_bf16_opt(temp, B_mat, out, B, R, H);

    // Apply scaling
    for (int i = 0; i < B * H; ++i) {
        out[i] = bf16((float)out[i] * scaling);
    }
}

// PyTorch bindings
void lora_down_bindings(
    torch::Tensor x,        // [B, H]
    torch::Tensor A,        // [R, H]
    torch::Tensor out) {    // [B, R]

    const bf16* x_ptr = (const bf16*)x.data_ptr();
    const bf16* A_ptr = (const bf16*)A.data_ptr();
    bf16* out_ptr = (bf16*)out.data_ptr();

    int B = x.size(0);
    int H = x.size(1);
    int R = A.size(0);

    lora_down_avx512_bf16(x_ptr, A_ptr, out_ptr, B, H, R);
}

void lora_up_bindings(
    torch::Tensor x,        // [B, R]
    torch::Tensor B_tensor, // [R, H]
    torch::Tensor out) {   // [B, H]

    const bf16* x_ptr = (const bf16*)x.data_ptr();
    const bf16* B_ptr = (const bf16*)B_tensor.data_ptr();
    bf16* out_ptr = (bf16*)out.data_ptr();

    int batch = x.size(0);
    int R = x.size(1);
    int H = B_tensor.size(1);

    lora_up_avx512_bf16_opt(x_ptr, B_ptr, out_ptr, batch, R, H);
}

void batch_lora_bindings(
    torch::Tensor x,        // [B, H]
    torch::Tensor A,        // [R, H]
    torch::Tensor B_tensor, // [R, H]
    torch::Tensor out,      // [B, H]
    float scaling) {

    const bf16* x_ptr = (const bf16*)x.data_ptr();
    const bf16* A_ptr = (const bf16*)A.data_ptr();
    const bf16* B_ptr = (const bf16*)B_tensor.data_ptr();
    bf16* out_ptr = (bf16*)out.data_ptr();

    int B = x.size(0);
    int H = x.size(1);
    int R = A.size(0);

    // Allocate temporary buffer for intermediate
    std::vector<bf16> temp(B * R);
    batch_lora_avx512_bf16(x_ptr, A_ptr, B_ptr, temp.data(), out_ptr, B, H, R, scaling);
}
