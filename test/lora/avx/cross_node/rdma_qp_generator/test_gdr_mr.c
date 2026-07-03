/* test/lora/avx/cross_node/rdma_qp_generator/test_gdr_mr.c */
/* Smoke test: gdrqp_create succeeds, gdrqp_gpu_ptr returns a valid GPU
 * pointer that can be written to via cudaMemset. */
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "gdrqpgen.h"

int main(void) {
    char errbuf[256] = {0};
    gdrqp_ctx *ctx = gdrqp_create("mlx5_0", 1, 16, 1 << 20, errbuf);
    if (!ctx) {
        fprintf(stderr, "gdrqp_create failed: %s\n", errbuf);
        return 1;
    }
    void *gpu = gdrqp_gpu_ptr(ctx);
    if (!gpu) {
        fprintf(stderr, "gdrqp_gpu_ptr returned NULL\n");
        gdrqp_destroy(ctx);
        return 1;
    }
    cudaError_t err = cudaMemset(gpu, 0xab, 1 << 20);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        gdrqp_destroy(ctx);
        return 1;
    }
    gdrqp_destroy(ctx);
    printf("OK\n");
    return 0;
}
