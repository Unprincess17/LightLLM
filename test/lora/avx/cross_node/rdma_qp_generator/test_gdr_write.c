/* test/lora/avx/cross_node/rdma_qp_generator/test_gdr_write.c */
/* Loopback: write A's GPU MR contents into B's GPU MR, verify B sees it. */
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "gdrqpgen.h"

#define BUFSZ (1 << 16) /* 64 KB */

int main(void) {
    char errbuf[256] = {0};
    gdrqp_ctx *a = gdrqp_create("mlx5_0", 1, 16, BUFSZ, errbuf);
    if (!a) { fprintf(stderr, "create A: %s\n", errbuf); return 1; }
    gdrqp_ctx *b = gdrqp_create("mlx5_0", 1, 16, BUFSZ, errbuf);
    if (!b) { gdrqp_destroy(a); return 1; }

    gdrqp_peer_info ainfo, binfo;
    gdrqp_get_local_info(a, &ainfo);
    gdrqp_get_local_info(b, &binfo);
    if (gdrqp_connect(a, &binfo, errbuf) != 0) {
        fprintf(stderr, "connect A: %s\n", errbuf); return 3;
    }
    if (gdrqp_connect(b, &ainfo, errbuf) != 0) {
        fprintf(stderr, "connect B: %s\n", errbuf); return 3;
    }

    /* Fill A's GPU buffer with a pattern. */
    char host[BUFSZ];
    for (int i = 0; i < BUFSZ; i++) host[i] = (char)(i & 0xff);
    cudaMemcpy(gdrqp_gpu_ptr(a), host, BUFSZ, cudaMemcpyHostToDevice);

    /* Zero B's GPU buffer. */
    cudaMemset(gdrqp_gpu_ptr(b), 0, BUFSZ);

    /* WRITE from A's MR into B's MR. */
    if (gdrqp_write(a, 0, 0, BUFSZ, errbuf) != 0) {
        fprintf(stderr, "write: %s\n", errbuf); return 4;
    }

    /* Verify B's GPU buffer matches the pattern. */
    char verify[BUFSZ];
    cudaMemcpy(verify, gdrqp_gpu_ptr(b), BUFSZ, cudaMemcpyDeviceToHost);
    for (int i = 0; i < BUFSZ; i++) {
        if (verify[i] != (char)(i & 0xff)) {
            fprintf(stderr, "byte %d mismatch: got %02x expected %02x\n",
                    i, (unsigned char)verify[i], i & 0xff);
            return 5;
        }
    }

    gdrqp_destroy(a);
    gdrqp_destroy(b);
    printf("OK\n");
    return 0;
}
