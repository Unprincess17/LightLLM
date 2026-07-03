/* test/lora/avx/cross_node/rdma_qp_generator/test_gdr_read.c */
#include <stdio.h>
#include <cuda_runtime.h>
#include "gdrqpgen.h"

#define BUFSZ (1 << 16)

int main(void) {
    char errbuf[256] = {0};
    gdrqp_ctx *a = gdrqp_create("mlx5_0", 1, 16, BUFSZ, errbuf);
    gdrqp_ctx *b = gdrqp_create("mlx5_0", 1, 16, BUFSZ, errbuf);
    gdrqp_peer_info ainfo, binfo;
    gdrqp_get_local_info(a, &ainfo);
    gdrqp_get_local_info(b, &binfo);
    gdrqp_connect(a, &binfo, errbuf);
    gdrqp_connect(b, &ainfo, errbuf);

    /* Fill B's GPU buffer; A will READ from B. */
    char host[BUFSZ];
    for (int i = 0; i < BUFSZ; i++) host[i] = (char)((i * 7) & 0xff);
    cudaMemcpy(gdrqp_gpu_ptr(b), host, BUFSZ, cudaMemcpyHostToDevice);

    cudaMemset(gdrqp_gpu_ptr(a), 0, BUFSZ);

    if (gdrqp_read(a, 0, 0, BUFSZ, errbuf) != 0) {
        fprintf(stderr, "read: %s\n", errbuf); return 4;
    }

    char verify[BUFSZ];
    cudaMemcpy(verify, gdrqp_gpu_ptr(a), BUFSZ, cudaMemcpyDeviceToHost);
    for (int i = 0; i < BUFSZ; i++) {
        if (verify[i] != (char)((i * 7) & 0xff)) {
            fprintf(stderr, "byte %d mismatch\n", i);
            return 5;
        }
    }

    gdrqp_destroy(a);
    gdrqp_destroy(b);
    printf("OK\n");
    return 0;
}
