/* test/lora/avx/cross_node/rdma_qp_generator/test_gdr_loopback.c */
/* Single-process: two ctx, connect each to the other's peer info. */
#include <stdio.h>
#include <string.h>
#include "gdrqpgen.h"

int main(void) {
    char errbuf[256] = {0};
    gdrqp_ctx *a = gdrqp_create("mlx5_0", 1, 16, 1 << 20, errbuf);
    if (!a) { fprintf(stderr, "create A: %s\n", errbuf); return 1; }
    gdrqp_ctx *b = gdrqp_create("mlx5_0", 1, 16, 1 << 20, errbuf);
    if (!b) { fprintf(stderr, "create B: %s\n", errbuf);
              gdrqp_destroy(a); return 1; }

    gdrqp_peer_info ainfo, binfo;
    if (gdrqp_get_local_info(a, &ainfo) != 0) return 2;
    if (gdrqp_get_local_info(b, &binfo) != 0) return 2;

    if (gdrqp_connect(a, &binfo, errbuf) != 0) {
        fprintf(stderr, "connect A: %s\n", errbuf);
        gdrqp_destroy(a); gdrqp_destroy(b); return 3;
    }
    if (gdrqp_connect(b, &ainfo, errbuf) != 0) {
        fprintf(stderr, "connect B: %s\n", errbuf);
        gdrqp_destroy(a); gdrqp_destroy(b); return 3;
    }

    gdrqp_destroy(a);
    gdrqp_destroy(b);
    printf("OK\n");
    return 0;
}
