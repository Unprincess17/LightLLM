/* test/lora/avx/cross_node/rdma_qp_generator/test_gdrqpgen_header.c */
/* Smoke test: header parses standalone and declares expected symbols. */
#include "gdrqpgen.h"

int main(void) {
    /* Compile-time checks for symbol presence */
    gdrqp_ctx *ctx = 0;
    gdrqp_peer_info info;
    (void)ctx;
    (void)info;
    return 0;
}
