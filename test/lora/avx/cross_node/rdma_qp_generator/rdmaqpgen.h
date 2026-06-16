// test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.h
#ifndef RDMAQPGEN_H
#define RDMAQPGEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rdmaqp_ctx rdmaqp_ctx;

/** Info exchanged during TCP handshake before QP connect. */
typedef struct {
    int      qpn_base;    /* first local QP number */
    int      lid;         /* local LID from ibv_query_port */
    char     gid[33];     /* GID string (hex:xxxx:...), nul-terminated */
    uint64_t mr_addr;     /* virtual address of registered MR */
    uint32_t mr_rkey;     /* rkey of registered MR */
} rdmaqp_peer_info;

/**
 * Create N RC QPs in INIT state, allocate and register one big MR,
 * create CQ + PD.  Returns NULL on error (message in errbuf).
 *
 * @param mlx_device  e.g. "mlx5_0"
 * @param ib_port     port number (1)
 * @param num_qps     number of RC QP pairs
 * @param qp_depth    send-queue WR slots per QP
 * @param msg_bytes   payload bytes per RDMA WRITE
 * @param errbuf      caller-allocated 256-byte buffer
 */
rdmaqp_ctx* rdmaqp_create(
    const char *mlx_device,
    int         ib_port,
    int         num_qps,
    int         qp_depth,
    int         msg_bytes,
    char        errbuf[256]);

/** Fill local peer info for TCP handshake. Returns 0 on success. */
int rdmaqp_get_local_info(rdmaqp_ctx *ctx, rdmaqp_peer_info *info);

/**
 * Transition all QPs INIT -> RTR -> RTS using the remote peer info.
 * Returns 0 on success, -1 on error (message in errbuf).
 */
int rdmaqp_connect(rdmaqp_ctx *ctx,
                   const rdmaqp_peer_info *remote,
                   char errbuf[256]);

/**
 * Start synchronous burst loop on a dedicated pthread.
 * Each burst: post all WRs to all QPs, poll CQ until all complete,
 * then sleep gap_us.  Loops until rdmaqp_stop() is called.
 *
 * @param target_gbps  0 = full speed; >0 enables coarse rate limiting
 */
int rdmaqp_start_burst_loop(rdmaqp_ctx *ctx,
                            long burst_us,
                            long gap_us,
                            int  target_gbps);

/** Signal burst loop to stop and join the thread. */
int rdmaqp_stop(rdmaqp_ctx *ctx);

/** Cumulative bytes successfully written (from CQ byte_len). */
uint64_t rdmaqp_bytes_sent(rdmaqp_ctx *ctx);

/** Tear down QPs, CQ, MR, PD, context. */
void rdmaqp_destroy(rdmaqp_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif /* RDMAQPGEN_H */
