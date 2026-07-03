/* test/lora/avx/cross_node/rdma_qp_generator/gdrqpgen.h */
#ifndef GDRQPGEN_H
#define GDRQPGEN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gdrqp_ctx gdrqp_ctx;

/** Info exchanged during TCP handshake before QP connect. */
typedef struct {
    int      qpn;          /* local QP number (single QP per context) */
    int      lid;          /* local LID from ibv_query_port */
    char     gid[40];      /* GID string */
    uint64_t mr_addr;      /* virtual address of registered GPU MR */
    uint32_t mr_rkey;      /* rkey of registered GPU MR */
    uint64_t mr_size;      /* size of registered MR in bytes */
} gdrqp_peer_info;

/**
 * Create context: PD, CQ, one RC QP in INIT.
 * Allocates CUDA buffer of `gpu_buffer_bytes` and registers it as IB MR
 * via nvidia_peermem.  Returns NULL on error (message in errbuf).
 */
gdrqp_ctx* gdrqp_create(
    const char *mlx_device,
    int         ib_port,
    int         qp_depth,
    size_t      gpu_buffer_bytes,
    char        errbuf[256]);

/** Fill local peer info for handshake. Returns 0 on success. */
int gdrqp_get_local_info(gdrqp_ctx *ctx, gdrqp_peer_info *info);

/** Transition QP INIT -> RTR -> RTS. Returns 0 on success. */
int gdrqp_connect(gdrqp_ctx *ctx,
                  const gdrqp_peer_info *remote,
                  char errbuf[256]);

/**
 * Issue blocking RDMA WRITE from local GPU MR to remote GPU MR.
 * Bytes are written at `local_offset` of the local MR, targeting
 * `remote_offset` of the remote MR.  Returns 0 on success.
 */
int gdrqp_write(gdrqp_ctx *ctx,
                size_t local_offset,
                size_t remote_offset,
                size_t nbytes,
                char errbuf[256]);

/** Issue blocking RDMA READ from remote GPU MR to local GPU MR. */
int gdrqp_read(gdrqp_ctx *ctx,
               size_t local_offset,
               size_t remote_offset,
               size_t nbytes,
               char errbuf[256]);

/** Return raw GPU pointer (cudaMalloc'd) for caller's use. */
void* gdrqp_gpu_ptr(gdrqp_ctx *ctx);

/** Return size of the GPU MR. */
size_t gdrqp_gpu_size(gdrqp_ctx *ctx);

/** Tear down QP, MR, CQ, PD, free CUDA buffer. */
void gdrqp_destroy(gdrqp_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif /* GDRQPGEN_H */
