/* test/lora/avx/cross_node/rdma_qp_generator/gdrqpgen.c */
#include "gdrqpgen.h"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct gdrqp_ctx {
    struct ibv_context *verbs;
    struct ibv_pd      *pd;
    struct ibv_cq      *cq;
    struct ibv_qp      *qp;
    struct ibv_mr      *mr;
    void               *gpu_buf;
    size_t              buf_size;
    int                 ib_port;
    uint16_t            lid;
    union ibv_gid       gid;
    int                 qp_depth;
    uint64_t            remote_addr;
    uint32_t            remote_rkey;
    uint64_t            remote_size;
};

static void gid_to_str(const union ibv_gid *gid, char *out, size_t outlen) {
    /* Format as colon-separated 16-bit hex. */
    snprintf(out, outlen,
             "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x",
             (unsigned)((gid->raw[0]<<8)|gid->raw[1]),
             (unsigned)((gid->raw[2]<<8)|gid->raw[3]),
             (unsigned)((gid->raw[4]<<8)|gid->raw[5]),
             (unsigned)((gid->raw[6]<<8)|gid->raw[7]),
             (unsigned)((gid->raw[8]<<8)|gid->raw[9]),
             (unsigned)((gid->raw[10]<<8)|gid->raw[11]),
             (unsigned)((gid->raw[12]<<8)|gid->raw[13]),
             (unsigned)((gid->raw[14]<<8)|gid->raw[15]));
}

static int str_to_gid(const char *s, union ibv_gid *out) {
    unsigned v[8];
    if (sscanf(s, "%x:%x:%x:%x:%x:%x:%x:%x",
               &v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6],&v[7]) != 8) {
        return -1;
    }
    for (int i = 0; i < 8; i++) {
        out->raw[2*i]     = (uint8_t)(v[i] >> 8);
        out->raw[2*i + 1] = (uint8_t)(v[i] & 0xff);
    }
    return 0;
}

gdrqp_ctx* gdrqp_create(const char *mlx_device, int ib_port,
                        int qp_depth, size_t gpu_buffer_bytes,
                        char errbuf[256]) {
    gdrqp_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        snprintf(errbuf, 256, "calloc failed");
        return NULL;
    }
    ctx->ib_port = ib_port;
    ctx->qp_depth = qp_depth;
    ctx->buf_size = gpu_buffer_bytes;

    /* Find IB device by name. */
    int ndev = 0;
    struct ibv_device **devs = ibv_get_device_list(&ndev);
    if (!devs) {
        snprintf(errbuf, 256, "ibv_get_device_list failed: %s",
                 strerror(errno));
        free(ctx);
        return NULL;
    }
    struct ibv_device *target = NULL;
    for (int i = 0; i < ndev; i++) {
        if (strcmp(ibv_get_device_name(devs[i]), mlx_device) == 0) {
            target = devs[i];
            break;
        }
    }
    if (!target) {
        snprintf(errbuf, 256, "device %s not found", mlx_device);
        ibv_free_device_list(devs);
        free(ctx);
        return NULL;
    }
    ctx->verbs = ibv_open_device(target);
    ibv_free_device_list(devs);
    if (!ctx->verbs) {
        snprintf(errbuf, 256, "ibv_open_device failed: %s", strerror(errno));
        free(ctx);
        return NULL;
    }

    /* Query port for LID and GID. */
    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx->verbs, ib_port, &port_attr) != 0) {
        snprintf(errbuf, 256, "ibv_query_port failed: %s", strerror(errno));
        goto err;
    }
    ctx->lid = port_attr.lid;
    if (ibv_query_gid(ctx->verbs, ib_port, 1, &ctx->gid) != 0) {
        snprintf(errbuf, 256, "ibv_query_gid failed: %s", strerror(errno));
        goto err;
    }

    /* PD. */
    ctx->pd = ibv_alloc_pd(ctx->verbs);
    if (!ctx->pd) {
        snprintf(errbuf, 256, "ibv_alloc_pd failed: %s", strerror(errno));
        goto err;
    }

    /* CQ. */
    ctx->cq = ibv_create_cq(ctx->verbs, qp_depth * 2, NULL, NULL, 0);
    if (!ctx->cq) {
        snprintf(errbuf, 256, "ibv_create_cq failed: %s", strerror(errno));
        goto err;
    }

    /* Allocate GPU buffer. */
    cudaError_t cerr = cudaMalloc(&ctx->gpu_buf, gpu_buffer_bytes);
    if (cerr != cudaSuccess) {
        snprintf(errbuf, 256, "cudaMalloc(%zu) failed: %s",
                 gpu_buffer_bytes, cudaGetErrorString(cerr));
        goto err;
    }

    /* Register GPU buffer with IB.  Requires nvidia_peermem. */
    int access = IBV_ACCESS_LOCAL_WRITE
               | IBV_ACCESS_REMOTE_WRITE
               | IBV_ACCESS_REMOTE_READ;
    ctx->mr = ibv_reg_mr(ctx->pd, ctx->gpu_buf, gpu_buffer_bytes, access);
    if (!ctx->mr) {
        snprintf(errbuf, 256,
                 "ibv_reg_mr(GPU) failed: %s (is nvidia_peermem loaded?)",
                 strerror(errno));
        goto err;
    }

    /* Create QP in INIT. */
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.send_cq = ctx->cq;
    qp_attr.recv_cq = ctx->cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.sq_sig_all = 1;
    qp_attr.cap.max_send_wr = qp_depth;
    qp_attr.cap.max_recv_wr = qp_depth;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    ctx->qp = ibv_create_qp(ctx->pd, &qp_attr);
    if (!ctx->qp) {
        snprintf(errbuf, 256, "ibv_create_qp failed: %s", strerror(errno));
        goto err;
    }
    struct ibv_qp_attr init_attr = {0};
    init_attr.qp_state = IBV_QPS_INIT;
    init_attr.port_num = ib_port;
    init_attr.pkey_index = 0;
    init_attr.qp_access_flags = access;
    if (ibv_modify_qp(ctx->qp, &init_attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX
                      | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        snprintf(errbuf, 256, "ibv_modify_qp(INIT) failed: %s",
                 strerror(errno));
        goto err;
    }

    return ctx;

err:
    if (ctx->qp) ibv_destroy_qp(ctx->qp);
    if (ctx->mr) ibv_dereg_mr(ctx->mr);
    if (ctx->gpu_buf) cudaFree(ctx->gpu_buf);
    if (ctx->cq) ibv_destroy_cq(ctx->cq);
    if (ctx->pd) ibv_dealloc_pd(ctx->pd);
    if (ctx->verbs) ibv_close_device(ctx->verbs);
    free(ctx);
    return NULL;
}

int gdrqp_get_local_info(gdrqp_ctx *ctx, gdrqp_peer_info *info) {
    if (!ctx || !info) return -1;
    info->qpn = ctx->qp->qp_num;
    info->lid = ctx->lid;
    gid_to_str(&ctx->gid, info->gid, sizeof(info->gid));
    info->mr_addr = (uint64_t)(uintptr_t)ctx->gpu_buf;
    info->mr_rkey = ctx->mr->rkey;
    info->mr_size = ctx->buf_size;
    return 0;
}

void* gdrqp_gpu_ptr(gdrqp_ctx *ctx) {
    return ctx ? ctx->gpu_buf : NULL;
}

size_t gdrqp_gpu_size(gdrqp_ctx *ctx) {
    return ctx ? ctx->buf_size : 0;
}

void gdrqp_destroy(gdrqp_ctx *ctx) {
    if (!ctx) return;
    if (ctx->qp) ibv_destroy_qp(ctx->qp);
    if (ctx->mr) ibv_dereg_mr(ctx->mr);
    if (ctx->gpu_buf) cudaFree(ctx->gpu_buf);
    if (ctx->cq) ibv_destroy_cq(ctx->cq);
    if (ctx->pd) ibv_dealloc_pd(ctx->pd);
    if (ctx->verbs) ibv_close_device(ctx->verbs);
    free(ctx);
}

/* gdrqp_write/read are implemented in later tasks; keep stubs. */
int gdrqp_connect(gdrqp_ctx *ctx, const gdrqp_peer_info *remote,
                  char errbuf[256]) {
    if (!ctx || !remote) {
        snprintf(errbuf, 256, "null ctx or remote");
        return -1;
    }

    /* INIT -> RTR */
    struct ibv_qp_attr rtr = {0};
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = IBV_MTU_1024;
    rtr.dest_qp_num = remote->qpn;
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 0;
    rtr.ah_attr.dlid = remote->lid;
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ctx->ib_port;

    /* RoCE / GID path: fill GRH if dlid is 0. */
    if (remote->lid == 0) {
        union ibv_gid rgid;
        if (str_to_gid(remote->gid, &rgid) != 0) {
            snprintf(errbuf, 256, "invalid remote GID %s", remote->gid);
            return -1;
        }
        rtr.ah_attr.is_global = 1;
        rtr.ah_attr.grh.dgid = rgid;
        rtr.ah_attr.grh.sgid_index = 1;
        rtr.ah_attr.grh.hop_limit = 1;
        rtr.ah_attr.grh.traffic_class = 0;
    }

    if (ibv_modify_qp(ctx->qp, &rtr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU
                      | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                      | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
        snprintf(errbuf, 256, "modify_qp(RTR) failed: %s", strerror(errno));
        return -1;
    }

    /* RTR -> RTS */
    struct ibv_qp_attr rts = {0};
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;

    if (ibv_modify_qp(ctx->qp, &rts,
                      IBV_QP_STATE | IBV_QP_TIMEOUT
                      | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY
                      | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
        snprintf(errbuf, 256, "modify_qp(RTS) failed: %s", strerror(errno));
        return -1;
    }
    ctx->remote_addr = remote->mr_addr;
    ctx->remote_rkey = remote->mr_rkey;
    ctx->remote_size = remote->mr_size;
    return 0;
}
int gdrqp_write(gdrqp_ctx *ctx, size_t local_offset, size_t remote_offset,
                size_t nbytes, char errbuf[256]) {
    if (!ctx) { snprintf(errbuf, 256, "null ctx"); return -1; }
    if (local_offset + nbytes > ctx->buf_size) {
        snprintf(errbuf, 256, "local offset+size exceeds local MR");
        return -1;
    }
    if (remote_offset + nbytes > ctx->remote_size) {
        snprintf(errbuf, 256, "remote offset+size exceeds remote MR");
        return -1;
    }

    struct ibv_sge sge = {
        .addr   = (uint64_t)(uintptr_t)ctx->gpu_buf + local_offset,
        .length = (uint32_t)nbytes,
        .lkey   = ctx->mr->lkey,
    };
    struct ibv_send_wr wr = {0};
    wr.wr_id = 0xABCD;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = ctx->remote_addr + remote_offset;
    wr.wr.rdma.rkey = ctx->remote_rkey;

    struct ibv_send_wr *bad = NULL;
    int ret = ibv_post_send(ctx->qp, &wr, &bad);
    if (ret != 0) {
        snprintf(errbuf, 256, "ibv_post_send: %s", strerror(ret));
        return -1;
    }

    /* Poll for completion. */
    struct ibv_wc wc;
    while (1) {
        int n = ibv_poll_cq(ctx->cq, 1, &wc);
        if (n < 0) {
            snprintf(errbuf, 256, "ibv_poll_cq failed");
            return -1;
        }
        if (n == 0) continue;
        if (wc.status != IBV_WC_SUCCESS) {
            snprintf(errbuf, 256, "WC status %d (%s) vendor_err=%d",
                     wc.status, ibv_wc_status_str(wc.status), wc.vendor_err);
            return -1;
        }
        break;
    }
    return 0;
}
int gdrqp_read(gdrqp_ctx *ctx, size_t local_offset, size_t remote_offset,
               size_t nbytes, char errbuf[256]) {
    if (!ctx) { snprintf(errbuf, 256, "null ctx"); return -1; }
    if (local_offset + nbytes > ctx->buf_size) {
        snprintf(errbuf, 256, "local offset+size exceeds local MR");
        return -1;
    }
    if (remote_offset + nbytes > ctx->remote_size) {
        snprintf(errbuf, 256, "remote offset+size exceeds remote MR");
        return -1;
    }

    struct ibv_sge sge = {
        .addr   = (uint64_t)(uintptr_t)ctx->gpu_buf + local_offset,
        .length = (uint32_t)nbytes,
        .lkey   = ctx->mr->lkey,
    };
    struct ibv_send_wr wr = {0};
    wr.wr_id = 0xABCE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = ctx->remote_addr + remote_offset;
    wr.wr.rdma.rkey = ctx->remote_rkey;

    struct ibv_send_wr *bad = NULL;
    int ret = ibv_post_send(ctx->qp, &wr, &bad);
    if (ret != 0) {
        snprintf(errbuf, 256, "ibv_post_send(READ): %s", strerror(ret));
        return -1;
    }

    struct ibv_wc wc;
    while (1) {
        int n = ibv_poll_cq(ctx->cq, 1, &wc);
        if (n < 0) {
            snprintf(errbuf, 256, "ibv_poll_cq failed");
            return -1;
        }
        if (n == 0) continue;
        if (wc.status != IBV_WC_SUCCESS) {
            snprintf(errbuf, 256, "READ WC status %d (%s)",
                     wc.status, ibv_wc_status_str(wc.status));
            return -1;
        }
        break;
    }
    return 0;
}
