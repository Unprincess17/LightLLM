// test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c
#define _GNU_SOURCE
#include "rdmaqpgen.h"

#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/verbs.h>

/* ------------------------------------------------------------------ */
/* Internal structures                                                 */
/* ------------------------------------------------------------------ */

struct rdmaqp_ctx {
    struct ibv_context    *ctx;
    struct ibv_pd         *pd;
    struct ibv_mr         *mr;
    struct ibv_cq         *cq;
    struct ibv_qp        **qps;         /* array of num_qps QPs */
    struct ibv_device    **dev_list;

    int    num_qps;
    int    qp_depth;
    int    msg_bytes;
    int    ib_port;
    int    gid_index;
    int    active_mtu;
    size_t mr_size;       /* msg_bytes * num_qps * qp_depth */

    /* local peer info (filled after create) */
    rdmaqp_peer_info local_info;

    /* remote peer info (filled by connect) */
    rdmaqp_peer_info remote_info;

    /* burst loop control */
    pthread_t burst_thread;
    volatile int running;
    volatile int stop_requested;

    /* stats */
    volatile uint64_t bytes_sent;
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static void snprintf_err(char *errbuf, size_t n, const char *fmt, ...) {
    if (!errbuf) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(errbuf, n, fmt, ap);
    va_end(ap);
}

static int gid_to_str(const union ibv_gid *gid, char *out, size_t outlen) {
    const uint8_t *raw = gid->raw;
    return snprintf(out, outlen,
        "%02x%02x:%02x%02x:%02x%02x:%02x%02x:"
        "%02x%02x:%02x%02x:%02x%02x:%02x%02x",
        raw[0],raw[1],raw[2],raw[3],raw[4],raw[5],raw[6],raw[7],
        raw[8],raw[9],raw[10],raw[11],raw[12],raw[13],raw[14],raw[15]);
}

static int str_to_gid(const char *s, union ibv_gid *gid) {
    unsigned raw[16];
    if (sscanf(s,
        "%2x%2x:%2x%2x:%2x%2x:%2x%2x:"
        "%2x%2x:%2x%2x:%2x%2x:%2x%2x",
        &raw[0],&raw[1],&raw[2],&raw[3],&raw[4],&raw[5],&raw[6],&raw[7],
        &raw[8],&raw[9],&raw[10],&raw[11],&raw[12],&raw[13],&raw[14],&raw[15]) != 16)
        return -1;
    for (int i = 0; i < 16; i++) gid->raw[i] = (uint8_t)raw[i];
    return 0;
}

/* ------------------------------------------------------------------ */
/* QP state machine: INIT -> RTR -> RTS                                 */
/* ------------------------------------------------------------------ */

static int modify_qp_to_init(struct ibv_qp *qp, int ib_port) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = ib_port;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT
              | IBV_QP_ACCESS_FLAGS;

    return ibv_modify_qp(qp, &attr, flags);
}

static int modify_qp_to_rtr(struct ibv_qp *qp, int ib_port,
                            int remote_qpn, int remote_lid,
                            const union ibv_gid *remote_gid,
                            int gid_index, int active_mtu) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state        = IBV_QPS_RTR;
    attr.path_mtu        = active_mtu;
    attr.dest_qp_num     = remote_qpn;
    attr.rq_psn          = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer   = 12;

    attr.ah_attr.dlid       = remote_lid;
    attr.ah_attr.sl         = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num   = ib_port;
    attr.ah_attr.is_global  = 1;
    attr.ah_attr.grh.dgid   = *remote_gid;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 1;

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU
              | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
              | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    return ibv_modify_qp(qp, &attr, flags);
}

static int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state      = IBV_QPS_RTS;
    attr.sq_psn        = 0;
    attr.timeout       = 14;
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;
    attr.max_rd_atomic = 1;

    int flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT
              | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY
              | IBV_QP_MAX_QP_RD_ATOMIC;

    return ibv_modify_qp(qp, &attr, flags);
}

/* ------------------------------------------------------------------ */
/* Public API: create / get_info / connect / destroy                   */
/* ------------------------------------------------------------------ */

rdmaqp_ctx* rdmaqp_create(
    const char *mlx_device,
    int         ib_port,
    int         num_qps,
    int         qp_depth,
    int         msg_bytes,
    char        errbuf[256])
{
    int ret;

    rdmaqp_ctx *c = calloc(1, sizeof(*c));
    if (!c) { snprintf_err(errbuf, 256, "calloc ctx failed: %s", strerror(errno)); return NULL; }

    c->num_qps   = num_qps;
    c->qp_depth  = qp_depth;
    c->msg_bytes = msg_bytes;
    c->ib_port   = ib_port;
    c->mr_size   = (size_t)msg_bytes * (size_t)num_qps * (size_t)qp_depth;

    /* --- device discovery --- */
    int ndev;
    c->dev_list = ibv_get_device_list(&ndev);
    if (!c->dev_list || ndev == 0) {
        snprintf_err(errbuf, 256, "no IB devices found");
        goto fail;
    }

    struct ibv_device *dev = NULL;
    for (int i = 0; i < ndev; i++) {
        if (strcmp(ibv_get_device_name(c->dev_list[i]), mlx_device) == 0) {
            dev = c->dev_list[i];
            break;
        }
    }
    if (!dev) {
        snprintf_err(errbuf, 256, "device '%s' not found", mlx_device);
        goto fail;
    }

    /* --- open context --- */
    c->ctx = ibv_open_device(dev);
    if (!c->ctx) {
        snprintf_err(errbuf, 256, "ibv_open_device failed: %s", strerror(errno));
        goto fail;
    }

    /* --- port attributes (LID, active MTU) --- */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(c->ctx, ib_port, &port_attr);
    if (ret) {
        snprintf_err(errbuf, 256, "ibv_query_port failed: %s", strerror(errno));
        goto fail;
    }
    c->local_info.lid = port_attr.lid;
    c->active_mtu = port_attr.active_mtu;

    /* --- GID (auto-detect first valid RoCEv2 or non-zero GID) --- */
    {
        union ibv_gid gid;
        int gid_index = -1;
        for (int gi = 0; gi < 16; gi++) {
            ret = ibv_query_gid(c->ctx, ib_port, gi, &gid);
            if (ret) break;
            /* Prefer RoCEv2, accept any non-zero GID */
            int is_nonzero = 0;
            for (int b = 0; b < 16; b++) { if (gid.raw[b]) { is_nonzero = 1; break; } }
            if (is_nonzero) { gid_index = gi; break; }
        }
        if (gid_index < 0) {
            snprintf_err(errbuf, 256, "no valid GID found on %s port %d", mlx_device, ib_port);
            goto fail;
        }
        ret = ibv_query_gid(c->ctx, ib_port, gid_index, &gid);
        if (ret) {
            snprintf_err(errbuf, 256, "ibv_query_gid failed: %s", strerror(errno));
            goto fail;
        }
        c->gid_index = gid_index;
        gid_to_str(&gid, c->local_info.gid, sizeof(c->local_info.gid));
    }

    /* --- PD --- */
    c->pd = ibv_alloc_pd(c->ctx);
    if (!c->pd) {
        snprintf_err(errbuf, 256, "ibv_alloc_pd failed: %s", strerror(errno));
        goto fail;
    }

    /* --- MR (registered for local+remote write) --- */
    void *mr_buf = malloc(c->mr_size);
    if (!mr_buf) {
        snprintf_err(errbuf, 256, "malloc MR buffer (%zu bytes) failed", c->mr_size);
        goto fail;
    }
    memset(mr_buf, 0, c->mr_size);
    c->mr = ibv_reg_mr(c->pd, mr_buf, c->mr_size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!c->mr) {
        snprintf_err(errbuf, 256, "ibv_reg_mr failed: %s", strerror(errno));
        free(mr_buf);
        goto fail;
    }
    c->local_info.mr_addr = (uint64_t)(uintptr_t)mr_buf;
    c->local_info.mr_rkey = c->mr->rkey;

    /* --- CQ (one CQ for all QPs) --- */
    c->cq = ibv_create_cq(c->ctx, num_qps * qp_depth, NULL, NULL, 0);
    if (!c->cq) {
        snprintf_err(errbuf, 256, "ibv_create_cq failed: %s", strerror(errno));
        goto fail;
    }

    /* --- QPs (INIT state) --- */
    c->qps = calloc(num_qps, sizeof(struct ibv_qp*));
    if (!c->qps) {
        snprintf_err(errbuf, 256, "calloc qps failed");
        goto fail;
    }

    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.send_cq = c->cq;
    qp_attr.recv_cq = c->cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr  = qp_depth;
    qp_attr.cap.max_recv_wr  = 1;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.sq_sig_all       = 0;  /* only signaled WRs generate CQEs */

    for (int i = 0; i < num_qps; i++) {
        c->qps[i] = ibv_create_qp(c->pd, &qp_attr);
        if (!c->qps[i]) {
            snprintf_err(errbuf, 256, "ibv_create_qp[%d] failed: %s", i, strerror(errno));
            goto fail;
        }
    }

    c->local_info.qpn_base = c->qps[0]->qp_num;

    printf("[rdmaqpgen] created %d QPs (base QPN=%d), MR=%zu bytes, device=%s\n",
           num_qps, c->local_info.qpn_base, c->mr_size, mlx_device);
    return c;

fail:
    rdmaqp_destroy(c);
    return NULL;
}

int rdmaqp_get_local_info(rdmaqp_ctx *c, rdmaqp_peer_info *info) {
    if (!c || !info) return -1;
    memcpy(info, &c->local_info, sizeof(*info));
    return 0;
}

int rdmaqp_connect(rdmaqp_ctx *c, const rdmaqp_peer_info *remote,
                   char errbuf[256]) {
    if (!c || !remote) {
        snprintf_err(errbuf, 256, "NULL argument");
        return -1;
    }
    memcpy(&c->remote_info, remote, sizeof(*remote));

    union ibv_gid rgid;
    if (str_to_gid(remote->gid, &rgid) != 0) {
        snprintf_err(errbuf, 256, "failed to parse remote GID: %s", remote->gid);
        return -1;
    }

    /* Transition each QP RESET -> INIT -> RTR -> RTS */
    for (int i = 0; i < c->num_qps; i++) {
        int remote_qpn = remote->qpn_base + i;
        int ret;

        ret = modify_qp_to_init(c->qps[i], c->ib_port);
        if (ret) {
            snprintf_err(errbuf, 256,
                         "modify_qp_to_init[%d] failed: %s",
                         i, strerror(-ret));
            return -1;
        }

        ret = modify_qp_to_rtr(c->qps[i], c->ib_port, remote_qpn,
                               remote->lid, &rgid, c->gid_index,
                               c->active_mtu);
        if (ret) {
            snprintf_err(errbuf, 256,
                         "modify_qp_to_rtr[%d] (remote_qpn=%d) failed: %s",
                         i, remote_qpn, strerror(-ret));
            return -1;
        }

        ret = modify_qp_to_rts(c->qps[i]);
        if (ret) {
            snprintf_err(errbuf, 256,
                         "modify_qp_to_rts[%d] failed: %s",
                         i, strerror(-ret));
            return -1;
        }
    }

    printf("[rdmaqpgen] %d QPs connected to remote QPN base %d\n",
           c->num_qps, remote->qpn_base);
    return 0;
}

void rdmaqp_destroy(rdmaqp_ctx *c) {
    if (!c) return;

    /* stop burst loop first */
    if (c->running) rdmaqp_stop(c);

    if (c->qps) {
        for (int i = 0; i < c->num_qps; i++) {
            if (c->qps[i]) ibv_destroy_qp(c->qps[i]);
        }
        free(c->qps);
    }

    if (c->cq)  ibv_destroy_cq(c->cq);

    if (c->mr) {
        void *buf = (void *)(uintptr_t)c->local_info.mr_addr;
        ibv_dereg_mr(c->mr);
        free(buf);
    }

    if (c->pd)  ibv_dealloc_pd(c->pd);
    if (c->ctx) ibv_close_device(c->ctx);
    if (c->dev_list) ibv_free_device_list(c->dev_list);

    free(c);
}

/* ------------------------------------------------------------------ */
/* Burst loop internals                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    rdmaqp_ctx *ctx;
    long        burst_us;
    long        gap_us;
    int         target_gbps;
} burst_args;

static void* burst_loop_thread(void *arg) {
    burst_args *ba = (burst_args*)arg;
    rdmaqp_ctx *c = ba->ctx;

    const int num_qps   = c->num_qps;
    const int qp_depth  = c->qp_depth;
    const int msg_bytes = c->msg_bytes;
    const uint64_t remote_addr = c->remote_info.mr_addr;
    const uint32_t remote_rkey = c->remote_info.mr_rkey;
    const uint32_t local_lkey  = c->mr->lkey;

    /* Pre-compute per-QP local buffer base offsets */
    uint64_t *local_offsets = calloc(num_qps, sizeof(uint64_t));
    uint64_t *remote_offsets = calloc(num_qps, sizeof(uint64_t));
    if (!local_offsets || !remote_offsets) {
        free(local_offsets);
        free(remote_offsets);
        free(ba);
        c->running = 0;
        return NULL;
    }
    for (int i = 0; i < num_qps; i++) {
        local_offsets[i]  = (uint64_t)(uintptr_t)c->mr->addr
                          + (size_t)i * (size_t)qp_depth * (size_t)msg_bytes;
        remote_offsets[i] = remote_addr
                          + (size_t)i * (size_t)qp_depth * (size_t)msg_bytes;
    }

    /* Per-QP ring indices (which WR slot to post next) */
    int *ring = calloc(num_qps, sizeof(int));
    if (!ring) {
        free(local_offsets);
        free(remote_offsets);
        free(ba);
        c->running = 0;
        return NULL;
    }

    /* Allocate SGE and WR arrays per QP */
    struct ibv_sge  *sges = calloc((size_t)num_qps * (size_t)qp_depth,
                                   sizeof(struct ibv_sge));
    struct ibv_send_wr *wrs = calloc((size_t)num_qps * (size_t)qp_depth,
                                     sizeof(struct ibv_send_wr));
    if (!sges || !wrs) {
        free(sges);
        free(wrs);
        free(ring);
        free(local_offsets);
        free(remote_offsets);
        free(ba);
        c->running = 0;
        return NULL;
    }

    /* Pre-fill SGEs and WRs (addresses are fixed per slot) */
    for (int q = 0; q < num_qps; q++) {
        for (int s = 0; s < qp_depth; s++) {
            int idx = q * qp_depth + s;
            sges[idx].addr   = local_offsets[q] + (size_t)s * msg_bytes;
            sges[idx].length = msg_bytes;
            sges[idx].lkey   = local_lkey;

            wrs[idx].wr_id      = idx;
            wrs[idx].opcode     = IBV_WR_RDMA_WRITE;
            wrs[idx].sg_list    = &sges[idx];
            wrs[idx].num_sge    = 1;
            wrs[idx].send_flags = IBV_SEND_SIGNALED;
            wrs[idx].next       = NULL;
            wrs[idx].wr.rdma.remote_addr = remote_offsets[q]
                                         + (size_t)s * msg_bytes;
            wrs[idx].wr.rdma.rkey        = remote_rkey;
        }
    }

    /* For rate limiting */
    uint64_t window_bytes = 0;
    struct timespec window_start, now;

    clock_gettime(CLOCK_MONOTONIC, &window_start);
    c->running = 1;

    while (!c->stop_requested) {
        /* --- Post all WRs to all QPs --- */
        int total_posted = 0;
        for (int q = 0; q < num_qps && !c->stop_requested; q++) {
            for (int s = 0; s < qp_depth; s++) {
                int idx = q * qp_depth + ring[q];
                struct ibv_send_wr *bad = NULL;
                int ret = ibv_post_send(c->qps[q], &wrs[idx], &bad);
                if (ret != 0) {
                    fprintf(stderr, "[rdmaqpgen] ibv_post_send QP[%d] failed: %s\n",
                            q, strerror(ret));
                    break;
                }
                ring[q] = (ring[q] + 1) % qp_depth;
                total_posted++;
            }
        }

        /* --- Poll CQ until all posted WRs complete --- */
        int completed = 0;
        int max_polls = total_posted * 10; /* safety valve */
        while (completed < total_posted && !c->stop_requested && max_polls-- > 0) {
            struct ibv_wc wc[256];
            int n = ibv_poll_cq(c->cq, 256, wc);
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "[rdmaqpgen] WC error: status=%d vendor_err=%d\n",
                            wc[i].status, wc[i].vendor_err);
                } else {
                    c->bytes_sent += wc[i].byte_len;
                    window_bytes  += wc[i].byte_len;
                }
                completed++;
            }
        }

        /* --- Rate limiting: adjust gap based on achieved rate --- */
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed_s = (now.tv_sec - window_start.tv_sec)
                         + (now.tv_nsec - window_start.tv_nsec) / 1e9;

        long effective_gap = ba->gap_us;
        if (ba->target_gbps > 0 && elapsed_s > 0.1) {
            double achieved_gbps = (window_bytes * 8.0) / (elapsed_s * 1e9);
            if (achieved_gbps > ba->target_gbps * 1.05) {
                double ratio = achieved_gbps / ba->target_gbps;
                effective_gap = (long)(ba->gap_us * ratio);
                if (effective_gap > 1000000L) effective_gap = 1000000L;
            } else if (achieved_gbps < ba->target_gbps * 0.95 && effective_gap > 100) {
                effective_gap = effective_gap * 9 / 10;
            }
            window_bytes = 0;
            window_start = now;
        }

        /* --- Gap between bursts --- */
        if (effective_gap > 0 && !c->stop_requested) {
            usleep((useconds_t)effective_gap);
        }
    }

    c->running = 0;
    free(local_offsets);
    free(remote_offsets);
    free(ring);
    free(sges);
    free(wrs);
    free(ba);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API: burst loop control                                       */
/* ------------------------------------------------------------------ */

int rdmaqp_start_burst_loop(rdmaqp_ctx *c, long burst_us, long gap_us,
                            int target_gbps) {
    if (!c || c->running) return -1;

    burst_args *ba = malloc(sizeof(burst_args));
    if (!ba) return -1;
    ba->ctx         = c;
    ba->burst_us    = burst_us;
    ba->gap_us      = gap_us;
    ba->target_gbps = target_gbps;

    c->stop_requested = 0;
    int ret = pthread_create(&c->burst_thread, NULL, burst_loop_thread, ba);
    if (ret != 0) {
        free(ba);
        fprintf(stderr, "[rdmaqpgen] pthread_create failed: %s\n", strerror(ret));
        return -1;
    }

    printf("[rdmaqpgen] burst loop started (burst=%ldus gap=%ldus target=%dGbps)\n",
           burst_us, gap_us, target_gbps);
    return 0;
}

int rdmaqp_stop(rdmaqp_ctx *c) {
    if (!c) return -1;
    c->stop_requested = 1;
    if (c->running || c->burst_thread) {
        pthread_join(c->burst_thread, NULL);
        c->burst_thread = 0;
    }
    return 0;
}

uint64_t rdmaqp_bytes_sent(rdmaqp_ctx *c) {
    if (!c) return 0;
    return c->bytes_sent;
}
