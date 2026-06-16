# RDMA RC-QP All-to-All Pressure Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `libibverbs`-based RDMA RC-QP pressure generator (`librdmaqpgen.so`) with a Python CFFI wrapper, then integrate it into the cross-node CoLoRA benchmark as a new `--ep-generator verbs_qp` mode.

**Architecture:** C library (`rdmaqpgen.c`) manages N RC QPs with RDMA WRITE burst loops. Python wrapper (`rdma_qp_pressure.py`) loads the .so via CFFI, handles TCP handshake between UM253↔UM251, and plugs into the existing `EPTrafficGenerator` class via `mode="verbs_qp"`. No changes to the benchmark orchestration — just a new generator mode.

**Tech Stack:** C11 + libibverbs (Mellanox OFED 54mlnx1), Python 3 + CFFI, pthreads

---

### Task 1: Create directory structure, header, and Makefile

**Files:**
- Create: `test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.h`
- Create: `test/lora/avx/cross_node/rdma_qp_generator/Makefile`

- [ ] **Step 1: Write the public header**

```c
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
 * Transition all QPs INIT → RTR → RTS using the remote peer info.
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
```

- [ ] **Step 2: Write the Makefile**

```makefile
# test/lora/avx/cross_node/rdma_qp_generator/Makefile
CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -O2 -fPIC
LDFLAGS  = -shared
LDLIBS   = -libverbs -lpthread

TARGET   = librdmaqpgen.so
SRC      = rdmaqpgen.c
OBJ      = rdmaqpgen.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $< $(LDLIBS)

$(OBJ): $(SRC) rdmaqpgen.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
```

- [ ] **Step 3: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/
git commit -m "feat(rdma-qp): add public header and Makefile for librdmaqpgen.so"
```

---

### Task 2: Implement rdmaqpgen.c (C library — QP lifecycle)

**Files:**
- Create: `test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c`

- [ ] **Step 1: Write rdmaqpgen.c — includes, struct, helpers**

```c
// test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c
#define _GNU_SOURCE
#include "rdmaqpgen.h"

#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
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
/* QP state machine: INIT → RTR → RTS                                 */
/* ------------------------------------------------------------------ */

static int modify_qp_to_rtr(struct ibv_qp *qp, int qp_num,
                            int ib_port, int remote_qpn, int remote_lid,
                            const union ibv_gid *remote_gid,
                            int gid_index) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state        = IBV_QPS_RTR;
    attr.path_mtu        = IBV_MTU_4096;
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
```

- [ ] **Step 2: Write rdmaqp_create()**

```c
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
    c->mr_size   = (size_t)msg_bytes * num_qps * qp_depth;

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

    /* --- port attributes (LID) --- */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(c->ctx, ib_port, &port_attr);
    if (ret) {
        snprintf_err(errbuf, 256, "ibv_query_port failed: %s", strerror(errno));
        goto fail;
    }
    c->local_info.lid = port_attr.lid;

    /* --- GID (index 0) --- */
    union ibv_gid gid;
    ret = ibv_query_gid(c->ctx, ib_port, 0, &gid);
    if (ret) {
        snprintf_err(errbuf, 256, "ibv_query_gid failed: %s", strerror(errno));
        goto fail;
    }
    gid_to_str(&gid, c->local_info.gid, sizeof(c->local_info.gid));

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
```

- [ ] **Step 3: Write rdmaqp_get_local_info() and rdmaqp_connect()**

```c
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

    /* Transition each QP INIT → RTR → RTS */
    for (int i = 0; i < c->num_qps; i++) {
        int remote_qpn = remote->qpn_base + i;

        if (modify_qp_to_rtr(c->qps[i], c->qps[i]->qp_num,
                             c->ib_port, remote_qpn, remote->lid,
                             &rgid, 0) != 0) {
            snprintf_err(errbuf, 256,
                         "modify_qp_to_rtr[%d] (remote_qpn=%d) failed: %s",
                         i, remote_qpn, strerror(errno));
            return -1;
        }

        if (modify_qp_to_rts(c->qps[i]) != 0) {
            snprintf_err(errbuf, 256,
                         "modify_qp_to_rts[%d] failed: %s",
                         i, strerror(errno));
            return -1;
        }
    }

    printf("[rdmaqpgen] %d QPs connected to remote QPN base %d\n",
           c->num_qps, remote->qpn_base);
    return 0;
}
```

- [ ] **Step 4: Write rdmaqp_destroy()**

```c
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
```

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c
git commit -m "feat(rdma-qp): implement QP lifecycle — create, connect, destroy"
```

---

### Task 3: Implement burst loop in rdmaqpgen.c

**Files:**
- Modify: `test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c` (append burst loop)

- [ ] **Step 1: Write burst loop internals**

```c
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
    for (int i = 0; i < num_qps; i++) {
        local_offsets[i]  = (uint64_t)(uintptr_t)c->mr->addr + (size_t)i * qp_depth * msg_bytes;
        remote_offsets[i] = remote_addr + (size_t)i * qp_depth * msg_bytes;
    }

    /* Per-QP ring indices (which WR slot to post next) */
    int *ring = calloc(num_qps, sizeof(int));

    /* Allocate SGE and WR arrays per QP */
    struct ibv_sge  *sges = calloc(num_qps * qp_depth, sizeof(struct ibv_sge));
    struct ibv_send_wr *wrs = calloc(num_qps * qp_depth, sizeof(struct ibv_send_wr));

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
            wrs[idx].wr.rdma.remote_addr = remote_offsets[q] + (size_t)s * msg_bytes;
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
        for (int q = 0; q < num_qps; q++) {
            for (int s = 0; s < qp_depth; s++) {
                int idx = q * qp_depth + ring[q];
                struct ibv_send_wr *bad = NULL;
                int ret = ibv_post_send(c->qps[q], &wrs[idx], &bad);
                if (ret != 0) {
                    fprintf(stderr, "[rdmaqpgen] ibv_post_send QP[%d] failed: %s\n",
                            q, strerror(ret));
                    /* Drain CQ and continue */
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
                /* Too fast — increase gap */
                double ratio = achieved_gbps / ba->target_gbps;
                effective_gap = (long)(ba->gap_us * ratio);
                if (effective_gap > 1000000L) effective_gap = 1000000L;
            } else if (achieved_gbps < ba->target_gbps * 0.95 && effective_gap > 100) {
                effective_gap = effective_gap * 9 / 10;
            }
            /* Reset window */
            window_bytes = 0;
            window_start = now;
        }

        /* --- Gap between bursts --- */
        if (effective_gap > 0 && !c->stop_requested) {
            usleep(effective_gap);
        }
    }

    c->running = 0;
    free(local_offsets);
    free(remote_offsets);
    free(ring);
    free(sges);
    free(wrs);
    return NULL;
}
```

- [ ] **Step 2: Write rdmaqp_start_burst_loop(), rdmaqp_stop(), rdmaqp_bytes_sent()**

```c
int rdmaqp_start_burst_loop(rdmaqp_ctx *c, long burst_us, long gap_us,
                            int target_gbps) {
    if (!c || c->running) return -1;

    burst_args *ba = malloc(sizeof(burst_args));
    ba->ctx        = c;
    ba->burst_us   = burst_us;
    ba->gap_us     = gap_us;
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
```

- [ ] **Step 3: Add missing include for stdarg.h at top of rdmaqpgen.c**

```c
/* Add after existing includes at the top of rdmaqpgen.c */
#include <stdarg.h>
```

- [ ] **Step 4: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c
git commit -m "feat(rdma-qp): implement burst loop with rate-limited RDMA WRITEs"
```

---

### Task 4: Build and smoke-test librdmaqpgen.so

- [ ] **Step 1: Build**

```bash
cd test/lora/avx/cross_node/rdma_qp_generator && make clean && make
```

Expected: `librdmaqpgen.so` created, no warnings.

- [ ] **Step 2: Verify symbols are exported**

```bash
nm -D test/lora/avx/cross_node/rdma_qp_generator/librdmaqpgen.so | grep -E 'rdmaqp_'
```

Expected: `rdmaqp_create`, `rdmaqp_get_local_info`, `rdmaqp_connect`, `rdmaqp_start_burst_loop`, `rdmaqp_stop`, `rdmaqp_bytes_sent`, `rdmaqp_destroy` all visible (T).

- [ ] **Step 3: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/librdmaqpgen.so
# If .gitignore excludes .so, note it; the Makefile will rebuild it.
# Check: git check-ignore -v test/lora/avx/cross_node/rdma_qp_generator/librdmaqpgen.so
git commit -m "build(rdma-qp): compiled librdmaqpgen.so"
```

---

### Task 5: Implement Python CFFI wrapper (rdma_qp_pressure.py)

**Files:**
- Create: `test/lora/avx/cross_node/rdma_qp_generator/rdma_qp_pressure.py`

- [ ] **Step 1: Write rdma_qp_pressure.py**

```python
#!/usr/bin/env python3
"""
Python CFFI wrapper for librdmaqpgen.so — RDMA RC-QP pressure generator.

Provides RDMATrafficGenerator with the same start/stop/is_running interface
as EPTrafficGenerator, so it plugs in as mode="verbs_qp".
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SO_PATH = str(_HERE / "librdmaqpgen.so")

# Build the .so if missing
if not os.path.exists(_SO_PATH):
    subprocess.run(["make", "-C", str(_HERE)], check=True)

from cffi import FFI  # noqa: E402

ffi = FFI()
ffi.cdef("""
typedef struct rdmaqp_ctx rdmaqp_ctx;

typedef struct {
    int      qpn_base;
    int      lid;
    char     gid[33];
    uint64_t mr_addr;
    uint32_t mr_rkey;
} rdmaqp_peer_info;

rdmaqp_ctx* rdmaqp_create(
    const char *mlx_device,
    int         ib_port,
    int         num_qps,
    int         qp_depth,
    int         msg_bytes,
    char        errbuf[256]);

int rdmaqp_get_local_info(rdmaqp_ctx *ctx, rdmaqp_peer_info *info);
int rdmaqp_connect(rdmaqp_ctx *ctx, const rdmaqp_peer_info *remote,
                   char errbuf[256]);
int rdmaqp_start_burst_loop(rdmaqp_ctx *ctx, long burst_us, long gap_us,
                            int target_gbps);
int rdmaqp_stop(rdmaqp_ctx *ctx);
uint64_t rdmaqp_bytes_sent(rdmaqp_ctx *ctx);
void rdmaqp_destroy(rdmaqp_ctx *ctx);
""")

_lib = ffi.dlopen(_SO_PATH)

# Link capacity constants
LINK_CAPACITY_GBPS = 200
DEFAULT_QP_DEPTH = 128
DEFAULT_MSG_BYTES = 65536   # 64 KB per WR
DEFAULT_NUM_QPS = 16
DEFAULT_BURST_US = 8000     # 8 ms
DEFAULT_GAP_US = 2000       # 2 ms


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed during handshake")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_json(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_json(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    size = struct.unpack("!I", header)[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


class RDMATrafficGenerator:
    """RDMA RC-QP pressure generator wrapping librdmaqpgen.so."""

    def __init__(
        self,
        local_ip: str,
        remote_ip: str,
        mlx_device: str = "mlx5_0",
        ib_port: int = 1,
        num_qps: int = DEFAULT_NUM_QPS,
        qp_depth: int = DEFAULT_QP_DEPTH,
        msg_bytes: int = DEFAULT_MSG_BYTES,
        control_port: int = 18515,
        remote_ssh_host: Optional[str] = None,
        burst_us: int = DEFAULT_BURST_US,
        gap_us: int = DEFAULT_GAP_US,
    ) -> None:
        self._local_ip = local_ip
        self._remote_ip = remote_ip
        self._mlx_device = mlx_device
        self._ib_port = ib_port
        self._num_qps = num_qps
        self._qp_depth = qp_depth
        self._msg_bytes = msg_bytes
        self._control_port = control_port
        self._remote_ssh_host = remote_ssh_host
        self._burst_us = burst_us
        self._gap_us = gap_us

        self._ctx = ffi.NULL
        self._running = False
        self._bw_pct = 0
        self._remote_server_started = False

    # ------------------------------------------------------------------
    # Public API (compatible with EPTrafficGenerator)
    # ------------------------------------------------------------------

    def start(self, bw_pct: int) -> None:
        if self._running:
            self.stop()

        self._bw_pct = bw_pct

        if bw_pct == 0:
            print(
                f"[RDMATrafficGenerator] disabled for 0% BW baseline "
                f"(local={self._local_ip}, remote={self._remote_ip})",
                flush=True,
            )
            return

        target_gbps = int(LINK_CAPACITY_GBPS * bw_pct / 100.0)

        # Create local context
        errbuf = ffi.new("char[256]")
        self._ctx = _lib.rdmaqp_create(
            self._mlx_device.encode(),
            self._ib_port,
            self._num_qps,
            self._qp_depth,
            self._msg_bytes,
            errbuf,
        )
        if self._ctx == ffi.NULL:
            msg = ffi.string(errbuf).decode()
            self._ctx = ffi.NULL
            raise RuntimeError(f"rdmaqp_create failed: {msg}")

        # Start remote server and handshake
        self._start_remote_server()
        local_info = self._handshake()

        # Connect QPs
        remote_info = ffi.new("rdmaqp_peer_info *")
        remote_info.qpn_base = local_info["remote_qpn_base"]
        remote_info.lid = local_info["remote_lid"]
        gid_bytes = local_info["remote_gid"].encode()
        ffi.memmove(remote_info.gid, gid_bytes, min(len(gid_bytes), 32))
        remote_info.gid[32] = 0
        remote_info.mr_addr = local_info["remote_mr_addr"]
        remote_info.mr_rkey = local_info["remote_mr_rkey"]

        ret = _lib.rdmaqp_connect(self._ctx, remote_info, errbuf)
        if ret != 0:
            msg = ffi.string(errbuf).decode()
            raise RuntimeError(f"rdmaqp_connect failed: {msg}")

        # Start burst loop
        ret = _lib.rdmaqp_start_burst_loop(
            self._ctx, self._burst_us, self._gap_us, target_gbps,
        )
        if ret != 0:
            raise RuntimeError("rdmaqp_start_burst_loop failed")

        self._running = True
        print(
            f"[RDMATrafficGenerator] started ({self._num_qps} QPs, {bw_pct}% BW, "
            f"target={target_gbps}Gbps, local={self._local_ip}, "
            f"remote={self._remote_ip})",
            flush=True,
        )

    def stop(self) -> None:
        if self._ctx != ffi.NULL:
            if self._running:
                _lib.rdmaqp_stop(self._ctx)
                self._running = False
            _lib.rdmaqp_destroy(self._ctx)
            self._ctx = ffi.NULL
        self._stop_remote_server()
        print("[RDMATrafficGenerator] stopped", flush=True)

    def is_running(self) -> bool:
        return self._running

    @property
    def bw_pct(self) -> int:
        return self._bw_pct

    # ------------------------------------------------------------------
    # Handshake internals
    # ------------------------------------------------------------------

    def _handshake(self) -> dict:
        """TCP handshake: exchange QP/GID/MR info, return remote info dict."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Try connecting with retries (server may still be starting)
        deadline = time.monotonic() + 10.0
        last_err = None
        while time.monotonic() < deadline:
            try:
                sock.connect((self._remote_ip, self._control_port))
                break
            except (ConnectionRefusedError, OSError) as exc:
                last_err = exc
                time.sleep(0.5)
        else:
            raise RuntimeError(
                f"Handshake: could not connect to {self._remote_ip}:"
                f"{self._control_port}: {last_err}"
            )

        try:
            # Send local info
            local = ffi.new("rdmaqp_peer_info *")
            _lib.rdmaqp_get_local_info(self._ctx, local)
            local_dict = {
                "qpn_base": local.qpn_base,
                "lid": local.lid,
                "gid": ffi.string(local.gid).decode(),
                "mr_addr": local.mr_addr,
                "mr_rkey": local.mr_rkey,
            }
            _send_json(sock, local_dict)

            # Receive remote info
            remote_dict = _recv_json(sock)
            return remote_dict
        finally:
            sock.close()

    def _start_remote_server(self) -> None:
        if not self._remote_ssh_host:
            return
        # The remote server is started by the same CFFI wrapper running
        # in server mode. We push a small server script via SSH.
        server_script = (
            f"import sys; sys.path.insert(0, '{_HERE.parent}'); "
            f"from rdma_qp_generator.rdma_qp_pressure import _run_server; "
            f"_run_server('{self._remote_ip}', {self._control_port}, "
            f"'{self._mlx_device}', {self._ib_port}, {self._num_qps}, "
            f"{self._qp_depth}, {self._msg_bytes})"
        )
        # Kill any stale server first
        subprocess.run(
            ["ssh", self._remote_ssh_host,
             "pkill -x ib_write_bw 2>/dev/null || true; "
             f"pkill -f 'rdma_qp_pressure.*_run_server' 2>/dev/null || true"],
            timeout=5, capture_output=True,
        )
        self._remote_proc = subprocess.Popen(
            ["ssh", self._remote_ssh_host,
             "nohup python3 -c " + shlex.quote(server_script) +
             " > /tmp/rdma_qp_server.log 2>&1 < /dev/null &"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._remote_server_started = True
        time.sleep(2)  # give the remote server time to bind

    def _stop_remote_server(self) -> None:
        if not self._remote_server_started or not self._remote_ssh_host:
            return
        subprocess.run(
            ["ssh", self._remote_ssh_host,
             "pkill -f 'rdma_qp_pressure.*_run_server' 2>/dev/null || true"],
            timeout=5, capture_output=True,
        )
        self._remote_server_started = False
```

- [ ] **Step 2: Write server-side helper (_run_server) in same file**

```python
# Append to rdma_qp_pressure.py:

import shlex  # noqa: E402 (add to top imports)

def _run_server(
    bind_ip: str,
    port: int,
    mlx_device: str,
    ib_port: int,
    num_qps: int,
    qp_depth: int,
    msg_bytes: int,
) -> None:
    """Run the RDMA QP server side (called via SSH from client)."""
    print(f"[rdma_qp_server] starting on {bind_ip}:{port}", flush=True)

    errbuf = ffi.new("char[256]")
    ctx = _lib.rdmaqp_create(
        mlx_device.encode(), ib_port, num_qps, qp_depth, msg_bytes, errbuf,
    )
    if ctx == ffi.NULL:
        msg = ffi.string(errbuf).decode()
        raise RuntimeError(f"server rdmaqp_create failed: {msg}")

    try:
        # Accept one TCP connection
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind((bind_ip, port))
        listen_sock.listen(1)
        print(f"[rdma_qp_server] listening on {bind_ip}:{port}", flush=True)

        conn, addr = listen_sock.accept()
        print(f"[rdma_qp_server] accepted from {addr}", flush=True)
        listen_sock.close()

        # Receive client info
        client_dict = _recv_json(conn)

        # Send local info
        local = ffi.new("rdmaqp_peer_info *")
        _lib.rdmaqp_get_local_info(ctx, local)
        local_dict = {
            "qpn_base": local.qpn_base,
            "lid": local.lid,
            "gid": ffi.string(local.gid).decode(),
            "mr_addr": local.mr_addr,
            "mr_rkey": local.mr_rkey,
        }
        _send_json(conn, local_dict)
        conn.close()

        # Connect QPs to client
        remote = ffi.new("rdmaqp_peer_info *")
        remote.qpn_base = client_dict["qpn_base"]
        remote.lid = client_dict["lid"]
        gid_bytes = client_dict["gid"].encode()
        ffi.memmove(remote.gid, gid_bytes, min(len(gid_bytes), 32))
        remote.gid[32] = 0
        remote.mr_addr = client_dict["mr_addr"]
        remote.mr_rkey = client_dict["mr_rkey"]

        ret = _lib.rdmaqp_connect(ctx, remote, errbuf)
        if ret != 0:
            msg = ffi.string(errbuf).decode()
            raise RuntimeError(f"server rdmaqp_connect failed: {msg}")

        # Start burst loop (full speed — rate limiting is client-side)
        ret = _lib.rdmaqp_start_burst_loop(ctx, 8000, 2000, 0)
        if ret != 0:
            raise RuntimeError("server rdmaqp_start_burst_loop failed")

        # Wait until killed
        print("[rdma_qp_server] burst loop running, waiting for SIGTERM",
              flush=True)
        while True:
            time.sleep(5)

    except KeyboardInterrupt:
        pass
    finally:
        _lib.rdmaqp_stop(ctx)
        _lib.rdmaqp_destroy(ctx)
        print("[rdma_qp_server] stopped", flush=True)
```

- [ ] **Step 3: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/rdma_qp_pressure.py
git commit -m "feat(rdma-qp): Python CFFI wrapper with TCP handshake and server mode"
```

---

### Task 6: Patch ep_traffic_generator.py to support verbs_qp mode

**Files:**
- Modify: `test/lora/avx/cross_node/ep_traffic_generator.py`

- [ ] **Step 1: Add `verbs_qp` mode to EPTrafficGenerator**

```python
# In ep_traffic_generator.py, in EPTrafficGenerator.__init__(), after the
# existing mode dispatch, add:

        self._verbs_gen = None
        if self.config.mode == "verbs_qp":
            from rdma_qp_generator.rdma_qp_pressure import RDMATrafficGenerator
            self._verbs_gen = RDMATrafficGenerator(
                local_ip=self.config.local_ip,
                remote_ip=self.config.remote_ip,
                mlx_device=self.config.mlx_device,
                ib_port=self.config.ib_port,
                num_qps=self.config.num_qps,
                qp_depth=self.config.qp_depth,
                msg_bytes=self.config.msg_bytes,
                control_port=self.config.base_port,
                remote_ssh_host=self.config.remote_ssh_host,
                burst_us=int(self.config.burst_on_ms * 1000),
                gap_us=int(self.config.burst_off_ms * 1000),
            )
```

- [ ] **Step 2: Delegate start/stop/is_running when verbs_gen is active**

```python
# In EPTrafficGenerator.start(), replace the existing start logic with:

    def start(self, bw_pct: int = 0) -> None:
        with self._lock:
            if self._running:
                self.stop()

            self._bw_pct = bw_pct
            self._running = True

            if self._verbs_gen is not None:
                self._verbs_gen.start(bw_pct)
                return

            # ... existing start logic for ib_write_bw and alltoall ...

    def stop(self) -> None:
        with self._lock:
            if self._verbs_gen is not None:
                self._verbs_gen.stop()
                self._running = False
                return
            # ... existing stop logic ...

    def is_running(self) -> bool:
        with self._lock:
            if self._verbs_gen is not None:
                return self._verbs_gen.is_running()
            # ... existing is_running logic ...
```

- [ ] **Step 3: Add num_qps, qp_depth, msg_bytes fields to EPTrafficConfig**

```python
# In EPTrafficConfig dataclass, add these fields:
    num_qps: int = 16
    qp_depth: int = 128
    msg_bytes: int = 65536
```

- [ ] **Step 4: Commit**

```bash
git add test/lora/avx/cross_node/ep_traffic_generator.py
git commit -m "feat(rdma-qp): add verbs_qp mode to EPTrafficGenerator"
```

---

### Task 7: Patch cross_node_benchmark.py with CLI args

**Files:**
- Modify: `test/lora/avx/cross_node/cross_node_benchmark.py`

- [ ] **Step 1: Add `verbs_qp` to `--ep-generator` choices**

```python
# In the EP Traffic argument group, change:
        "--ep-generator",
        choices=("ib_write_bw", "alltoall"),
        default="ib_write_bw",

# To:
        "--ep-generator",
        choices=("ib_write_bw", "alltoall", "verbs_qp"),
        default="ib_write_bw",
```

- [ ] **Step 2: Add QP-specific CLI args**

```python
# In the EP Traffic argument group, add:
    g.add_argument(
        "--ep-qps",
        type=int,
        default=16,
        help="Number of RC QPs for verbs_qp generator",
    )
    g.add_argument(
        "--ep-qp-depth",
        type=int,
        default=128,
        help="Send-queue WR depth per QP for verbs_qp generator",
    )
    g.add_argument(
        "--ep-msg-bytes",
        type=int,
        default=65536,
        help="RDMA WRITE payload bytes for verbs_qp generator",
    )
```

- [ ] **Step 3: Pass new args to EPTrafficConfig**

```python
# In main(), where EPTrafficConfig is constructed, add:
        num_qps=args.ep_qps,
        qp_depth=args.ep_qp_depth,
        msg_bytes=args.ep_msg_bytes,
```

- [ ] **Step 4: Commit**

```bash
git add test/lora/avx/cross_node/cross_node_benchmark.py
git commit -m "feat(rdma-qp): add --ep-generator verbs_qp and QP CLI args to benchmark"
```

---

### Task 8: Unit tests

**Files:**
- Create: `test/lora/avx/cross_node/rdma_qp_generator/test_rdma_qp_generator.py`

- [ ] **Step 1: Write tests for EPTrafficConfig verbs_qp mode**

```python
import unittest

from ep_traffic_generator import EPTrafficConfig


class TestVerbsQPTrafficConfig(unittest.TestCase):
    def test_verbs_qp_server_command_is_not_used(self):
        """verbs_qp mode handles server lifecycle via Python, not server_cmd."""
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        # server_cmd is not the dispatch path for verbs_qp
        # (RDMATrafficGenerator handles it internally)
        self.assertEqual(config.mode, "verbs_qp")

    def test_verbs_qp_client_command_is_not_used(self):
        """client_cmd is not used for verbs_qp mode."""
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        cmd = config.client_cmd(50)
        # Should still return something reasonable (not crash)
        self.assertIsInstance(cmd, list)

    def test_default_qp_params(self):
        config = EPTrafficConfig(
            local_ip="10.10.1.1",
            remote_ip="10.10.1.3",
            mode="verbs_qp",
        )
        self.assertEqual(config.num_qps, 16)
        self.assertEqual(config.qp_depth, 128)
        self.assertEqual(config.msg_bytes, 65536)
```

- [ ] **Step 2: Write CFFI import test**

```python
    def test_can_import_rdma_qp_pressure(self):
        """librdmaqpgen.so loads and symbols resolve."""
        from rdma_qp_generator import rdma_qp_pressure
        self.assertTrue(hasattr(rdma_qp_pressure, 'RDMATrafficGenerator'))
        self.assertTrue(hasattr(rdma_qp_pressure, '_lib'))
        # Verify key C functions are callable
        lib = rdma_qp_pressure._lib
        self.assertIsNotNone(lib.rdmaqp_create)
        self.assertIsNotNone(lib.rdmaqp_destroy)
```

- [ ] **Step 3: Run tests**

```bash
cd /home/shufan/LightLLM-integrate-to-SLoRA && python -m pytest test/lora/avx/cross_node/rdma_qp_generator/test_rdma_qp_generator.py -v
```

Expected: all tests pass. The CFFI import test will fail if `librdmaqpgen.so` hasn't been built — that's expected, build first with `make`.

- [ ] **Step 4: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/test_rdma_qp_generator.py
git commit -m "test(rdma-qp): unit tests for verbs_qp config and CFFI import"
```

---

### Task 9: Add IB counter validation to Python wrapper

**Files:**
- Modify: `test/lora/avx/cross_node/rdma_qp_generator/rdma_qp_pressure.py`

- [ ] **Step 1: Add counter validation to RDMATrafficGenerator**

```python
# Add to RDMATrafficGenerator class in rdma_qp_pressure.py:

    def _validate_counters(self) -> None:
        """Read IB port counters before and during burst; assert traffic flows."""
        counter_paths = {
            "xmit": Path(f"/sys/class/infiniband/{self._mlx_device}"
                         f"/ports/{self._ib_port}/counters/port_xmit_data"),
            "rcv": Path(f"/sys/class/infiniband/{self._mlx_device}"
                        f"/ports/{self._ib_port}/counters/port_rcv_data"),
        }
        scale = 4  # bytes per counter unit

        before = {}
        for name, path in counter_paths.items():
            before[name] = int(path.read_text().strip()) * scale

        # Wait for traffic to flow
        time.sleep(1.0)

        after = {}
        for name, path in counter_paths.items():
            after[name] = int(path.read_text().strip()) * scale

        deltas = {name: after[name] - before[name] for name in counter_paths}
        print(f"[RDMATrafficGenerator] IB counter deltas: {deltas}", flush=True)

        if deltas["xmit"] <= 0 and deltas["rcv"] <= 0:
            raise RuntimeError(
                f"IB port counters did not increase after burst start: {deltas}. "
                f"RDMA traffic may not be flowing."
            )
```

- [ ] **Step 2: Call validation after burst loop starts**

```python
# In RDMATrafficGenerator.start(), after rdmaqp_start_burst_loop succeeds
# and before printing the "started" message, add:

        # Validate that traffic is actually flowing
        self._validate_counters()
```

- [ ] **Step 3: Commit**

```bash
git add test/lora/avx/cross_node/rdma_qp_generator/rdma_qp_pressure.py
git commit -m "feat(rdma-qp): add IB port counter validation after burst start"
```

---

### Task 10: Two-node integration test

> **Note:** Tasks 10 and 11 are manual integration steps, not automated — they require both nodes to be reachable.

- [ ] **Step 1: Verify the build on UM251**

```bash
ssh 10.10.1.3 "cd /home/shufan/LightLLM-integrate-to-SLoRA && make -C test/lora/avx/cross_node/rdma_qp_generator clean all"
```

Expected: `librdmaqpgen.so` built on UM251.

- [ ] **Step 2: Run a focused benchmark sweep**

```bash
python test/lora/avx/cross_node/cross_node_benchmark.py \
    --master-addr 10.10.1.1 \
    --master-port 29500 \
    --server-port 29501 \
    --ranks 64 \
    --num-miss-list 2 \
    --ep-bw-pct-list 0,50,90 \
    --warmup 5 \
    --iters 50 \
    --output-dir results/cross_node_benchmark_verbs_qp \
    --mlx-device mlx5_0 \
    --ep-generator verbs_qp \
    --ep-ssh-host 10.10.1.3 \
    --ep-qps 16 \
    --ep-qp-depth 128 \
    --ep-msg-bytes 65536
```

Expected:
- Handshake completes for each EP level
- Burst loop starts/stops cleanly
- CSV rows generated for all 3 EP levels
- No crashes or `RuntimeError`

- [ ] **Step 3: Verify CSV output**

```bash
cat results/cross_node_benchmark_verbs_qp/cross_node_benchmark.csv
```

Expected: 9 rows (3 EP levels × 3 strategies), each with valid `total_ms`.

- [ ] **Step 4: Check for non-monotonic contention**

```bash
python - << 'PY'
import csv
rows = list(csv.DictReader(open("results/cross_node_benchmark_verbs_qp/cross_node_benchmark.csv")))
for r in rows:
    print(f"EP={r['ep_bw_pct']} S{r['strategy']} total={float(r['total_ms'])*1000:.0f}us")
PY
```

Expected: S2 at EP=50% should show higher latency than S2 at EP=0%, and ideally S2 at EP=75% should not be monotonic with EP=50%.

- [ ] **Step 5: Commit results (if results dir is tracked) or note them**

---

### Task 10: Self-review checklist

- [ ] All C symbols exported correctly
- [ ] Python CFFI wrapper handles error paths (create failure, connect failure, handshake timeout)
- [ ] Remote server lifecycle managed cleanly (started per EP level, killed on stop)
- [ ] Burst loop exits cleanly on `rdmaqp_stop()` (no stuck pthreads)
- [ ] IB counter validation present (can be added as follow-up if time-constrained)
- [ ] No `.so` files committed if gitignored
