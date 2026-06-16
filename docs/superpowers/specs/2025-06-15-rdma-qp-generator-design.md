# RDMA RC-QP All-to-All Pressure Generator — Design Spec

**Date:** 2025-06-15
**Author:** autoresearch-assisted design session
**Goal:** Replace `ib_write_bw` with a true `libibverbs` RC-QP generator that produces non-monotonic contention curves and per-strategy asymmetry in the cross-node CoLoRA benchmark.

---

## Motivation (Why)

The current EP traffic generator wraps `ib_write_bw` (perftest). `ib_write_bw` is a continuous streaming benchmark that produces smooth, monotonic contention curves. Real MoE EP all-to-all traffic creates sharp contention "steps" at specific loads due to:

1. **QP head-of-line blocking** — multiple RC QPs sharing the same link, one stalled WR blocks all QPs on the same port.
2. **RDMA WR queue depth exhaustion** — limited SQ depth per QP causes backpressure that cascades to other QPs.
3. **Per-strategy asymmetry** — S2 (bidirectional small RDMA messages, many QPs) degrades faster than S1 (one large RDMA read, few QPs) under QP contention.

The new generator replaces `ib_write_bw` with a custom C library that creates N RC QPs between UM253↔UM251, issues bursty RDMA WRITEs, and validates throughput via `/sys/class/infiniband` port counters.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 cross_node_benchmark.py                  │
│  (unchanged — already supports --ep-generator <choice>)  │
└─────────────────────┬────────────────────────────────────┘
                      │ calls .start(bw_pct) / .stop()
┌─────────────────────▼────────────────────────────────────┐
│              ep_traffic_generator.py                     │
│  EPTrafficGenerator                                      │
│    mode="verbs_qp" → delegates to RDMATrafficGenerator  │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│           rdma_qp_pressure.py (NEW)                      │
│  RDMATrafficGenerator                                    │
│    - loads librdmaqpgen.so via CFFI                      │
│    - TCP handshake with remote server                    │
│    - calls rdmaqp_start_burst_loop() / rdmaqp_stop()     │
│    - IB counter validation                               │
└─────────────────────┬────────────────────────────────────┘
                      │ CFFI (dlopen + ffi.cdef)
┌─────────────────────▼────────────────────────────────────┐
│           librdmaqpgen.so (NEW)                           │
│  C library: libibverbs RDMA RC QP management             │
│    - rdmaqp_create()      → PD, MR, CQ, N× QPs           │
│    - rdmaqp_start_burst_loop() → pthread burst loop      │
│    - rdmaqp_stop()        → signal + join                │
│    - rdmaqp_bytes_sent()  → cumulative tx bytes          │
│    - rdmaqp_destroy()     → teardown                      │
└──────────────────────────────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.c` | C library — QP create/burst/stop/destroy |
| `test/lora/avx/cross_node/rdma_qp_generator/rdmaqpgen.h` | C header — public API declarations |
| `test/lora/avx/cross_node/rdma_qp_generator/Makefile` | Build `librdmaqpgen.so` |
| `test/lora/avx/cross_node/rdma_qp_generator/rdma_qp_pressure.py` | Python CFFI wrapper |
| `test/lora/avx/cross_node/rdma_qp_generator/test_rdma_qp_generator.py` | Unit + integration tests |
| `test/lora/avx/cross_node/ep_traffic_generator.py` | Patch: add `"verbs_qp"` mode |

---

## C Library API (`rdmaqpgen.h`)

```c
typedef struct rdmaqp_ctx rdmaqp_ctx;

rdmaqp_ctx* rdmaqp_create(
    const char* mlx_device,    // "mlx5_0"
    int ib_port,               // 1
    int num_qps,               // 8–64 RC QPs
    int qp_depth,              // WRs per QP SQ (e.g. 128)
    int msg_bytes,             // per-WR payload (64KB–4MB)
    const char* remote_gid,    // GID string from ibv_query_gid
    int remote_qpn_base,       // first remote QP number
    char errbuf[256]
);

int rdmaqp_start_burst_loop(
    rdmaqp_ctx* ctx,
    long burst_us,             // ON duration (e.g. 8000 = 8ms)
    long gap_us,               // OFF duration (e.g. 2000 = 2ms)
    int target_gbps            // 0 = full speed; >0 scales gap
);

int rdmaqp_stop(rdmaqp_ctx* ctx);

uint64_t rdmaqp_bytes_sent(rdmaqp_ctx* ctx);

void rdmaqp_destroy(rdmaqp_ctx* ctx);
```

### Internal structure

Each RC QP:
- Connected to one remote QP via `ibv_modify_qp(RTR → RTS)`
- Posts `qp_depth` RDMA WRITEs into a pre-registered memory region
- WRs are posted in a ring-buffer pattern: the burst loop posts all WRs, polls CQ for completions, then reposts. No dynamic allocation in the hot path.

Memory region:
- One large MR (`msg_bytes * num_qps * qp_depth` bytes) registered with `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE`
- Divided into per-QP segments, each further divided into `qp_depth` slots
- Remote side has a matching MR (registered but unused — just a target for RDMA WRITE)

---

## Handshake Protocol

TCP control socket on port `base_port` (default 18515). Client = UM253, Server = UM251.

```
Server                            Client
  |                                 |
  |<---------- TCP connect ---------|
  |                                 |
  |-- JSON {qpn_base, gid, lid} -->|   (server QP info)
  |                                 |
  |<-- JSON {qpn_base, gid, lid} --|   (client QP info)
  |                                 |
  |          [both modify QPs RTR→RTS]
  |                                 |
  |-- JSON {ready: true} ---------->|
  |                                 |
  |<-- JSON {ready: true} ----------|
  |                                 |
  |===== burst loop begins =========|
```

Python `RDMATrafficGenerator` manages the TCP handshake before calling `rdmaqp_start_burst_loop()`.

---

## Rate Control

`target_gbps > 0` scales the gap period:
- Measure bytes sent per burst via CQ completions
- Compute achieved rate over the last window
- Adjust `gap_us` to converge on target: `gap_us = max(0, burst_us * (achieved_gbps / target_gbps - 1))`
- `target_gbps = 0` → `gap_us = 0`, full-speed bursts

This is a coarse windowed rate limiter (not per-message pacing). Good enough for first pass.

---

## Python Wrapper Interface

```python
class RDMATrafficGenerator:
    def __init__(self, config: EPTrafficConfig):
        """Load librdmaqpgen.so, discover local GID."""

    def start(self, bw_pct: int) -> None:
        """Start remote server, handshake, launch burst loop."""

    def stop(self) -> None:
        """Stop burst loop, destroy QPs, kill remote server."""

    def is_running(self) -> bool:
        """Check if burst loop is alive."""

    @property
    def bw_pct(self) -> int: ...
```

Integrated into `EPTrafficGenerator`:
- `config.mode == "verbs_qp"` → creates `RDMATrafficGenerator` internally
- No changes to `start()/stop()` public API

---

## Counter Validation

Same pattern as `ep_alltoall_traffic.py`:
1. Read `/sys/class/infiniband/<dev>/ports/<port>/counters/port_xmit_data` before start
2. Poll during burst loop (once per second)
3. Assert deltas > 0 → `RuntimeError` if counters don't increase
4. Counter scale: raw value × 4 bytes

---

## Benchmark Integration

No changes to `cross_node_benchmark.py`. Usage:

```bash
python cross_node_benchmark.py \
    --ep-generator verbs_qp \
    --ep-ssh-host 10.10.1.3 \
    --ep-qps 16 \
    --ep-qp-depth 128 \
    --ep-msg-bytes 65536 \
    ... (existing args unchanged)
```

New CLI args in `cross_node_benchmark.py`:
- `--ep-qps` (int, default 16): number of RC QPs
- `--ep-qp-depth` (int, default 128): WR depth per QP
- `--ep-msg-bytes` (int, default 65536): bytes per RDMA WRITE

---

## Success Criteria

1. **Build:** `make` produces `librdmaqpgen.so`, Python `import rdma_qp_pressure` succeeds
2. **Handshake:** Server/client QP handshake completes, burst loop starts
3. **Counter validation:** `port_xmit_data` increases during burst, validation passes
4. **Non-monotonic contention:** At EP=50% with 16 QPs, S2 degradation ratio > at EP=25% or EP=75%
5. **Per-strategy asymmetry:** S2 degrades ≥2× faster than S1 when QP count ≥ 8 at moderate EP load

---

## Out of Scope (deferred)

- GPUDirect RDMA (GPU memory registration)
- N-rank all-to-all (>2 nodes)
- Trace replay from MoE communication traces
- Per-message pacing rate limiter (coarse windowed is sufficient)
- QP priority/QoS markings

---

## Risks

| Risk | Mitigation |
|------|------------|
| `libibverbs` API mismatch (Mellanox OFED vs upstream) | Check `ibv_get_device_list` at startup, fail fast with actionable message |
| GID discovery across subnets | Both nodes on same IB subnet (10.10.1.0/24); use GID index 0 (RoCEv2) or port GID |
| QP state machine errors (RTR→RTS) | Validate `ibv_modify_qp` return codes, print `errno` on failure |
| CFFI callback overhead for timing | Hot loop is pure C (no Python callbacks); Python only calls start/stop |
| Remote server lifecycle | SSH-managed start/stop per EP level, same as existing `ib_write_bw` pattern |
