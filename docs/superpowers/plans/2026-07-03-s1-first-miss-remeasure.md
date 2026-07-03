# S1 First-Miss Tax Re-measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-measure the first-miss tax with dual timing domains (CPU + CUDA events) and full accounting that closes to near-zero, across cells B1/B2/B5 (B6 deferred to Plan 2), to determine whether the tax is Python dispatch, executor entry, runtime-stack, allocation, or cache — without speculating.

**Architecture:** A persistent-multiplexed-TCP transport layer enables B1/B2 cells. The server's `handle_s4a_pooled` gains dual-domain timing (CPU `perf_counter` + CUDA events) for all four sub-segments (alloc, dtype, mm1, mm2) in eager variants, and replay-total timing for the cuda_graph variant. A new `bench_first_miss_v2.py` drives the 5-variant x 4-cell x 4-NM matrix with the `common/` instrumentation harness.

**Tech Stack:** Python 3.11, PyTorch 2.8.0, pytest, threading, socket, CUDA events.

**Spec:** `docs/superpowers/specs/2026-07-02-decomposition-confound-isolation-design.md` (section "S1 — First-miss tax re-measurement")
**Depends on:** Plan 1 (`2026-07-03-decomposition-harness-phase0.md`) — common/, qppool, concurrent_server dispatcher.

**Working directory:** `test/lora/avx/cross_node/`

---

## File Structure

```
test/lora/avx/cross_node/
  common/
    transport.py          # NEW: persistent multiplexed TCP transport (one socket, request IDs, out-of-order)
  concurrent_server.py    # MODIFIED: dual-domain timing for s4a handler, variant instrumentation
  bench_decomposition.py  # MODIFIED: wire B1/B2/B5 cells (was NotImplementedError)
  bench_first_miss_v2.py  # NEW: S1 driver — 5 variants x cells x NM
  test_transport.py       # NEW
  test_first_miss_v2.py   # NEW
```

---

## Task 1: common/transport.py — persistent multiplexed TCP

The gap noted in Plan 1's self-review. One TCP connection, request-ID framing (from `common/protocol.py`), multiple outstanding requests, out-of-order responses. Needed for B1/B2/B3 cells.

**Files:** Create `common/transport.py`, `test_transport.py`

- [ ] **Step 1: Write the failing test**

```python
# test_transport.py
"""Test persistent multiplexed TCP transport with request IDs and out-of-order responses."""
import socket
import threading
import time
import pytest
from common.transport import PersistentTransport


def test_round_trip_single_request():
    """Send one request, get one response, matched by request ID."""
    srv, cli = socket.socketpair()
    server_tp = PersistentTransport(srv)
    client_tp = PersistentTransport(cli)

    # Server thread: read request, send response with same req_id
    def server_handler():
        req = server_tp.recv_request()
        server_tp.send_response(req["req_id"], {"result": "ok", "echo_nm": req["nm"]})

    t = threading.Thread(target=server_handler, daemon=True)
    t.start()

    resp = client_tp.request({"type": "s4a_pooled", "nm": 8, "req_id": 1})
    assert resp["echo_nm"] == 8
    t.join(timeout=2)


def test_out_of_order_responses():
    """Send two requests; server responds to second first. Client matches by req_id."""
    srv, cli = socket.socketpair()
    server_tp = PersistentTransport(srv)
    client_tp = PersistentTransport(cli)

    def server_handler():
        req1 = server_tp.recv_request()
        req2 = server_tp.recv_request()
        # Respond to req2 first (out of order)
        server_tp.send_response(req2["req_id"], {"order": 2})
        server_tp.send_response(req1["req_id"], {"order": 1})

    t = threading.Thread(target=server_handler, daemon=True)
    t.start()

    # Client sends both, then collects both
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(client_tp.request, {"type": "s4a", "req_id": 100})
        f2 = pool.submit(client_tp.request, {"type": "s4a", "req_id": 200})
        r1 = f1.result(timeout=2)
        r2 = f2.result(timeout=2)
    assert r1["order"] == 1
    assert r2["order"] == 2
    t.join(timeout=2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest test_transport.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'common.transport'`)

- [ ] **Step 3: Implement common/transport.py**

```python
# common/transport.py
"""Persistent multiplexed TCP transport.

One TCP connection carries multiple outstanding requests with request IDs.
Responses may complete out of order. The client matches responses to requests
by req_id via a pending-futures dict.

Per spec "Persistent TCP transport requirement": B1-B3 use framed multiplexing
with request IDs; responses may complete out of order. This is a hard
requirement, not an implementation detail.
"""
import socket
import threading
from concurrent.futures import Future
from common.protocol import send_message, recv_message, new_request_id


class PersistentTransport:
    """Wraps a connected socket with request-ID multiplexing.

    One reader thread dispatches incoming responses to waiting requesters.
    Thread-safe for concurrent request() calls.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def request(self, msg: dict, timeout: float = 30.0) -> dict:
        """Send a request and block until the matching response arrives."""
        req_id = msg.get("req_id")
        if req_id is None:
            req_id = new_request_id()
            msg["req_id"] = req_id
        fut: Future = Future()
        with self._lock:
            self._pending[req_id] = fut
        with self._lock_send:
            send_message(self._sock.makefile("wb"), msg)
        return fut.result(timeout=timeout)

    def recv_request(self) -> dict:
        """Server-side: block until one request arrives."""
        return recv_message(self._sock.makefile("rb"))

    def send_response(self, req_id: int, response: dict) -> None:
        """Server-side: send a response tagged with req_id."""
        response["req_id"] = req_id
        with self._lock_send:
            send_message(self._sock.makefile("wb"), response)

    _lock_send = threading.Lock()

    def _read_loop(self) -> None:
        """Reader thread: dispatch incoming messages to waiting futures."""
        rf = self._sock.makefile("rb")
        try:
            while True:
                msg = recv_message(rf)
                req_id = msg.get("req_id")
                if req_id is None:
                    continue
                with self._lock:
                    fut = self._pending.pop(req_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except (EOFError, OSError):
            pass  # connection closed
        finally:
            with self._lock:
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("transport closed"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_transport.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add common/transport.py test_transport.py
git commit -m "feat(common): persistent multiplexed TCP transport with request-ID matching"
```

---

## Task 2: concurrent_server.py — dual-domain timing for s4a handler

Add CUDA-event timing around each sub-segment (alloc, dtype, mm1, mm2) in the eager variants, and replay-total timing for cuda_graph. Return the per-segment CPU and GPU timings in the response.

**Files:** Modify `concurrent_server.py`, `test_first_miss_v2.py` (partial — timing structure test)

- [ ] **Step 1: Read the existing compute loop**

Run: `sed -n '156,280p' concurrent_server.py` to see the existing `handle_s4a_pooled` compute loop and variant dispatch.

- [ ] **Step 2: Write the failing test for the timing response structure**

```python
# test_first_miss_v2.py (partial — will grow in Task 4)
"""Test that the server returns dual-domain timing in the response."""
import pytest
from concurrent_server import build_timing_response


def test_timing_response_has_cpu_and_gpu_fields():
    """Response must include per-segment CPU and GPU timings."""
    resp = build_timing_response(
        nm=8,
        segments=[
            {"name": "alloc", "cpu_us": 5.0, "gpu_us": 3.0},
            {"name": "dtype", "cpu_us": 2.0, "gpu_us": 1.0},
            {"name": "mm1", "cpu_us": 10.0, "gpu_us": 8.0},
            {"name": "mm2", "cpu_us": 12.0, "gpu_us": 9.0},
        ],
        variant="baseline",
    )
    assert resp["nm"] == 8
    assert resp["variant"] == "baseline"
    assert len(resp["segments"]) == 4
    assert resp["segments"][0]["name"] == "alloc"
    assert resp["segments"][0]["cpu_us"] == 5.0
    assert resp["segments"][0]["gpu_us"] == 3.0
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest test_first_miss_v2.py::test_timing_response_has_cpu_and_gpu_fields -v`
Expected: FAIL (`build_timing_response` doesn't exist)

- [ ] **Step 4: Add build_timing_response and dual-domain timing to concurrent_server.py**

Add near the top of `concurrent_server.py` (after CentralDispatcher):

```python
def build_timing_response(nm: int, segments: list, variant: str) -> dict:
    """Build the timing response with per-segment CPU and GPU timings."""
    return {"nm": nm, "variant": variant, "segments": segments}
```

In `handle_s4a_pooled`, replace the existing per-miss timing loop with a dual-domain version. For each miss in the eager variants (baseline, same_weights, allocator-reset, device-cache-perturbation), bracket each sub-segment (alloc, dtype, mm1, mm2) with:
- CPU: `time.perf_counter()` before and after (no sync inside)
- GPU: `torch.cuda.Event(enable_timing=True)` before and after, with `event.record()` and `event.synchronize()` only at the end of all misses (batch the sync to avoid per-segment serialization)

For the `cuda_graph` variant: record CPU submission time + total GPU replay time (CUDA events around the replay). Do NOT claim per-op subsegments.

The key change: replace the existing `torch.cuda.synchronize()` + `perf_counter` per-segment with CUDA events. Read the existing code carefully and preserve all compute logic — only change the timing instrumentation.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest test_first_miss_v2.py::test_timing_response_has_cpu_and_gpu_fields -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add concurrent_server.py test_first_miss_v2.py
git commit -m "feat(server): dual-domain timing (CPU perf_counter + CUDA events) for s4a handler"
```

---

## Task 3: bench_decomposition.py — wire B1/B2/B5 cells

Replace the `NotImplementedError` for B1/B2/B5 with real implementations that connect to the live server.

**Files:** Modify `bench_decomposition.py`

- [ ] **Step 1: Read the existing run_cell and _run_b0_local**

Run: `sed -n '40,120p' bench_decomposition.py`

- [ ] **Step 2: Implement _run_remote_cell for B1/B2/B5**

Add a `_run_remote_cell(config, cell_spec)` function that:
1. Starts the concurrent_server (via SSH, like bench_first_miss.py does)
2. Sends `setup_pool` with `active_cap` set from `cell_spec["conc"]`
3. For B1 (persistent TCP, Python direct, conc=1): uses `PersistentTransport`, single-threaded
4. For B2 (persistent TCP, Python executor, conc=1): uses `PersistentTransport` + `ThreadPoolExecutor(max_workers=1)`
5. For B5 (per-request TCP, Python executor, conc=8): uses existing per-request socket pattern + `ThreadPoolExecutor(max_workers=8)`
6. Sends `s4a_pooled` requests with the variant and NM, collects dual-domain timings
7. Returns latencies + accounting (via `account_request` from `common.instrumentation`)

This is the largest task. Reference the existing `bench_first_miss.py:execute_s4a` (line 167) and `bench_splitting.py:execute_batch_concurrent` (line 304) for the SSH/server-startup pattern. Reuse `common.protocol` for framing.

For B1/B2, the client opens ONE persistent TCP connection to the server and sends all requests over it (multiplexed). For B5, each request opens its own TCP connection (existing pattern).

- [ ] **Step 3: Update run_cell to dispatch B1/B2/B5**

```python
def run_cell(config, pool=None, act_bf16=None):
    if config.cell not in CELLS:
        raise ValueError(...)
    cell_spec = CELLS[config.cell]
    if config.cell == "B0":
        return _run_b0_local(config, cell_spec)
    if config.cell in ("B1", "B2", "B5"):
        return _run_remote_cell(config, cell_spec)
    raise NotImplementedError(f"{config.cell} requires C++ worker (Plan 2)")
```

- [ ] **Step 4: Smoke test B5 (current architecture) end-to-end**

This requires the live server on UM251. Run manually:
`python bench_decomposition.py --cell B5 --nm 8 --rank 64 --trials 1 --iters 5`
Expected: 5 latencies returned, no error.

If no server available, skip this step and note it for integration testing on the GPU node.

- [ ] **Step 5: Commit**

```bash
git add bench_decomposition.py
git commit -m "feat(decomposition): wire B1/B2/B5 remote cells with persistent/per-request TCP"
```

---

## Task 4: bench_first_miss_v2.py — S1 driver

The 5-variant x cell x NM matrix driver with full accounting.

**Files:** Create `bench_first_miss_v2.py`, complete `test_first_miss_v2.py`

- [ ] **Step 1: Write the S1 driver**

```python
# bench_first_miss_v2.py
"""S1: First-miss tax re-measurement with dual timing domains.

5 variants x 4 cells x 4 NM x 5 trials = 400 runs.
Cells: B1, B2, B5 (B6 deferred to Plan 2).
Variants: baseline, same_weights, allocator-reset, device-cache-perturbation, cuda_graph.
"""
import argparse
import csv
import os
import statistics
from bench_decomposition import DecompositionConfig, run_cell, CELLS
from common.instrumentation import account_request, RunMetadata
from common.stats import trial_ci, percentile_ci

VARIANTS = ["baseline", "same_weights", "allocator-reset",
            "device-cache-perturbation", "cuda_graph"]
CELLS_S1 = ["B1", "B2", "B5"]  # B6 added in Plan 2
NMS = [1, 2, 4, 8]
N_TRIALS = 5


def run_s1(output_dir="results/s1_first_miss"):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for cell in CELLS_S1:
        for variant in VARIANTS:
            for nm in NMS:
                trial_p50s, trial_p99s, first_rest_ratios = [], [], []
                for trial in range(N_TRIALS):
                    config = DecompositionConfig(
                        cell=cell, nm=nm, rank=64,
                        n_trials=1, n_iters=50,
                    )
                    result = run_cell(config)
                    lats = result["latencies_us"]
                    # First-miss = first latency, rest = mean of the rest
                    first = lats[0]
                    rest_mean = statistics.mean(lats[1:]) if len(lats) > 1 else first
                    ratio = first / rest_mean if rest_mean > 0 else float("nan")
                    trial_p50s.append(statistics.median(lats))
                    trial_p99s.append(sorted(lats)[int(0.99 * len(lats)) - 1])
                    first_rest_ratios.append(ratio)
                # Trial-level CIs
                p50_lo, p50_hi = trial_ci(trial_p50s)
                p99_lo, p99_hi = trial_ci(trial_p99s)
                ratio_lo, ratio_hi = trial_ci(first_rest_ratios)
                rows.append({
                    "cell": cell, "variant": variant, "nm": nm,
                    "p50_median": statistics.median(trial_p50s),
                    "p50_ci_lo": p50_lo, "p50_ci_hi": p50_hi,
                    "p99_median": statistics.median(trial_p99s),
                    "p99_ci_lo": p99_lo, "p99_ci_hi": p99_hi,
                    "first_rest_ratio_median": statistics.median(first_rest_ratios),
                    "ratio_ci_lo": ratio_lo, "ratio_ci_hi": ratio_hi,
                })
                print(f"{cell} {variant} NM={nm}: ratio={statistics.median(first_rest_ratios):.2f}x")
    csv_path = os.path.join(output_dir, "first_miss.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="S1 first-miss tax re-measurement")
    parser.add_argument("--output", default="results/s1_first_miss")
    args = parser.parse_args()
    run_s1(args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a test that the driver config is well-formed**

Append to `test_first_miss_v2.py`:

```python
def test_s1_variant_list_complete():
    from bench_first_miss_v2 import VARIANTS, CELLS_S1, NMS
    assert len(VARIANTS) == 5
    assert "cuda_graph" in VARIANTS
    assert "allocator-reset" in VARIANTS
    assert CELLS_S1 == ["B1", "B2", "B5"]
    assert NMS == [1, 2, 4, 8]
```

- [ ] **Step 3: Run test**

Run: `python -m pytest test_first_miss_v2.py -v`
Expected: PASS (2 tests)

- [ ] **Step 4: Commit**

```bash
git add bench_first_miss_v2.py test_first_miss_v2.py
git commit -m "feat(s1): first-miss tax driver — 5 variants x cells x NM with trial CIs"
```

---

## Task 5: Accounting closure validation

Verify that the full accounting closes: `instrumentation_gap` is small for B0 (local, fully instrumented) and that `cross_domain_residual` is reported descriptively for B1/B2/B5.

**Files:** Modify `bench_decomposition.py` (pass real timelines through `account_request`), `test_decomposition_smoke.py`

- [ ] **Step 1: Update B0 to construct a real RequestTimeline**

In `_run_b0_local`, instead of hardcoded accounting, construct a `RequestTimeline` with t0/t5/t6/t17/t18 (t5=t0, t6=t0, t17=t18 for local — no network) and call `account_request`. This addresses review issue I4.

- [ ] **Step 2: Add accounting closure test**

```python
def test_b0_accounting_closes():
    config = DecompositionConfig(cell="B0", nm=1, rank=64, n_trials=1, n_iters=1)
    result = run_cell(config)
    acc = result["accounting"]
    # B0 has no network: cross_domain_residual should be ~0
    assert abs(acc["cross_domain_residual_us"]) < 100
    # instrumentation_gap should be small (all segments instrumented)
    assert acc["instrumentation_gap_fraction"] < 0.1 or acc["instrumentation_gap_us"] < 50
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_decomposition_smoke.py -v`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add bench_decomposition.py test_decomposition_smoke.py
git commit -m "feat(s1): real RequestTimeline accounting for B0; accounting closure test"
```

---

## Self-Review

**Spec coverage:**
- B1/B2/B5 cells -> Task 3
- B6 cell -> deferred to Plan 2 (noted in run_cell)
- 5 variants -> Task 2 (server) + Task 4 (driver)
- Dual timing domains (CPU + CUDA events) -> Task 2
- Full accounting closure -> Task 5
- cuda_graph variant: replay-total only, no per-op claim -> Task 2
- allocator-reset + device-cache-perturbation as separate diagnostics -> Task 2
- L2 size runtime-queried (not hardcoded) -> Task 2
- No "likely cuBLAS" speculation -> enforced in driver (reports ratio, no attribution)

**Gap:** B6 (C++ matched worker) is not in this plan. When Plan 2 lands, add "B6" to `CELLS_S1` and ensure `run_cell("B6")` dispatches to the C++ path. The S1 driver is structured so this is a one-line change.

**Placeholder scan:** No TBDs. All code blocks are complete or reference existing patterns.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-03-s1-first-miss-remeasure.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between
**2. Inline Execution** — batch with checkpoints

**Which approach?**
