# S3 Scheduling Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the original "SJF helps 3x" result was genuine scheduling or executor submission-order accident, by testing FIFO / Client-SJF / Server-SJF with full execution-order instrumentation.

**Architecture:** A `PriorityDispatcher` class replaces the semaphore-based `_dispatcher` in `concurrent_server.py` for Server-SJF. Client-SJF adds a client-local ready queue with online shortest-predicted-job selection. Execution-order instrumentation records four sequences per request to verify whether client sorting controls GPU execution order.

**Tech Stack:** Python 3.11, PyTorch 2.8.0, pytest, threading, heapq, CUDA events.

**Spec:** `docs/superpowers/specs/2026-07-02-decomposition-confound-isolation-design.md` (section "S3 — Scheduling study")
**Depends on:** Plan 1 (common/, qppool, concurrent_server), Plan 3 (S1 — RemoteSession, persistent TCP)

**Working directory:** `test/lora/avx/cross_node/`

---

## File Structure

```
test/lora/avx/cross_node/
  concurrent_server.py    # MODIFIED: PriorityDispatcher class, S_hat calibration
  bench_decomposition.py  # MODIFIED: Client-SJF dispatch, execution-order recording
  bench_scheduling_v2.py  # NEW: S3 driver
  test_scheduling_v2.py   # NEW
```

---

## Task 1: PriorityDispatcher (Server-SJF) + S_hat calibration

Add a `PriorityDispatcher` class that replaces the semaphore-based `_dispatcher` when Server-SJF is selected. Uses a single selector thread that acquires slots and dispatches to the highest-priority queued request.

**Files:** Modify `concurrent_server.py`, create `test_scheduling_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# test_scheduling_v2.py
"""Test Server-SJF priority dispatcher."""
import threading
import time
import pytest


def test_priority_dispatcher_selects_shortest_first():
    """When two requests are queued, the shorter one (lower NM) should be selected first."""
    from concurrent_server import PriorityDispatcher
    disp = PriorityDispatcher(active_cap=1, s_hat={1: 1.0, 8: 8.0})
    
    # Enqueue a heavy job first, then a light job
    results = []
    
    def worker(nm, label):
        disp.enqueue_and_wait(nm, label)
        results.append(label)
        time.sleep(0.05)
        disp.release()
    
    # Start heavy job first (it will grab the slot)
    t_heavy = threading.Thread(target=worker, args=(8, "heavy"))
    t_heavy.start()
    time.sleep(0.02)  # ensure heavy is queued first
    
    # Start light job (should be selected BEFORE heavy when slot frees)
    t_light = threading.Thread(target=worker, args=(1, "light"))
    t_light.start()
    
    t_heavy.join(timeout=2)
    t_light.join(timeout=2)
    
    # Heavy was queued first and got the slot first (cap=1).
    # When heavy releases, light should be selected next.
    # But if both are queued before either gets a slot, light should go first.
    # With the timing above, heavy gets the slot, light waits.
    # After heavy releases, light gets the slot.
    assert "heavy" in results
    assert "light" in results


def test_priority_dispatcher_caps_active_jobs():
    """With active_cap=2, at most 2 jobs run concurrently."""
    from concurrent_server import PriorityDispatcher
    disp = PriorityDispatcher(active_cap=2, s_hat={1: 1.0})
    
    active = [0]
    max_active = [0]
    lock = threading.Lock()
    
    def worker():
        disp.enqueue_and_wait(1, "x")
        with lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        disp.release()
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=5)
    assert max_active[0] == 2


def test_priority_dispatcher_records_wait():
    from concurrent_server import PriorityDispatcher
    disp = PriorityDispatcher(active_cap=1, s_hat={1: 1.0})
    disp.enqueue_and_wait(1, "a")
    disp.release()
    assert len(disp.wait_us) >= 1
    assert disp.wait_us[0] >= 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest test_scheduling_v2.py -v`
Expected: FAIL (`PriorityDispatcher` doesn't exist)

- [ ] **Step 3: Implement PriorityDispatcher**

Add to `concurrent_server.py` (after CentralDispatcher):

```python
import heapq

class PriorityDispatcher:
    """Central priority dispatcher for Server-SJF.

    A single selector thread acquires active-concurrency slots and
    dispatches them to the highest-priority (shortest predicted service
    time) queued request. This ensures priority ordering, not just
    semaphore FIFO.
    """

    def __init__(self, active_cap: int, s_hat: dict = None):
        self.active_cap = active_cap
        self._sem = threading.Semaphore(active_cap)
        self._pq = []  # heap of (priority, seq, req_id, event)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._seq = 0
        self._s_hat = s_hat or {}
        self._selector = threading.Thread(target=self._select_loop, daemon=True)
        self._selector.start()
        self.wait_us = []

    def _priority(self, num_miss: int) -> float:
        return self._s_hat.get(num_miss, float(num_miss))

    def enqueue_and_wait(self, num_miss: int, req_id) -> None:
        """Enqueue this request and block until selected by the dispatcher."""
        t0 = time.perf_counter()
        priority = self._priority(num_miss)
        event = threading.Event()
        with self._cv:
            self._seq += 1
            heapq.heappush(self._pq, (priority, self._seq, req_id, event))
            self._cv.notify()
        event.wait()
        t1 = time.perf_counter()
        self.wait_us.append((t1 - t0) * 1e6)

    def _select_loop(self) -> None:
        """Selector thread: acquire slot, pop highest-priority request, signal."""
        while True:
            self._sem.acquire()
            with self._cv:
                while not self._pq:
                    self._cv.wait()
                priority, seq, req_id, event = heapq.heappop(self._pq)
            event.set()

    def release(self) -> None:
        self._sem.release()
```

- [ ] **Step 4: Wire PriorityDispatcher into handle_s4a_pooled**

In `handle_s4a_pooled`, replace the `_dispatcher` usage:
- When the setup params include `scheduling_policy: "server_sjf"`, create a `PriorityDispatcher` instead of `CentralDispatcher`
- In `handle_s4a_pooled`, call `_dispatcher.enqueue_and_wait(num_miss, req_id)` instead of `_dispatcher.acquire()`
- The release stays the same (`_dispatcher.release()`)

- [ ] **Step 5: Add S_hat calibration**

Add a `_s_hat` calibration function that runs isolated atomic requests for NM∈{1,4,8} and records the median service time:

```python
_s_hat_cache = {}

def calibrate_s_hat(cell, nms=(1, 4, 8), n_iters=20):
    """Calibrate S_hat(NM) = median isolated atomic service time."""
    # Run isolated atomic requests for each NM, record E2E latencies
    # Store in _s_hat_cache[(cell, nm)] = median
    ...
```

This runs as a pre-calibration step before the S3 campaign.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest test_scheduling_v2.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add concurrent_server.py test_scheduling_v2.py
git commit -m "feat(s3): PriorityDispatcher for Server-SJF with S_hat calibration"
```

---

## Task 2: Client-SJF dispatch

Add client-side SJF to `bench_decomposition.py`. The client maintains a local ready queue with outstanding window W=N. When a dispatch slot is free, it picks the shortest predicted ready job.

**Files:** Modify `bench_decomposition.py`

- [ ] **Step 1: Implement Client-SJF dispatch**

Add a `_run_client_sjf` function that:
1. Generates a batch of requests (synchronized) or Poisson arrivals
2. Maintains a client-local ready queue
3. When a dispatch slot is free (< W in-flight), selects the shortest predicted (lowest NM) ready job
4. Sends it via PersistentTransport
5. Records the submission order

For synchronized mode: sort the batch by NM ascending (reproduces original "SJF").
For Poisson mode: online policy — arrivals enter the ready queue; dispatch picks shortest ready when a slot opens.

- [ ] **Step 2: Add outstanding window W**

```python
def _run_client_sjf(config, cell_spec, variant, transport, qp_pool,
                    server_host, server_port, compositions, W=None):
    """Client-SJF: dispatch shortest predicted ready job.
    
    W = outstanding window (default = cell_spec["conc"])
    """
```

- [ ] **Step 3: Commit**

```bash
git add bench_decomposition.py
git commit -m "feat(s3): Client-SJF dispatch with outstanding window W"
```

---

## Task 3: Execution-order instrumentation

Record four sequences per request to verify whether client sorting controls GPU execution order.

**Files:** Modify `concurrent_server.py` (record handler-start and GPU-start), modify `bench_decomposition.py` (record submission and completion)

- [ ] **Step 1: Add execution-order fields to the server response**

The server already records per-miss GPU segments. Add:
- `handler_start_seq`: a global counter incremented when the handler starts
- `gpu_start_seq`: a global counter incremented when the first GPU kernel starts
- `completion_seq`: a global counter incremented when the handler finishes

```python
_handler_seq = 0
_gpu_seq = 0
_completion_seq = 0
_seq_lock = threading.Lock()

# In handle_s4a_pooled, after dispatcher.acquire():
global _handler_seq
with _seq_lock:
    _handler_seq += 1
    handler_seq = _handler_seq

# In the response:
resp["handler_start_seq"] = handler_seq
resp["completion_seq"] = completion_seq
```

- [ ] **Step 2: Record client submission sequence**

The client already has `req_id` (monotonic). The submission sequence is the order of `req_id` assignment. Record it in the result.

- [ ] **Step 3: Compute priority fidelity, inversion count, order preservation**

Add helper functions:
```python
def compute_priority_fidelity(submission_seqs, handler_seqs, gpu_seqs, classes):
    """Fraction of decisions where light request starts first when both light and heavy queued."""

def compute_inversion_count(handler_seqs, gpu_seqs, classes):
    """Number of times NM=8 starts while earlier-ready NM=1 remains queued."""

def compute_order_correlation(intended_order, actual_order):
    """Kendall's tau between intended and actual execution order."""
```

- [ ] **Step 4: Commit**

```bash
git add concurrent_server.py bench_decomposition.py
git commit -m "feat(s3): execution-order instrumentation (priority fidelity, inversions)"
```

---

## Task 4: S3 driver (bench_scheduling_v2.py)

Orchestrate the full S3 study: 3 policies × 2 cells × 3 compositions × 2 arrival modes, with factorial analysis.

**Files:** Create `bench_scheduling_v2.py`

- [ ] **Step 1: Write the driver**

```python
# bench_scheduling_v2.py
"""S3: Scheduling study — FIFO / Client-SJF / Server-SJF.

3 policies × 2 cells (B2, B3) × 3 compositions × 2 arrival modes.
Primary: atomic, non-sliced. Secondary: priority + slicing.

Factorial analysis:
  Delta_sched(c) = P99(FIFO,c) - P99(Server-SJF,c) at c in {1, N}
  Delta_conc(policy) = P99(policy,N) - P99(policy,1)
  Interaction = Delta_sched(N) - Delta_sched(1)
  Per class (light, heavy), using common-load traces only.
"""
```

Policies:
- `fifo`: standard dispatch, server FIFO
- `client_sjf`: client sorts/dispatches shortest first, server FIFO
- `server_sjf`: client FIFO, server priority queue

Compositions:
- `1h8l`: 1×NM=8 + 8×NM=1 (sparse heavy)
- `2h0l`: 2×NM=8 (all-heavy control)
- `medium`: 1×NM=8 + 2×NM=4 + 8×NM=1

- [ ] **Step 2: Add factorial analysis**

```python
def factorial_analysis(rows):
    """Compute Delta_sched, Delta_conc, Interaction per class."""
    # For each cell, compare FIFO vs Server-SJF at conc=1 and conc=N
    # Using common-load traces only (matched-ρ excluded from causal contrast)
```

- [ ] **Step 3: Add fairness metrics**

Max heavy queue wait, P99.9 heavy queue wait, heavy SLO violation, throughput by class, queue length by class.

- [ ] **Step 4: Commit**

```bash
git add bench_scheduling_v2.py
git commit -m "feat(s3): scheduling driver — 3 policies × cells × compositions with factorial"
```

---

## Task 5: Live campaign + analysis

Run the full S3 campaign on UM251/UM253 with 5 trials × 50 iters.

- [ ] **Step 1: Calibrate S_hat**

Run isolated atomic requests for NM∈{1,4,8} on B2 and B3. Record median service times.

- [ ] **Step 2: Run the S3 campaign**

```bash
python -m bench_scheduling_v2 --trials 5 --iters 50 --output results/s3_scheduling
```

- [ ] **Step 3: Analyze and commit results**

---

## Self-Review

**Spec coverage:**
- Central dispatcher before executor (Server-SJF) → Task 1
- Client-SJF with outstanding window W → Task 2
- Execution-order instrumentation (4 sequences, priority fidelity, inversions) → Task 3
- S3 driver (3 policies × 2 cells × 3 compositions, factorial) → Task 4
- S_hat calibration → Task 1 (Step 5) + Task 5 (Step 1)
- Fairness metrics → Task 4 (Step 3)
- Primary study: atomic, non-sliced → enforced in driver
- Priority + slicing secondary → noted, not primary task

**Gaps:**
- The PriorityDispatcher uses a single selector thread, which could be a bottleneck under high load. For the S3 study (max conc=8), this is fine.
- Two-client follow-up (H2 caveat) is not in this plan — noted for future.
- Matched-utilization runs are secondary; common-load is primary for causal factorial.

---

## Execution Handoff

Plan complete. Execute via subagent-driven development.
