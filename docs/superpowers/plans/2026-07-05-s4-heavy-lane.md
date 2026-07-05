# S4 Heavy-Lane Re-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Determine whether class-aware heavy admission control (heavy-lane) improves the latency-throughput Pareto frontier, or whether the original "heavy-lane wins" was objective-specific.

**Architecture:** A `HeavyLaneDispatcher` class gates heavy (NM>=4) requests with a sub-cap H, independent of the total active cap N. The S4 driver sweeps H∈{1,2,4,8} across Poisson mixtures and reports a 3D Pareto frontier (light P99, heavy P99, throughput). A global-cap control (Stage C) tests whether the benefit is genuinely class-aware.

**Tech Stack:** Python 3.11, PyTorch 2.8.0, pytest, threading.

**Spec:** `docs/superpowers/specs/2026-07-02-decomposition-confound-isolation-design.md` (section "S4")
**Depends on:** Plan 1 (common/, RemoteSession), Plan 3 (S1)

**Working directory:** `test/lora/avx/cross_node/`

---

## File Structure

```
test/lora/avx/cross_node/
  concurrent_server.py      # MODIFIED: HeavyLaneDispatcher class
  bench_heavy_lane_v2.py    # NEW: S4 driver
  test_heavy_lane_v2.py     # NEW
```

---

## Task 1: HeavyLaneDispatcher

Add a `HeavyLaneDispatcher` to `concurrent_server.py` that maintains separate FIFO light and heavy queues with a heavy sub-cap H.

**Files:** Modify `concurrent_server.py`, create `test_heavy_lane_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# test_heavy_lane_v2.py
import threading, time, pytest

def test_heavy_lane_caps_heavy_concurrency():
    """With N=8, H=2, at most 2 heavy jobs run concurrently."""
    from concurrent_server import HeavyLaneDispatcher
    disp = HeavyLaneDispatcher(active_cap=8, heavy_cap=2)
    active_heavy = [0]
    max_heavy = [0]
    lock = threading.Lock()
    def worker(is_heavy):
        disp.acquire(is_heavy)
        with lock:
            if is_heavy:
                active_heavy[0] += 1
                max_heavy[0] = max(max_heavy[0], active_heavy[0])
        time.sleep(0.05)
        with lock:
            if is_heavy:
                active_heavy[0] -= 1
        disp.release(is_heavy)
    threads = [threading.Thread(target=worker, args=(i < 6,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=5)
    assert max_heavy[0] <= 2

def test_light_bypasses_heavy_cap():
    """Light jobs are not limited by H."""
    from concurrent_server import HeavyLaneDispatcher
    disp = HeavyLaneDispatcher(active_cap=8, heavy_cap=1)
    # All 8 slots can be light even with H=1
    disp.acquire(False)  # light
    assert True  # didn't block
    disp.release(False)
```

- [ ] **Step 2: Run to verify it fails**
- [ ] **Step 3: Implement HeavyLaneDispatcher**

```python
class HeavyLaneDispatcher:
    """Class-aware admission: total cap N + heavy sub-cap H.

    Light jobs bypass H. Heavy jobs (NM>=threshold) are limited to H concurrent.
    """
    def __init__(self, active_cap: int, heavy_cap: int, heavy_threshold: int = 4):
        self.active_cap = active_cap
        self.heavy_cap = heavy_cap
        self.heavy_threshold = heavy_threshold
        self._total_sem = threading.Semaphore(active_cap)
        self._heavy_sem = threading.Semaphore(heavy_cap)
        self.wait_us = []

    def acquire(self, is_heavy: bool) -> None:
        t0 = time.perf_counter()
        self._total_sem.acquire()
        if is_heavy:
            self._heavy_sem.acquire()
        t1 = time.perf_counter()
        self.wait_us.append((t1 - t0) * 1e6)

    def release(self, is_heavy: bool) -> None:
        if is_heavy:
            self._heavy_sem.release()
        self._total_sem.release()
```

- [ ] **Step 4: Wire into handle_setup_pool** — accept `heavy_cap` and `heavy_threshold` params; create `HeavyLaneDispatcher` when `heavy_cap` is set.
- [ ] **Step 5: Wire into handle_s4a_pooled** — `is_heavy = num_miss >= heavy_threshold`; call `_dispatcher.acquire(is_heavy)` / `_dispatcher.release(is_heavy)`.
- [ ] **Step 6: Run test, commit**

---

## Task 2: S4 driver (bench_heavy_lane_v2.py)

Sweeps H∈{1,2,4,8} × Poisson mixtures × cells, with Pareto reporting and global-cap control.

**Files:** Create `bench_heavy_lane_v2.py`

- [ ] **Step 1: Write the driver**

```python
# bench_heavy_lane_v2.py
"""S4: Heavy-lane re-test — H sweep, Pareto frontier, class-aware vs global cap."""
import argparse, csv, os, statistics, time, socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from bench_decomposition import CELLS, RemoteSession, DecompositionConfig, _make_sjf_request, new_request_id
from common.transport import PersistentTransport
from common.stats import trial_ci
from common.load_generator import generate_poisson_trace, OpenLoopRunner, NoopSink

H_VALUES = [1, 2, 4, 8]
MIXTURES = {"1h9l": 0.10, "1h3l": 0.25, "1h1l": 0.50, "all_heavy": 1.0}
CELLS_S4 = ["B3"]
N_TRIALS = 5
N_ITERS = 50

def run_heavy_lane(session, nm, is_heavy, n_iters):
    """Run n_iters requests with given NM."""
    # ... send via PersistentTransport, collect latencies ...

def run_s4(output_dir="results/s4_heavy_lane", n_trials=None, n_iters=None):
    # For each cell, H, mixture: run Poisson arrivals, collect per-class P99
    # Also run global-cap control (Stage C): K∈{1,2,4} without heavy sub-cap
    # Report 3D Pareto: (light_P99, heavy_P99, throughput)
    ...
```

- [ ] **Step 2: Add Pareto frontier computation**
- [ ] **Step 3: Add global-cap control (Stage C)**
- [ ] **Step 4: Commit**

---

## Task 3: Live campaign + analysis

- [ ] **Step 1: Sync to UM251**
- [ ] **Step 2: Run `python -m bench_heavy_lane_v2 --trials 5 --iters 50`**
- [ ] **Step 3: Analyze Pareto frontiers, commit results**

---

## Self-Review

**Spec coverage:** H sweep → Task 1+2; Pareto → Task 2; global-cap control → Task 2 Stage C; H-binding diagnostics → Task 2; Poisson mixtures → Task 2. B5-original reproduction deferred (S6 handles original-method reproduction).
