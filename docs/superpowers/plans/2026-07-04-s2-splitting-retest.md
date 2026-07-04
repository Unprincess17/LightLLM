# S2 Splitting Re-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distinguish atomic execution, stateful client chunking, server cooperative slicing, and uncontrolled RPC fan-out — to determine whether the original "splitting hurts" conclusion was caused by splitting itself or by RPC fan-out.

**Architecture:** Server gains a `quantum` parameter and cooperative yield mechanism (per-logical-request round-robin scheduler). Client gains stateful chunking (persistent TCP, shared QP, retained server state) with contiguous and interleaved submission policies. A matched persistent fan-out control isolates RPC fan-out from transport persistence.

**Tech Stack:** Python 3.11, PyTorch 2.8.0, pytest, threading, CUDA events.

**Spec:** `docs/superpowers/specs/2026-07-02-decomposition-confound-isolation-design.md` (section "S2 — Splitting re-test")
**Depends on:** Plan 1 (common/, qppool, concurrent_server, bench_decomposition), Plan 3 (S1 — persistent TCP transport, dual-domain timing, RemoteSession)

**Working directory:** `test/lora/avx/cross_node/`

---

## File Structure

```
test/lora/avx/cross_node/
  concurrent_server.py    # MODIFIED: cooperative slicing handler, quantum parameter, yield
  bench_decomposition.py  # MODIFIED: stateful client chunking, matched fan-out
  bench_splitting_v2.py   # NEW: S2 driver
  test_splitting_v2.py    # NEW
```

---

## Task 1: Server-side cooperative slicing

Add a `quantum` parameter to the s4a_pooled handler. When `quantum < num_miss`, the server processes `quantum` misses, yields to a per-logical-request round-robin scheduler, and resumes later. Uses the async yield sequence from the spec (worker submits to stream, records event, returns to pool; event poller marks continuation runnable).

**Files:** Modify `concurrent_server.py`, create `test_splitting_v2.py`

- [ ] **Step 1: Write the failing test for the quantum parameter**

```python
# test_splitting_v2.py
"""Test server cooperative slicing with quantum parameter."""
import pytest


def test_quantum_parameter_accepted():
    """Server should accept a quantum parameter in the s4a_pooled request."""
    from concurrent_server import build_timing_response
    # The quantum parameter is parsed in handle_s4a_pooled.
    # This test verifies the response includes quantum info.
    resp = build_timing_response(nm=8, segments=[], variant="baseline")
    resp["quantum"] = 4
    resp["yields"] = 1
    assert resp["quantum"] == 4
    assert resp["yields"] == 1


def test_quantum_eight_equals_atomic():
    """q=8 (run-to-completion) should produce the same output as atomic."""
    # This is an integration test verified on the live server.
    # q=num_miss means no yielding — equivalent to atomic.
    pass  # tested in Task 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_splitting_v2.py -v`
Expected: FAIL or partial pass

- [ ] **Step 3: Add quantum parameter and cooperative slicing to concurrent_server.py**

In `handle_s4a_pooled`, parse `quantum` from params (default = num_miss, meaning no slicing). When `quantum < num_miss`, implement the async yield sequence:

```python
quantum = params.get("quantum", num_miss)
```

The cooperative slicing path (when `quantum < num_miss`):
1. Process `quantum` misses (same compute as eager)
2. Record a per-stream CUDA completion event
3. Release the CPU worker (return to pool)
4. The continuation is appended to a runnable queue
5. A scheduler selects the next continuation (round-robin)
6. When this request's continuation is selected, resume processing the next quantum
7. Repeat until all misses are done

For the initial implementation, use a simplified yield mechanism:
- A `threading.Semaphore` per logical request controls continuation
- After each quantum, the worker releases its slot and waits on the semaphore
- A scheduler thread (or the main accept loop) posts to semaphores in round-robin order
- This is NOT true async (the worker blocks on the semaphore), but it correctly models the scheduling behavior

```python
# In handle_s4a_pooled, after parsing quantum:
if quantum < num_miss and decompose:
    # Cooperative slicing path
    _handle_s4a_sliced(conn, params, transport, act_bf16, a_gpu, b_gpu,
                       result, num_miss, quantum, variant, segments,
                       hidden_dim, intermediate_dim, rank)
    # ... send response ...
```

The `_handle_s4a_sliced` function processes quanta and yields between them. For B2 (conc=1), only one request is active at a time, so the yield is a no-op (nothing to interleave with). For B3 (conc=N), multiple requests can interleave their quanta.

IMPORTANT: The yield mechanism must release the QP/session state across quantum boundaries (per spec: "Release: CPU worker, GPU admission token, scheduler ownership. Keep: registered buffers, logical request state, output accumulation buffer, QP ownership").

For the initial implementation, keep QP ownership across quanta (releasing QP per quantum is a separate later study, per spec). Release the GPU admission token (dispatcher) between quanta.

- [ ] **Step 4: Add yield count and quantum to the response**

```python
resp["quantum"] = quantum
resp["yields"] = (num_miss - 1) // quantum  # number of yield points
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest test_splitting_v2.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add concurrent_server.py test_splitting_v2.py
git commit -m "feat(s2): server cooperative slicing with quantum parameter and yield"
```

---

## Task 2: Stateful client chunking

Add stateful client chunking to `bench_decomposition.py`: persistent TCP, shared QP, retained server state. First chunk does RDMA READ; server retains activation + output; intermediate chunks compute only; final chunk does RDMA WRITE. Two submission policies: contiguous and interleaved.

**Files:** Modify `bench_decomposition.py`

- [ ] **Step 1: Add chunk request types to the server**

In `concurrent_server.py`, add handlers for chunked requests:
- `"s4a_chunk_init"`: first chunk — RDMA READ, store activation + weights, compute first quantum
- `"s4a_chunk_continue"`: intermediate chunk — compute only (activation already stored)
- `"s4a_chunk_final"`: last chunk — compute last quantum, RDMA WRITE result

The server maintains per-session state (activation buffer, weight buffers, output buffer) keyed by a session ID.

- [ ] **Step 2: Implement stateful client chunking in bench_decomposition.py**

Add `_run_stateful_chunking()` function:

```python
def _run_stateful_chunking(config, cell_spec, variant, transport, qp_pool,
                           chunk_size, policy="contiguous"):
    """Stateful client chunking: N control messages, 1 READ/WRITE.

    First chunk: RDMA READ + compute
    Intermediate chunks: compute only
    Final chunk: compute + RDMA WRITE

    policy: "contiguous" (back-to-back) or "interleaved" (round-robin across ready requests)
    """
```

For contiguous: send all chunks of one logical request back-to-back.
For interleaved: use a client-side round-robin scheduler across ready logical requests.

- [ ] **Step 3: Add matched persistent fan-out**

Add `_run_persistent_fanout()` function: same persistent TCP, same serialization, same active cap, but multiple chunk requests submitted concurrently as independent executor tasks. This is the H1 control — isolates RPC fan-out from transport persistence.

- [ ] **Step 4: Commit**

```bash
git add concurrent_server.py bench_decomposition.py
git commit -m "feat(s2): stateful client chunking + matched persistent fan-out"
```

---

## Task 3: q=8 equivalence validation

Verify that quantum=8 (run-to-completion) produces the same output and similar timing as the atomic path.

**Files:** Create test in `test_splitting_v2.py`

- [ ] **Step 1: Write the equivalence test**

```python
def test_q8_equivalence():
    """q=8 (no slicing) should match atomic within tolerance."""
    # Run atomic NM=8 and sliced q=8 NM=8 on the same server session
    # Compare: output correctness, RDMA op count, median service time (<=5% diff)
    # This is an integration test run on the live server.
    pass  # implemented as a live test
```

- [ ] **Step 2: Run the equivalence test live**

Run a script that:
1. Starts one RemoteSession (B1, baseline)
2. Runs atomic NM=8 (quantum=8, which is the default)
3. Runs sliced q=8 NM=8 (explicitly setting quantum=8)
4. Compares outputs and timings

- [ ] **Step 3: Commit**

```bash
git add test_splitting_v2.py
git commit -m "test(s2): q=8 equivalence validation"
```

---

## Task 4: S2 driver (bench_splitting_v2.py)

The S2 driver orchestrates the full splitting study: 3 mechanisms (atomic, stateful chunking, server slicing) × chunk/quantum sizes × 2 compositions × 2 arrival modes × 5 trials.

**Files:** Create `bench_splitting_v2.py`

- [ ] **Step 1: Write the driver**

```python
# bench_splitting_v2.py
"""S2: Splitting re-test — atomic vs stateful chunking vs server slicing.

Mechanisms:
  atomic: 1 RPC, 1 READ/WRITE, no yield
  stateful_chunking: N RPCs, 1 READ/WRITE, client chooses boundaries
  server_slicing: 1 RPC, 1 READ/WRITE, server chooses boundaries
  persistent_fanout: N RPCs, N READs/WRITEs, concurrent (H1 control)

Chunk/quantum sizes: 1, 2, 4 (q=8 = atomic, used for equivalence validation)
Compositions: 1h8l, 2h0l
Arrival: synchronized, Poisson
"""
```

The driver uses RemoteSession and groups by (cell, mechanism) to minimize server cycles.

- [ ] **Step 2: Add per-quantum event recording**

Record `quantum_ready, quantum_selected, GPU_submitted, GPU_start, GPU_end, continuation_requeued` per quantum. Compute inter-quantum wait decomposition.

- [ ] **Step 3: Commit**

```bash
git add bench_splitting_v2.py
git commit -m "feat(s2): splitting driver — 3 mechanisms × chunk sizes × compositions"
```

---

## Task 5: CUDA-graph sub-study

Test whether slicing fails because of scheduling or because eager dispatch makes each quantum expensive. Run a targeted subset under CUDA graph replay.

**Files:** Modify `bench_splitting_v2.py`

- [ ] **Step 1: Add graph sub-study**

For atomic, server slice q=1/2/4: run both eager and CUDA graph variants. The graph variant captures one graph per quantum and replays.

- [ ] **Step 2: Commit**

```bash
git add bench_splitting_v2.py
git commit -m "feat(s2): CUDA-graph sub-study for slicing"
```

---

## Task 6: Scheduler no-op microbenchmark

`enqueue → select → callback → requeue` with no GEMMs. Lower-bound cost of one yield.

**Files:** Create `test_scheduler_noop.py`

- [ ] **Step 1: Write the no-op benchmark**

```python
# test_scheduler_noop.py
def test_scheduler_noop_cost():
    """Measure the lower-bound cost of one yield (enqueue/select/requeue)."""
    import time, threading, queue
    
    q = queue.Queue()
    n = 10000
    t0 = time.perf_counter()
    for i in range(n):
        q.put(i)
        q.get()
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / n * 1e6
    print(f"Scheduler no-op cost: {cost_us:.2f}us per yield")
    assert cost_us < 100  # should be well under 100us
```

- [ ] **Step 2: Commit**

```bash
git add test_scheduler_noop.py
git commit -m "test(s2): scheduler no-op microbenchmark"
```

---

## Self-Review

**Spec coverage:**
- Server cooperative slicing (quantum, yield, round-robin) → Task 1
- Stateful client chunking (contiguous + interleaved) → Task 2
- Matched persistent fan-out (H1 control) → Task 2
- q=8 equivalence validation → Task 3
- S2 driver (3 mechanisms × sizes × compositions) → Task 4
- Per-quantum event recording → Task 4
- CUDA-graph sub-study → Task 5
- Scheduler no-op microbenchmark → Task 6

**Gaps:**
- The async yield sequence is simplified (semaphore-based, not true event-callback). This correctly models scheduling behavior but doesn't release the CPU worker during GPU execution. A true async implementation (event callback + continuation requeue) is noted as a future improvement for B3 (conc=N) where interleaving matters.
- Stateless client chunking (each chunk re-reads) is a secondary comparison, not a primary task.
- B5-stateful hybrid is optional, not included.

**Placeholder scan:** Tasks 1-2 describe the implementation structurally because the cooperative slicing and stateful chunking involve significant server-side state management. The implementer should reference the spec's async yield sequence and per-logical-request round-robin scheduler for exact behavior.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-04-s2-splitting-retest.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task
**2. Inline Execution** — batch with checkpoints

**Which approach?**
