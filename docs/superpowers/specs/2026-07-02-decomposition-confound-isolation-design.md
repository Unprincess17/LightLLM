# Control-Plane Decomposition & Confound Isolation — Design Spec

## Status

Implementation specification approved; execution remains decision-gated.
Supersedes the experimental program in
`2026-07-01-nm-weight-followup-program-design.md` for the six confound families
S1-S6. Experiments 2, 7, 8, 10 from that document remain separately scoped.

## Context

The NM-weight follow-up summary (`docs/nm-weight-followup-summary.html`,
2026-07-02) drew six conclusions that an external review found stronger than the
evidence supports. Code inspection of `bench_splitting.py`, `bench_capacity.py`,
`bench_first_miss.py`, and `concurrent_server.py` confirmed the flagged confounds
are real:

- `bench_splitting.py` makes split-1/2/4 multiply TCP sockets, QP acquisitions,
  executor workers, and CUDA launches simultaneously. The result rejects
  *client-side RPC fan-out*, not splitting as a systems concept.
- `bench_capacity.py` always uses 256 executor workers; `pool.borrow()` is the
  true gate. "Pool=32 worse" is an admission-control effect, not a physical-QP
  effect. Service-time measurement excludes TCP connect+send, so rho is
  underestimated. Inflight cap 256 + `pass` backpressure = closed-loop.
- `bench_first_miss.py` `cuda_graph` variant times only `miss_i_alloc` (graph
  replay) and hardcodes dtype/mm1/mm2 = 0.0; sync placement differs across
  variants (1x/miss vs 4x/miss).
- `concurrent_server.py` opens a new TCP connection per request (`SO_LINGER`
  RST). Cannot distinguish Python from TCP from executor from GIL.
- The summary's Poisson table reports heavy-lane light P99 going 20,218->39,983 us
  (~=2x worse) while the text claims "improves 2x (20ms->16ms)." Unresolved
  inconsistency.

## Goal

A single research program that isolates the six confound families through a
matched decomposition matrix and six targeted studies. The program answers:

> Is the observed ceiling caused by remote data movement, connection management,
> Python/executor orchestration, GPU submission, or harmful active concurrency?

## Guiding thesis (expected narrative arc)

This section states the qualitative story the program is designed to support
*or refute*. It is not a claim of results. Specific magnitudes are intentionally
omitted; pre-registering target numbers would bias the analysis. Each clause
maps to a study that can falsify it.

1. **The single-request path is viable.** The first-miss tax is real but smaller
   than originally reported, and is primarily a property of the Python/PyTorch
   execution path rather than unavoidable remote-GPU computation. *(S1, with B6
   as the runtime-stack replacement control.)* "Caused by Python/PyTorch
   dispatch" is defensible; "dominant contributor" is safer than claiming every
   source has been independently excluded.

2. **Large jobs are not inherently bad; uncontrolled fan-out is.** The original
   "splitting hurts" result conflated splitting with RPC fan-out. Server-side
   cooperative slicing at an appropriate quantum is expected to reduce
   head-of-line blocking without the catastrophic heavy P99 of client fan-out.
   Any reported improvement must carry its tested context: active concurrency,
   workload composition, offered load, baseline policy, slicing quantum.
   *(S2.)*

3. **Submission priority can be counterproductive under concurrency-dependent
   interference.** Naive Client-SJF can allow many light requests to reach the
   GPU together, causing interference that outweighs the queueing benefit.
   Scheduling must account for both job size *and* the execution interference
   created by simultaneously active jobs. This also explains why the original
   SJF result was misleading: it measured a submission artifact rather than
   controlled server-side scheduling. *(S3, with execution-order
   instrumentation.)*

4. **Heavy-lane is a graded admission policy, not a universal winner.** H=1 can
   be too restrictive for sparse-heavy workloads, creating its own queueing
   catastrophe. A small but non-unit heavy concurrency limit is expected to
   balance heavy-job queueing against concurrent-execution interference. The
   value of class-aware gating is workload-dependent; sometimes ordinary bounded
   concurrency captures most of the benefit. *(S4, with the global-cap control.)*

5. **Physical QP provisioning and active execution must be separated.** A large
   preconnected QP pool is not harmful by itself. Performance degrades when the
   pool implicitly permits excessive active concurrency. The architecture can
   provision many QPs while independently enforcing a smaller active cap.
   *(S5.)*

6. **Mixed workloads lose capacity nonlinearly.** Mixture capacity is expected
   to fall below the simple service-demand prediction, with the interaction
   ratio quantifying heterogeneity overhead. A small heavy fraction can be more
   damaging to tail latency than its average work contribution suggests.
   *(S6, with the linear mixture-capacity baseline.)*

### Refinements to two initial reviewer hypotheses

The program is also designed to test two hypotheses from the original review
that may require softening:

- **Timing-method hypothesis.** If, under the corrected matched server
  lifecycle, CUDA-event and synchronized wall-clock measurements produce nearly
  identical first-miss ratios, then timing method was not the dominant confound
  in this regime. The larger inflation may have come from server lifecycle
  differences. *(S1.)*

- **Python/TCP bottleneck hypothesis.** The prototype may be control-path-
  sensitive at low concurrency but concurrency/interference-limited at high
  load. Declaring either Python or GPU to be "the bottleneck" is too coarse;
  the dominant factor changes with operating regime. *(Anchor matrix + S4/S5/S6.)*

The expected default design emerging from the campaign, if the thesis holds:

> C++ recovery worker + server-side slicing + decoupled QP provisioning/admission
> + bounded heavy concurrency + open-loop capacity-based admission thresholds.

The exact active cap, quantum, and heavy cap remain workload- and load-dependent.

## North-star follow-up (out of scope for this program)

This program determines *how to build the remote path correctly*. It does not
establish *when* remote recovery is better than CoLoRA's local CPU recovery.
The next case-study layer should compare:

- CoLoRA local CPU recovery
- Remote GPU recovery, original prototype
- Remote GPU recovery, improved design (per this program)
- Local-GPU oracle

using: decode latency and tail; throughput; local CPU memory saved; PCIe
traffic saved; network traffic; remote GPU cost and utilization; rank and
adapter size; miss rate and burstiness; EP interference; multiple inference
clients. That comparison turns the current mechanism result into the larger
systems claim:

> Under which workload and resource conditions does disaggregated remote
> recovery outperform local CPU miss recovery?

## Architecture

> **B0 is an external lower bound. B1–B9 form a matched-contrast matrix over
> transport persistence, runtime stack, and active concurrency. Each targeted
> study uses the smallest matched subset needed to isolate its confound.**

### Three axes

| Axis | Values |
|------|--------|
| Control transport | persistent TCP (multiplexed, request IDs, out-of-order responses), per-request TCP |
| Handler runtime | Python direct, Python executor, C++ matched worker |
| Active concurrency | 1, N (=8) |

The RDMA + GPU data plane is identical for every remote cell.

### Ten cells

| Cell | Transport | Runtime | Conc | Purpose |
|------|-----------|---------|-----:|---------|
| B0 | local buffers (external) | Python direct | 1 | Lower bound |
| B1 | persistent TCP | Python direct | 1 | Remote baseline |
| B2 | persistent TCP | Python executor | 1 | Executor overhead at conc=1 |
| B3 | persistent TCP | Python executor | N | Python concurrency |
| B4 | per-request TCP | Python executor | 1 | Connect cost, Python, conc=1 |
| B5 | per-request TCP | Python executor | N | Current architecture |
| B6 | persistent TCP | C++ matched worker | 1 | Runtime-stack contrast, conc=1 |
| B7 | persistent TCP | C++ matched worker | N | C++ concurrency |
| B8 | per-request TCP | C++ matched worker | 1 | Connect cost, C++, conc=1 |
| B9 | per-request TCP | C++ matched worker | N | Full C++ counterpart |

### Valid matched contrasts (each changes exactly one axis)

```
B2 - B1   executor/queue overhead, Python, conc=1
B3 - B2   concurrency effect, Python, persistent TCP
B4 - B2   per-request TCP connection cost, Python, conc=1
B5 - B3   per-request TCP connection cost, Python, conc=N
B6 - B2   Python vs C++ runtime stack, persistent TCP, conc=1
B7 - B3   Python vs C++ runtime stack, persistent TCP, conc=N
B7 - B6   concurrency effect, C++, persistent TCP
B8 - B6   connection cost, C++, conc=1
B9 - B7   connection cost, C++, conc=N
B8 - B4   Python vs C++, per-request TCP, conc=1
B9 - B5   Python prototype vs matched C++ counterpart, conc=N
```

### "C++ matched worker" definition

B6–B9 use a handler semantically matched to the Python executor path: same queue
policy, same active-concurrency cap, same request lifetime, same serialization
schema, same QP acquisition behavior, same GPU stream policy. The claim is
**runtime-stack replacement benefit**, not "pure Python-language overhead." If a
fundamentally different async architecture is introduced, the contrast is renamed
accordingly. Enforced as a design constraint.

### Optional B0.5 — Remote RDMA-doorbell control path

Remote cell with no TCP on the measured path. Practical triggers: pre-posted
RDMA SEND, RDMA WRITE into a command ring + doorbell, dedicated control QP, or
continuously polled remote command queue. Preserves same RDMA READ + GPU compute
+ RDMA WRITE. If implemented, cleanly separates RDMA-data-plane cost from
TCP-control cost. Optional; if absent, no remote-vs-local gap is attributed to
TCP specifically.

### Primary cells per sub-experiment

| Study | Cells | Rationale |
|-------|-------|-----------|
| S1 | B1, B2, B6; B5 as reproduction | B2 = quiet Python path. B6 = runtime-stack replacement. B1 = executor-entry effect. |
| S2 | B1/B2 (atomic+client-chunk), B2/B3 (slicing), B5 | B1 cannot support slicing (no executor). |
| S3 | B2, B3 | Matched executor substrate, conc 1 vs N. |
| S4 | B3, B5-admission, B5-original | Transport + token-placement isolation. |
| S5 | B3, B5 | Pool/cap under matched concurrency. |
| S6 | B3, B5-corrected, B5-original-generator | Architecture + generator isolation. |

## Common instrumentation

### 19 timestamps (t0–t18), stable schema

```
t0  arrival generated         t9  QP acquired
t1  client admission          t10 RDMA READ starts
t2  TCP connect start         t11 RDMA READ completes
t3  TCP connected             t12 GPU work submitted
t4  request serialized        t13 GPU work starts (CUDA event)
t5  request sent              t14 GPU work completes (CUDA event)
t6  server request received   t15 RDMA WRITE starts
t7  handler queued            t16 RDMA WRITE completes
t8  handler starts            t17 response sent
                                t18 client receives response
```

### Timing domains

- **Client-local:** t0–t1, t2–t3, t18 (and t4–t5 if client-side). t18 = client
  receives response.
- **Server-local:** t6–t7, t8–t9, t10–t11, t12–t14, t15–t17. t17 = response
  sent (server event).
- **Cross-domain gaps (t5->t6 request wire, t17->t18 response wire):** without
  PTP, neither directional gap can be computed independently. The only
  defensible non-PTP quantity is a combined residual (below).

### Per-cell interval map

Every cell declares which timestamps are present. Persistent-TCP cells (B1/B2/
B3/B6/B7) have `t2=t3=null` (no per-request connect); connection-level
timestamps are recorded once at setup, not per request. Per-request-TCP cells
(B4/B5/B8/B9) have t2/t3 per request. **Never insert zero** for absent
timestamps — zero would imply free per-request connection setup. Null is
recorded as null and excluded from interval arithmetic.

### Accounting model

Use host-local spans first, so that uninstrumented local intervals do not leak
into the cross-domain residual.

```
client_request_span = t5 - t0      (one clock domain)
server_span         = t17 - t6     (one clock domain)
E2E                 = t18 - t0     (one clock domain)

client_local_gap    = client_request_span - sum(instrumented client-local subintervals)
server_local_gap    = server_span         - sum(instrumented server-local subintervals)
instrumentation_gap = client_local_gap + server_local_gap

cross_domain_residual = E2E - client_request_span - server_span
```

This yields the exact decomposition:

```
E2E = measured_client_subintervals
    + client_local_gap
    + measured_server_subintervals
    + server_local_gap
    + cross_domain_residual
```

The `cross_domain_residual` now corresponds to the combined request/response
cross-host remainder (wire + kernel-buffer + omitted cross-host control), not to
uninstrumented local queueing. The `instrumentation_gap` is the local
accounting-quality check.

Report `instrumentation_gap_us`, `instrumentation_gap_fraction`. Flag if **both**
`instrumentation_gap_fraction > 5%` **and** `instrumentation_gap_us > 50us`.
The 50us absolute threshold is calibrated from a no-op control-message
experiment.

Report `cross_domain_residual_us` and `cross_domain_residual_fraction`
descriptively, but do not use either as a pass/fail accounting check (it is
unavoidable without PTP and is closed by construction).

### Client-side admission timestamps

Freeze the meaning of client-side stages (do not overload `t1`):

```
c0 = entered client ingress queue
c1 = selected for network dispatch
c2 = system admission decision (if acknowledged)
t0 = scheduled generation time
t5 = request sent
```

`t1` (client admission) is retained for backward compatibility but maps to `c1`
in the new schema. The four client-side concepts — ingress queue, dispatch
selection, system admission, server admission — are distinct and must not share
one timestamp.

### Dual timing domains for GPU work

CPU submission wall-clock (t12) **and** CUDA-event elapsed (t13->t14). Reported
separately, never subtracted across domains.

### Scheduler-specific timestamps (S3, S4)

```
s0 = entered class queue
s1 = first became eligible
s2 = selected by dispatcher
s3 = heavy token acquired (if applicable)
s4 = submitted to executor
s5 = handler started
```

Report `class_queue_wait`, `heavy_ineligible_wait`,
`eligible_but_not_selected_wait`, `executor_wait`.

### Rules

- No cross-host clock subtraction unless PTP verified. Client-measured E2E +
  server-local segments + request IDs for correlation + CUDA events for GPU.
- No causal attribution of residuals without a dedicated experiment.
- Counterbalanced config order.
- **Trials:** >=3 for debugging; >=5 with CIs, or long runs with bootstrap CIs for
  final P99 claims. Short-sync runs (200 obs) labeled exploratory only.
- Fixed or recorded GPU clocks; identical tensor shapes/weights; output
  correctness checks per variant; success/failure counts; CIs on all percentiles.
- P99.9 only with >=10,000 class samples; else P99 + max + CCDF.
- Distinguish P99 pooled over all requests vs distribution of per-trial P99.
- **Paired arrival traces** across policies/cells where applicable: same class
  sequence, interarrival times, request IDs, tensor dims, seeds. Matched-rho reuses
  class sequence, scales interarrivals.

## S1 — First-miss tax re-measurement

### Reviewer concern

`cuda_graph` variant times only `miss_i_alloc` and hardcodes dtype/mm1/mm2=0.0;
sync placement differs (1x/miss vs 4x/miss). Subsegment sums non-comparable.
Total-service vs subsegment-sum inconsistency unresolved (same_weights/cache_flush
show 9–12us segments but 1.19ms total).

### Cells

B1, B2, B6 (primary); B5 (reproduction).

### Variants

1. **baseline** — full per-miss decomposition (alloc, dtype, mm1, mm2), each
   bracketed by CUDA events.
2. **same_weights** — weight index fixed to 0 for all misses.
3. **cache_flush** — split into two separate diagnostics (separate configs, not
   one combined variant):
   - **allocator-reset:** `torch.cuda.empty_cache()` before each miss (affects
     the caching allocator, *not* L2). Recorded as a diagnostic, not as a
     "flush."
   - **device-cache-perturbation:** scratch write sized from the **runtime-
     queried** device L2 cache size (via CUDA device attribute
     `cudaDevAttrL2CacheSize`, not a hardcoded constant), not a fixed 64 MB.
     Scratch write may globally perturb concurrent work; reported as a
     separate diagnostic, not bundled into "cache_flush."
   Perturbation cost is reported separately and not hidden:
   `perturbation_cost_us`, `compute_after_perturbation_us`,
   `total_E2E_including_perturbation_us`. A variant must not appear faster
   only because its preparation cost was omitted from the request total.
4. **cuda_graph** — capture one graph per weight index, replay per miss. **Per-op
   subsegment timing is NOT comparable to eager variants:** a single graph replay
   is one submitted unit, and external CUDA events measure total graph time, not
   internal alloc/dtype/mm1/mm2. Report instead:
   - graph replay CPU submission time
   - graph replay total GPU time (CUDA events around replay)
   - E2E service time
   
   Do not claim directly comparable per-operation graph segments. (Options B
   — internal event-record nodes — and C — separate per-op graphs — are noted
   as non-primary: B changes the graph being measured; C destroys the
   launch-amortization property being studied.) Allocation semantics also
   differ: practical CUDA graphs require stable/preallocated buffers, so the
   graph "alloc" segment is not equivalent to eager allocation.

### Timing

Dual domain for eager variants: CPU submission wall-clock (`perf_counter`, no
sync inside) + GPU execution (CUDA events before/after each op). For the
cuda_graph variant: CPU submission + total GPU replay time + E2E only (above).

Full accounting per request: `instrumentation_gap` closes near zero (see common
instrumentation). The CUDA-graph path does not expose comparable
alloc/dtype/mm1/mm2 subsegments, but total graph GPU time remains fully covered
by the replay interval (t13->t14). Reduced subsegment granularity is **not**
treated as instrumentation gap.

### What it answers

- Does the tax persist in B2 after fixing the timing method?
- Does B6 materially reduce the first/rest ratio? If yes -> the Python runtime
  stack contributed to the tax. If the tax persists, replacing the Python stack
  was insufficient; the residual remains **unattributed** and is investigated
  using the allocator, weight-reuse, cache-perturbation, and CUDA-graph
  diagnostics. Do NOT conclude "GPU-side" from B6 persistence alone — a
  residual could also come from CUDA-runtime initialization, C++ framework
  dispatch, allocator behavior, library initialization, graph/capture setup, or
  another unisolated control-path component.
- Does `same_weights`/`allocator-reset`/`device-cache-perturbation` change the
  tax under correct timing?
- Is the residual first/rest ratio under cuda_graph still 1.45x? If so, **no
  attribution claim** — "residual exists, root cause not isolated."

### Explicit non-claims

- No "likely cuBLAS algorithm selection" without a dedicated experiment.
- No "Python dispatch" claim from B5 alone — only from B6-B2 matched contrast.

### Configs

5 variants (baseline, same_weights, allocator-reset, device-cache-perturbation,
cuda_graph) x 4 cells x 4 NM (1,2,4,8) x 5 trials = **400 runs**.

## S2 — Splitting re-test (client-side fix + server-side cooperative slicing)

### Reviewer concern

Original multiplies TCP sockets, QP acquisitions, executor workers, CUDA launches
simultaneously. Rejects RPC fan-out, not splitting. Server cooperative slicing
untested.

### Goal

Distinguish four mechanisms with controlled comparisons:
1. **Atomic** — 1 control message, 1 READ/WRITE, no yield
2. **Stateful client interleaving** — N control messages, 1 READ/WRITE, client
   chooses boundaries/order
3. **Server cooperative slicing** — 1 control message, 1 READ/WRITE, server
   chooses boundaries/order
4. *(Original B5 fan-out)* — reference only

Central comparison: **(2) vs (3)** — same 1 READ/1 WRITE, same interleaving,
differs only in scheduling location + control count.

### Cells

- B1 (atomic + stateful client chunking only — no executor -> no server scheduler)
- B2 (primary slicing diagnostic; scheduler with one active quantum)
- B3 (slicing under real concurrent GPU pressure)
- B5 (atomic reproduction + original fan-out reproduction + server slicing
  initiated by one B5 request)

B5 cannot run stateful client chunking (needs persistent multiplexed TCP + shared
QP + retained server state). Optional B5-stateful hybrid: per-request TCP +
persistent logical session + shared QP + retained state, reported under its own
name.

### Persistent TCP transport requirement

B1–B3 use framed multiplexing with request IDs; responses may complete out of
order. Hard requirement, not implementation detail.

### Scheduling policies (deterministic, preregistered)

**Server slicing — per-logical-request round-robin:**
- At most one quantum from each logical request runnable at a time
- After quantum completes, continuation appended to tail
- Newly arrived light requests enter same runnable queue at tail
- Scheduler selects queue head
- Separates quantum-boundary introduction from priority policies (S3)

**Client interleaving — round-robin across ready logical requests:**
- At most one outstanding chunk per logical request
- Newly ready logical request appended to tail
- Next chunk sent from queue head

**Contiguous client chunking:** chunks of one logical request sent back-to-back.

### Asynchronous yield sequence

```
1. Worker submits one quantum to a CUDA stream
2. Worker records a per-stream CUDA completion event
3. Worker immediately returns to the worker pool         <- CPU released
4. Event poller/callback observes quantum completion
5. GPU admission token released                          <- GPU token released
6. Continuation marked runnable, appended to scheduler queue
7. Scheduler later selects the continuation
8. Next quantum submitted
```

- CPU worker released immediately after submission (step 3)
- GPU admission token released only at quantum completion (step 5)
- Continuation runnable only after completion (step 6)
- QP/session state retained across suspension
- No global `torch.cuda.synchronize()`; no busy-wait; no pre-submission of
  future quanta

### Mechanism (b): Stateful client chunking

- `max_workers=1` for chunks of one logical request
- Persistent multiplexed TCP, shared QP, retained server state
- **Stateful session:** first chunk does READ; server retains activation +
  output; intermediate chunks compute only; final chunk does WRITE
- Two submission policies: contiguous + interleaved (round-robin)
- Chunk sizes: 1, 2, 4
- **Stateless variant** (each chunk re-reads) run as secondary comparison

### Mechanism (c): Server cooperative slicing

One RPC carries `NM=8` + `quantum`. One READ, quantum-boundary cooperative
scheduling, one WRITE. Quantum sizes: **1, 2, 4**.

### Mechanism (d): Matched persistent fan-out (H1 control, mandatory)

To isolate RPC fan-out from transport persistence, add a B3-matched fan-out
variant:

- Same persistent multiplexed TCP connection as B3
- Same serialization, same active cap N=8
- Multiple chunk requests submitted **concurrently** (not sequentially) as
  independent executor tasks, each acquiring its own QP
- No server-side session state shared across chunks (stateless fan-out)

This is the matched control for H1. Compare within B3:
- persistent concurrent fan-out (mechanism d)
- persistent sequential/stateful chunking (mechanism b)
- server slicing (mechanism c)

Without this variant, H1 cannot attribute the catastrophe specifically to
fan-out; it can only show that *bundled removal* of fan-out + repeated connect +
repeated QP-acquisition eliminates the result.

### q=8 equivalence validation

Required before interpreting q=1/2/4:
- Output correctness within numerical tolerance
- Same RDMA op count, same GPU work
- **Median service-time difference <=5%, P99 difference <=10%**

If q=8 differs materially, slicing state machine confounds q=1/2/4.

### Scheduler-only no-op microbenchmark

`enqueue -> select -> callback -> requeue` with no GEMMs. Lower-bound cost of one
yield.

### Logical latency definition (uniform)

**Primary: logical E2E** = original logical request `t0` -> final result available
at client `t18`. Separately: server service span, GPU compute span, RDMA
completion span.

### Offered-load methodology

Two modes:
1. **Common offered load** — (a) safe common rate below min capacity of all
   mechanisms; (b) common rate near atomic baseline knee
2. **Matched utilization** — each mechanism at rho=0.7 and rho=0.85 of its own
   capacity

Arrival process generates **logical requests**, not chunks.

### Per-quantum event recording

`quantum_ready, quantum_selected, GPU_submitted, GPU_start, GPU_end,
continuation_requeued` per quantum. Per-miss/per-kernel CUDA events where
feasible. Kernel overlap from GPU event intervals on common device timeline or
CUPTI/Nsight subset — not CPU submission timestamps.

### S1 x S2 interaction: CUDA-graph sub-study

| Policy | Eager | CUDA graph |
|--------|------:|-----------:|
| Atomic | ✓ | ✓ |
| Server slice q=1 | ✓ | ✓ |
| Server slice q=2 | ✓ | ✓ |
| Server slice q=4 | ✓ | ✓ |

### Preregistered hypotheses

| Hyp | Expected evidence |
|-----|-------------------|
| H1: RPC fan-out caused catastrophic split | **Primary (matched):** at persistent transport and same active cap, concurrent independent chunk RPCs produce materially worse logical-request latency and/or throughput than sequential stateful chunking. **Caveat:** persistent fan-out still changes concurrent executor tasks, QP acquisitions, RDMA op count, and server state sharing, so the conclusion is "independent RPC fan-out as an implementation bundle is harmful even after connection setup is removed" — a pure "concurrency alone" claim would require a stateful concurrent-chunk ablation (optional). **Reproduction:** original B5 fan-out remains reproduction evidence. |
| H2: Contiguous chunking cannot reduce HOL | No material light-P99 improvement; worse heavy latency or throughput. Negative control. |
| H3: Interleaved client chunking reduces light P99 | Lower light P99, bounded heavy slowdown. Caveat: single-client only; not generalized to multi-client. |
| H4: Server slicing provides cleanest mitigation | Lower light P99 than atomic + client chunking at similar throughput. Strongest comparison: server slicing vs stateful interleaved client chunking. |
| H5: q=1 is too fine | Dispatch/yield overhead, throughput loss. CUDA-graph subset especially important. |
| H6: q=2 or q=4 is best tradeoff | Upper 95% CI of light-P99 ratio <= 0.80; lower 95% CI of throughput ratio >= 0.90; upper 95% CI of heavy-P99 ratio <= 1.25. |
| H7: Slicing's HOL benefit is larger under heterogeneous contention | Larger light-P99 benefit in 1h8l; little/no benefit in 2h0l except possible contention-control; potential fairness improvement among heavy. |

### Staged campaign

**Stage A — diagnostic selection (B2, B3, 1h8l):** 13 variants x 2 cells x 1
composition x 2 arrival modes x 5 trials = 260 runs. + matched persistent
fan-out 3 chunk sizes x 1 cell (B3) x 1 composition x 2 modes x 5 trials = 30
runs. + q=8 equivalence 20 runs. + scheduler no-op 10 runs. **Stage A: ~320
runs.**

**B5 reference:** 3 fan-out sizes x 1 cell x 2 compositions x 2 modes x 5 trials
= 60 runs.

**Stage B — validation:** atomic (B1/B2/B3/B5) + best client (B1/B2/B3) + best
slicing (B2/B3/B5) x 2 compositions x 2 modes x 5 trials = 200 runs (220 with
B5-stateful).

**CUDA-graph sub-study:** 4 policies x 2 (eager/graph) x 2 cells x 1 composition
x 2 modes x 5 trials = 160 runs.

**S2 total: ~740-760 runs.**

## S3 — Scheduling study

### Reviewer concern

Original "SJF" is client-side submission sort only. With
`ThreadPoolExecutor(max_workers=N)`, submission order != execution order. 3x gain
for both classes suspicious. Server-side priority queuing untested.

### Cells

B2 (persistent TCP, Python executor, active cap=1), B3 (active cap=N=8).
**Executor pool size fixed across B2/B3; only admission semaphore changes.**
`B2: workers=N, active cap=1` / `B3: workers=N, active cap=N`.

B2 permits multiple admitted/queued requests but only one active logical request
at a time in the primary atomic study (one active quantum at a time in the
sliced secondary study). Queued requests visible to scheduler.

**Multiplexed transport inherited from S2.**

### Primary study: atomic, non-sliced requests only

Slicing disabled. Priority+slicing is secondary.

### Policy definitions (preregistered)

| Policy | Client behavior | Server behavior |
|--------|-----------------|-----------------|
| FIFO | Dispatch oldest ready request | Central queue dispatches in arrival order |
| Client-SJF | Dispatch shortest predicted job among currently ready local jobs | Central queue remains FIFO |
| Server-SJF | Dispatch FIFO | Central priority queue chooses shortest predicted job; at most `active_cap` dispatched |

- Client-SJF under sync reduces to batch sort (reproduces original as reference)
- Client-SJF under Poisson is online: requests enter client-local ready queue in
  arrival order; whenever dispatch slot free, shortest predicted ready job chosen.
  No arbitrary batching window.
- Server-SJF uses **central dispatcher before executor**: `request received ->
  admitted to central server queue -> priority dispatcher selects ->
  active-concurrency token acquired -> submitted to executor/GPU path`. Selected
  job runs to completion (primary study).

### Client outstanding-request window

```
client_outstanding_cap = W (= N = 8 primary; W=2N sensitivity)
```

At most W requests sent but unfinished; additional arrivals wait in client ready
queue. Same W for all policies and both B2/B3. Record
`client_ready_queue_wait = dispatch_start - client_queue_entry`.

### Priority key: calibrated predicted service time

```
S_hat_cell(NM) = median isolated atomic service time, same cell's runtime/transport
NM in {1,4,8} calibrated separately
```

For sliced execution (secondary): progress-aware remaining time
`R_hat(total_nm, completed_nm) = sum predicted costs of unprocessed misses + final
copy/RDMA-WRITE`. Model knows whether first-miss/setup cost already paid.

Prediction accuracy reported: MAPE, predicted-vs-actual rank correlation.

### Execution-order instrumentation

Four sequences per request: client submission, server handler-start, GPU-kernel-start (CUDA events on common device timeline or CUPTI/Nsight subset), completion.

**Summary metrics:**
- **Priority fidelity** = fraction of decisions where light request starts first
  when both light and heavy queued
- **Inversion count** = number of times NM=8 job starts while earlier-ready NM=1
  remains queued
- **Order preservation** via Kendall's tau / Spearman with tied-NM and
  concurrent-start handling

### Offered-load methodology

1. **Common logical offered load** — at minimum one point below min capacity of
   all policies; plus near atomic baseline knee
2. **Matched utilization** — each policy at rho=0.7, 0.85 of own capacity

### Arrival modes

- Synchronized (200-obs, **exploratory only**)
- Poisson (60–120s, >=5 trials, **final claims**) — 1h8l, 2h0l, medium
  composition `1xNM=8 + 2xNM=4 + 8xNM=1`

### Fairness and starvation metrics

Max heavy queue wait; P99.9 heavy queue wait (long runs, >=10k samples); fraction
heavy waiting > threshold; completed throughput by class; queue length by class
over time. No aging initially (pure SJF characterization).

### Preregistered hypotheses

| Hyp | Expected evidence |
|-----|-------------------|
| H1: Client-side sorting does not reliably control GPU execution order at B3 | Client intended order weaker correlation with GPU-start than server-priority; original both-class improvement does not consistently reproduce; benefit correlates with actual execution order or reduced active concurrency |
| H2: Server-SJF outperforms Client-SJF | Higher priority fidelity, fewer inversions, lower light P99 at comparable throughput. Caveat: single client may approximate; two-client follow-up strengthens. |
| H3 (factorial): Decompose scheduling vs concurrency | `Delta_sched(c) = P99(FIFO,c) - P99(Server-SJF,c)` at c in {1,N}; `Delta_conc(policy) = P99(policy,N) - P99(policy,1)`; `Interaction = Delta_sched(N) - Delta_sched(1)`. Calculated **per class** (light, heavy separately) using ratios + absolute. **Primary factorial analysis uses paired common-load traces at the same absolute logical arrival rate.** Matched-utilization runs are used only for normalized operating-point characterization and are NOT used for the causal factorial contrast (different absolute rates confound the effect). Interpretation: `Delta_sched(1)>0` = queue-order benefit without GPU concurrency; `Delta_sched(N) > Delta_sched(1)` = scheduling also mitigates concurrent interference; large `Delta_conc` across all policies = active concurrency dominant factor; `FIFO@B2` beats all `@B3` policies = active concurrency dominates scheduling. |
| H4 (neutral): Priority and slicing may be complementary when light arrives behind active heavy | Priority alone cannot preempt. Success (multi-objective): light P99 improves >=20% vs standalone priority, heavy worsens <=25%, throughput falls <=10%, no starvation increase. May overlap rather than add. |

### Staged campaign

- **Stage A:** 3 policies x 2 cells x 1 composition x 1 safe common load x 5
  trials = 30 runs.
- **Stage B:** 3 policies x 2 cells x 3 compositions x {sync 1 condition +
  Poisson 4 load conditions} x 5 trials = 90 sync + 360 Poisson = 450 runs.
- **Priority+slicing secondary:** 1 policy x 2 quanta x 2 cells x 1 composition
  x 2 modes x 5 trials = 40 runs.
- **Two-client follow-up:** 2 policies x 2 cells x 1 composition x 1 mode x 5
  trials = 20 runs (independent queues/connections, fixed aggregate rate).

**S3 total: ~540 runs.**

## S4 — Heavy-lane re-test

### Reviewer concern

(1) Poisson table reports light P99 20,218->39,983 us but text claims "improves
2x." (2) Original tested only semaphore=1 and one mixture; universality
unestablished.

### Cells

- B3 (primary)
- B5-admission (per-request TCP, heavy token acquired *before* executor/GPU
  admission) — fair comparison with B3
- B5-original (original semaphore placement, exact reproduction) — resolves
  transcription inconsistency only

### Heavy-lane mechanism (exact admission rule)

```
Maintain separate FIFO light and heavy queues.

A job may start if:
  total_active < N  AND  (job is light  OR  active_heavy < H)

When an active slot becomes free:
  choose the oldest eligible request across the two queue heads.
  If the oldest request is a blocked heavy job,
  an eligible light request may bypass it.
```

At most H active heavy; at most N active total; no dedicated reservation; light
may occupy >N-H slots if <H heavy active; heavy FIFO within class; light FIFO
within class; bypass only when heavy temporarily ineligible.

### Heavy threshold

`heavy if NM >= 4; light if NM = 1`. NM>=8 threshold sensitivity for medium
composition.

### H sweep

`H in {1, 2, 4, 8}`. H=8 = no heavy sub-cap (bounded by total N=8).

### Mixtures

**Synchronized** (collectively exercise all H values):
| Label | Composition |
|-------|-------------|
| 2h8l | 2xNM=8 + 8xNM=1 (distinguishes H=1 vs H>=2 only) |
| 8h16l | 8xNM=8 + 16xNM=1 (distinguishes all H) |
| 8h0l | 8xNM=8 (all-heavy control; distinguishes all H) |

1h8l retained only for B5-original inconsistency reproduction.

**Poisson** (genuinely distinct heavy fractions):
| Label | Heavy fraction |
|-------|---------------:|
| 1h9l | ~10% |
| 1h3l | ~25% |
| 1h1l | ~50% |
| all-heavy | 100% (sustained, from S4 not S6) |

### Offered-load methodology

1. Common logical offered load — below min capacity of all H; near baseline knee
2. Matched utilization — each (cell, mixture, H) at rho=0.7, 0.85 of own
   `C(cell, mixture, H)`

### Class-aware vs global-cap control (Stage C, frozen workload)

Preregistered workload (not chosen after viewing Stage B):
- **cell:** B3
- **mixture:** Poisson 1h3l (25% heavy)
- **loads:** (1) safe common below min capacity of all 7 policies; (2) common
  rate near ungated H=8 knee

Compare: heavy-lane (N=8, H in {1,2,4}) vs global cap (K in {1,2,4}, no heavy
sub-cap) vs ungated (N=8, H=8). If heavy-lane preserves light latency +
throughput better than global K=2, benefit is genuinely class-aware. All-heavy
cannot demonstrate uniquely heavy-lane benefit (H=global K operationally).

**Normalized SLO thresholds** use the isolated service time for the same cell,
class, and runtime path (e.g., `P(latency > 2x isolated)`, not an absolute
target chosen post-hoc).

### Resolving the light-P99 inconsistency (Stage A)

H in {1,8} x B5-original x 1h8l x Poisson at load where heavy cap binds x >=5
trials x corrected 19-timestamp instrumentation. Three outcomes:
- light P99 worsens at H=1 -> old table correct, text was error, heavy-lane
  heavy-favorable
- light P99 improves at H=1 -> old table was error
- neither historical value reproduces within uncertainty -> mark inconsistency
  unresolved; compare original vs corrected metric code, class filters,
  timestamp boundaries, and realized load to identify the source

### B5-original reproduction fidelity

Preserve original executor width, QP pool, connection lifecycle, arrival
calibration, composition, semaphore placement, latency calculation *alongside*
corrected 19-timestamp metric on same requests. Determines whether inconsistency
was prose transcription, CSV extraction, class filtering, latency definition, or
measurement-code error.

### Metrics

Per H x mixture x cell:
- Light/heavy P50/P95/P99 with CI (pooled + per-trial)
- Overall P99, per-class throughput, overall throughput
- Per-class queue wait, waiting-time CCDF
- Heavy slowdown vs isolated service
- **SLO attainment at multiple thresholds**: P(latency > 2x isolated), P(>5x
  isolated), P(>10ms), P(>30ms) — no single arbitrary target
- Starvation / max wait
- **H-binding diagnostics**: active-heavy distribution, fraction time
  active_heavy=H, heavy-token wait, fraction arrivals encountering cap, avg/max
  heavy queue length
- P99.9 only with >=10,000 class samples

### Pareto reporting

**Three-dimensional frontier**: minimize light P99, minimize heavy P99, maximize
throughput. Plus **constrained Pareto**: light/heavy P99 frontier among policies
satisfying `throughput >= 90% of baseline` + `rejection <=1%` + `timeout <=1%` +
`end-of-run backlog stable/drained`. Frontiers separate for common-load vs
matched-utilization.

**Dominance with uncertainty**: A dominates B only if A no worse on all
constrained objectives and significantly better on at least one via paired-trial
CIs. Baseline = same cell/mixture/load, H=8.

### Open-loop backlog and censoring

Record generated/admitted/started/completed/rejected/timed-out/unfinished/
final-queue-length. Drain phase: 60–120s generation -> bounded drain -> include
full E2E -> separately report censored. Queue-length slope near end reported;
growing queue = unstable even if completed P99 bounded.

### Capacity stability

`C(cell, mixture, H)` = highest offered rate where generated~=admitted~=completed,
queue statistically stable, timeout/rejection <=1%. Throughput plateau with
growing queue is not stable capacity.

### Preregistered hypotheses

| Hyp | Expected evidence |
|-----|-------------------|
| H1 (neutral): H=1 does not universally dominate H=8 across light latency, heavy latency, throughput | Any of: heavy improves/light worsens; light improves/heavy worsens; both improve but throughput falls; H=1 worse on all; effects vary by mixture/load. B5-original resolves table-vs-prose. |
| H2: At least one intermediate H lies on Pareto frontier or maximizes constrained SLO/throughput objective | H=2 or H=4 better balance than both extremes for >=1 mixture; need not strictly dominate on every metric. |
| H3: Pareto-optimal H changes with heavy fraction | Sparse prefers stricter gating; dense may require larger H for throughput; all-heavy exposes pure concurrency-control tradeoff. |
| H4: In all-heavy, lower H may improve P99 via limiting harmful simultaneous heavy execution | Mechanism not attributed specifically to concurrent-kernel submission until GPU traces isolate it. |
| H5: No single H minimizes all objectives simultaneously; best H flips by objective | Different objectives or SLO constraints select different H values; no H significantly dominates every other H across light P99, heavy P99, and throughput. Constrained objective: maximize throughput s.t. light P99 <= L_slo, heavy P99 <= H_slo. |

### Staged campaign

- **Stage A:** H in {1,8} x B5-original x 1h8l x 1 Poisson load x 5 trials = 10 runs.
- **Stage B sync:** 4 H x 2 cells (B3, B5-admission) x 3 mixtures x 1 burst x 5
  trials = 120 runs.
- **Stage B Poisson:** 4 H x 2 cells x 3 mixtures x 4 load conditions x 5 trials
  = 480 runs.
- **Sustained all-heavy Poisson (H4 final):** 4 H x 2 cells x 2 loads x 5 trials
  = 80 runs.
- **Stage C:** 7 policies x 1 cell x 1 mixture x 2 loads x 5 trials = 70 runs.

**S4 total: ~760 runs.**

## S5 — Physical-pool x active-cap x executor x stream matrix

### Reviewer concern

Original "pool=32 worse" confounds physical QP count with active concurrency.
Executor=256 always; `pool.borrow()` is true gate. Never tested pool=32 with
active cap=8.

### Four independent axes

| Axis | Symbol | Values |
|------|--------|--------|
| Physical QP pool | P | 8, 16, 32, 64 |
| Active admission cap | A | 8, 16, 32 |
| Executor workers | E | 8, 16, 32 |
| CUDA streams | S | 1, 2, 4 |

**Effective concurrency is bounded by `min(P, A, E)`.** A full cross-product is
not a valid independent factorial design because many combinations are
structurally capped (e.g., with E=8, the values A=16 and A=32 are non-binding;
P=8, A=32 cannot demonstrate 32-way active QP-backed behavior). Therefore S5
uses **three separate matched sweeps**, each varying one axis while holding the
others high enough to be non-binding.

### Three matched sweeps

**Sweep 1 — Physical-pool effect (hold active behavior fixed):**
```
A=8, E>=8, P in {8, 16, 32, 64}
```
Primary contrast: `P=8,A=8` vs `P=32,A=8` (and `P=64,A=8`).

**Sweep 2 — Active-concurrency effect (hold pool and executor non-binding):**
```
P=64, E=32, A in {8, 16, 32}
```
Primary contrast: `P=64,A=8` vs `P=64,A=32`.

**Sweep 3 — Executor-width overhead at fixed active concurrency:**
```
P=64, A=8, E in {8, 16, 32}
```
Here `A=8` is deliberately binding; `E>=A` is non-binding. Because every
`E >= A`, the active cap remains 8. Any remaining difference is attributable to
executor-width overhead, queueing implementation, or GIL/thread-management
effects — not additional active work.

Primary contrast: `E=8` vs `E=32` at fixed (P=64, A=8).

**Optional coupled sensitivity (secondary, labeled as such):**
```
P=64, A=32, E in {8, 16, 32}
```
Labeled "executor bottleneck / effective-concurrency sensitivity," NOT
"independent executor-width effect," because here E<A makes executor width an
additional concurrency limiter.

**Stream sensitivity:** paired cells for H1 equivalence at each stream count:
`(P=8,A=8,E=8)` and `(P=32,A=8,E=8)` for every `S in {1, 2, 4}`, on B3. This
directly tests H1 at each stream count rather than selecting unrelated
representative configurations.

### Cells

B3 (primary), B5 (reference — note B5's per-request TCP makes pool/cap harder to
separate).

### Central tests (with equivalence, not "within CI")

> Does `P=32, A=8` match `P=8, A=8`?

Equivalence test: throughput ratio in [0.95, 1.05] and P99 ratio in [0.90, 1.10]
via TOST or paired-trial CI. "Within CI" alone does not establish equivalence.

### Confound check (now feasible)

> Does `P=64, A=32` (Sweep 2) degrade like original "pool=32"?

At `P=64, E=32, A=32`, active concurrency can actually reach 32 (not capped by
P or E). If this degrades like the original pool=32 result, active concurrency
was the real factor.

### Pool-acquisition instrumentation

`qp_wait_us` per request; active-QP distribution over time; fraction blocking on
pool; physical QP memory/resource cost (reported, not swept).

### Metrics

Per sweep x cell: throughput; light/heavy P50/P95/P99 with CI; `qp_wait_us`
distribution; active-QP distribution; fraction blocking on pool; GPU kernel
overlap (CUPTI/Nsight subset); per-class queue wait; capacity stability.

### Configs

**Sweep 1 (physical-pool):** 4 P x A=8 x E=8 x S=1 x 2 cells x 2 loads x 5
trials = 80 runs.

**Sweep 2 (active-concurrency):** P=64 x 3 A x E=32 x S=1 x 2 cells x 2 loads x
5 trials = 60 runs.

**Sweep 3 (executor-width):** P=64 x A=8 x 3 E x S=1 x 2 cells x 2 loads x 5
trials = 60 runs.

**Stream sensitivity (paired H1 cells):** `(P=8,A=8,E=8)` and `(P=32,A=8,E=8)`
for every `S in {1, 2, 4}`, on B3. The `S=1` configurations for both P-cells
already appear in Sweep 1; if those results are reused, stream sensitivity adds
**40 unique runs** (4 new cells: 2 P-cells x 2 new S values {2,4} x 2 loads x 5
trials). If S=1 is intentionally rerun as validation, add 20 duplicated
validation runs (state which).

### Historical reproduction subset (B5-original-pool)

The new matched sweeps use combinations such as `P=64, E=32, A=32`. None
reproduces the historical setup exactly. Add an explicit reproduction subset:

```
B5-original-pool:
P in {8, 16, 32}
E = 256
A coupled to P through the original pool.borrow() behavior
original CUDA-stream policy
original workload and offered-load calibration
corrected instrumentation alongside original metrics on the same requests
```

3 pool sizes x 2 loads x 5 trials = **30 runs**.

This distinguishes:
- **historical reproduction:** P in {8,16,32} under old coupled mechanism
- **causal active-cap test:** P=64, E=32, A in {8,16,32} (Sweep 2)
- **physical-pool test:** A=8, E=8, P in {8,16,32,64} (Sweep 1)

Without this subset, the new active-concurrency behavior can be called
**consistent with** the old result, but not that the old result was reproduced.

**S5 total: ~285 runs.** (Sweep 1: 80 + Sweep 2: 60 + Sweep 3: 60 + stream
sensitivity: 40 unique + historical reproduction: 30 = 270 matched-sweep +
historical; +15 buffer for S=1 validation reruns if used.)

### Preregistered hypotheses

| Hyp | Expected evidence |
|-----|-------------------|
| H1: Physical pool size does not matter once active concurrency controlled | `P=32,A=8` equivalent to `P=8,A=8` (TOST: throughput ratio in [0.95,1.05], P99 ratio in [0.90,1.10]) |
| H2: At fixed sufficiently large physical pool and executor width, increasing active cap reproduces prior degradation | `P=64,A=32` (E=32, non-binding) degrades like original pool=32; matches B5 reproduction |
| H3a (neutral): At fixed P=64, A=8, determine whether increasing E beyond A causes a material runtime overhead | `E=8` vs `E=32` at (P=64, A=8): equivalence test (throughput ratio in [0.95,1.05], P99 ratio in [0.90,1.10]) — if equivalent, no material overhead; if not, executor-width overhead detected |
| H3b (secondary): When E<A, executor width acts as additional concurrency limiter | Coupled sensitivity (P=64, A=32, E in {8,16,32}) shows E-dependent degradation correlated with effective-concurrency change |
| H4: CUDA stream count shifts absolute throughput but not the (P,A) equivalence | Stream count changes throughput uniformly; H1 equivalence holds at each S |

## S6 — Per-class capacity recalibration

### Reviewer concern

Service time excludes TCP -> rho underestimated. Inflight cap 256 + backpressure =
closed-loop. No generated/admitted/rejected split. "Flat P99" is artifact.

### Cells

B3 (primary), B5-corrected (per-request TCP, Python executor, N=8, corrected
open-loop generator), B5-original-generator (original rho, inflight cap 256,
backpressure, original metrics).

### Unique workloads (5)

1. NM=1 only
2. NM=8 only / all-heavy (collapsed)
3. 1h9l (10% heavy)
4. 1h3l (25% heavy)
5. 1h1l (50% heavy)

Sustained all-heavy reused from S4 when cell/cap/trace/duration/protocol match.

### Three-stage admission (primary: open queueing, no rejection)

```
t0 = scheduled arrival time
c0 = actual insertion into client ingress queue
c1 = selected for network dispatch
c2 = accepted by the server/control path
t18 = final response received

application E2E       = t18 - t0      (NOTE: t18 - t0, not t0 - t18)
generator lateness   = c0 - t0
client queue wait    = c1 - c0
dispatch/accept wait = c2 - c1
```

**Primary capacity mode (open queueing, no rejection):**

```
client_ingress_queue_capacity = number of scheduled arrivals in the full trace
admission_timeout             = none
overload_rejection            = disabled
overflow                      = invalid-run condition (flag, do not silently drop)
```

In overload, the system should exhibit: growing client/server queues;
accepted/completed rate below generated rate; unfinished requests at generation
end; drain censoring. Do NOT hide overload by rejecting enough arrivals to
stabilize the queue — that would measure an admission limiter, not raw service
capacity. The queue is sized from the full pre-generated trace, not merely
`10x max in-flight` (at 1.05C over a long run, 10x in-flight may be too small
and create artificial overflow).

**Separate admission-policy mode (explicit rejection):** a separate mode may
explicitly reject requests when the system cannot admit. This mode is used for
admission-control studies (S4 heavy-lane, S5 active-cap), NOT for S6 raw
capacity estimation.

Distinguish: not-yet-offered (c0 not reached), offered-but-rejected (c2 reject,
admission-policy mode only), admitted-and-queued (c2 admit, s0 not reached),
admitted-and-started (s5 reached).

### True open-loop generator

Arrivals independent of completion. No `pass` backpressure, no executor blocking.
In primary capacity mode, no rejection: every generated arrival enters the
ingress queue and waits. Generator records: generated, admitted (c2 accept),
client-queued, server-queued, started, completed, rejected (admission-policy
mode only), timed-out, unfinished-at-measurement-end, final-queue-length.

### Capacity definition

`C(cell, mixture)` = highest offered rate where **all of**:
- scheduled generation rate ~= actual c0 rate (within 1%)
- c0 arrival rate ~= c2 accepted rate (within 1%)
- c2 accepted rate ~= completion rate during generation (within 1%)
- client/server queue-slope CI includes zero
- rejection + timeout <=1% (rejection = 0 in primary open-queueing mode)

Not `admitted~=completed` alone. Measured during steady generation window, not
after drain.

### Bracketed capacity estimate

Low-load validation -> geometric increase to instability -> bracket boundary ->
binary-search to +/-5% -> `C_low` (highest stable), `C_high` (lowest unstable),
estimated C = interpolation. Use C_low for conservative matched-rho.

### Generator validation (Phase 0)

No-op sink benchmark: same trace code, same timestamping, minimal handler, no
RDMA/GPU. Verify scheduled vs actual generation rate, dispatch lateness, timer
resolution, client CPU saturation, max sustainable generator rate. Record
`scheduled_arrival_time`, `actual_generation_time`, `actual_dispatch_time`.
Async/multi-process generator if per-request TCP would bottleneck.

### Load sweep

```
lambda = 0.05C, 0.1C, 0.2C, 0.4C, 0.6C, 0.75C, 0.85C, 0.9C, 0.95C, 1.0C, 1.05C
```

### Measurement duration and drain

- Exploratory: 20s to estimate C
- Final: 60–120s generation near knee; longer if P99 unstable
- Drain: stop generation -> bounded drain -> include full E2E -> separately report
  censored
- Queue-length slope near end reported

### P99 cliff definition

Primary = lowest stable lambda where P99 >= 2x low-load P99. Optional: segmented
regression / change-point / max curvature. H4 neutral: characterize whether
transition sharp or gradual per workload.

### Drain censoring

If censored fraction <=1%: ordinary percentiles. If >1%: mark ordinary P99
invalid; report censor count + lower bounds + Kaplan-Meier survival estimate +
CCDF with censoring markers.

### Utilization (renamed)

`active-slot occupancy`, `GPU busy fraction` (CUPTI/Nsight subset), `CPU handler
utilization` — reported separately. No "GPU saturated" claim from active-slot
occupancy alone.

### Linear mixture-capacity baseline

```
1 / C_pred(p) = (1-p)/C1 + p/C8
interaction_ratio = C_observed(p) / C_pred(p)
```

Near 1 = average service demand explains capacity; <1 = heterogeneity overhead;
>1 = amortization.

### Original "flat P99" reproduction

B5-original-generator alongside corrected generator on same workload. Report
both. Determines whether "flat P99" was closed-loop backpressure artifact, rho
miscalibration, or real stability.

### Preregistered hypotheses

| Hyp | Expected evidence |
|-----|-------------------|
| H1: Original generator suppresses load variation via backpressure | Nominal rate increases but realized dispatch/admission plateaus; corrected open-loop exposes queue growth/rejection/latency/instability not visible in original |
| H2: Original capacity estimate omits material control-path/concurrency costs | Predicted capacity exceeds empirical stable capacity by a preregistered materiality margin: `C_pred / C_empirical >= 1.20` (i.e., predicted is at least 20% above empirical); anchor decomposition identifies omitted time; nominal rho=0.2 maps to much higher empirical fraction |
| H3: C(NM=1) > C(NM=8); mixture may deviate from linear prediction | Estimate whether mixture capacity differs from linear service-demand prediction; interaction_ratio may be below, near, or above 1 |
| H4 (neutral): Each workload has measurable latency-load relationship; characterize whether transition sharp or gradual, how differs by workload |

### Claim scope

S6 provides capacity evidence used *with* S4/S5 to select admission limits; does
not independently determine per-class admission cap.

### Configs

- Final long runs: 5 workloads x 2 cells x 11 loads x 5 trials = 550 runs
- Exploratory calibration: 5 workloads x 2 cells x ~5 probes x 3 trials = 150
  runs
- Original-method reproduction: 1 cell x 2 workloads x 9 rates x 3 trials = 54
  runs

**S6 total: ~754 runs.**

## Program execution order (decision-gated)

- **Phase 0 — Harness validation:** common instrumentation, output correctness,
  no-op generator validation, B0–B9 decomposition subset.
- **Phase 1 — Timing & baseline capacity:** S1; S6 exploratory calibration for
  B3/B5; establish safe common loads and approximate knees.
- **Phase 2 — Diagnostic policy stages:** S2 Stage A, S3 Stage A, S4 Stage A,
  S5 diagnostic matrix.
- **Phase 3 — Freeze architecture:** S5 determines whether N=8 remains
  appropriate or another cap/pool config preferred.
- **Phase 4 — Full policy validation:** S2–S4 final common-load and
  matched-utilization campaigns (using Phase 3's frozen architecture).
- **Phase 5 — Final capacity curves:** full S6 curve for baseline and selected
  improved configuration.

### Decision gates

- Stage A inconclusive -> debug/redesign before Stage B
- Policy clearly dominated -> not run through full matrix
- Two cells statistically equivalent -> retain one
- No slicing benefit at any diagnostic point -> reduce CUDA-graph slicing matrix
- S5 changes preferred active cap -> update S6 and policy-validation cells

~4,000–4,200 runs treated as adaptive campaign, not unconditional batch.

## Program totals

| Study | Runs |
|-------|-----:|
| Anchor (B0-B9 subset) | ~250 |
| S1 | 400 |
| S2 | ~750 |
| S3 | ~540 |
| S4 | ~760 |
| S5 | ~285 |
| S6 | ~754 |
| **Total** | **~3,739** |

Approximately 3,735-3,755 frozen nominal runs, with a decision-gated upper
range near 4,000 depending on selected-configuration validation, optional
stateful-concurrent S2 ablation, EP-validation subset, and diagnostic
follow-ups.

## Frozen workload and environment

All primary S1-S6 confound-isolation runs use the following frozen environment
unless otherwise specified:

```
activation shape / token batch: single-token decode, batch=1
hidden dimension H:              2048
intermediate/output dim I:       2048
rank:                            64 (current regime; R=256/512 is Exp 2/8, separate)
dtype: activation f16 -> f32 compute; A, B f32; result f16
weight layout:                   contiguous, A=[R, H], B=[R, I]
compute:                         y = x @ A.T @ B      where x=[1, H], y=[1, I]
CUDA streams:                    1 primary (S=1) unless stream sweep
server GPU:                      RTX 6000 Ada (UM251), L2 size runtime-queried
client GPU:                      A100-80GB (UM253)
CUDA / driver / PyTorch:         recorded per run (immutable metadata, below)
rdma-core:                       recorded per run
HCA / port / GID / link:         mlx5_0, HDR 200 Gbps, recorded per run
QP type:                         RC; message sizes recorded per request
CPU affinity / NUMA:             pinned and recorded per run
```

Rank is scoped to R=64 for this program. Claims do not extend to R=256/512
without the separate Exp 2/8 studies.

**Asymmetric correctness-only test:** because H=I=2048, transposition or
dimension-order errors can be masked. A separate correctness test uses
`H != I` (e.g., H=2048, I=1024) to catch incorrect layout assumptions before
the 3,000-plus-run program begins. Not part of the performance campaign; run
once per code change.

## EP regime

All primary S1-S6 confound-isolation runs use EP=0. This is intentional: the
program isolates the control path and GPU/RDMA data path without synthetic EP
interference.

> S1-S6 isolate the control path without synthetic EP interference. Coexistence
> claims require a later EP validation subset.

After Phase 3 (architecture freeze), a selected validation subset runs:
- baseline configuration at EP=0
- selected improved configuration at EP=0
- both at EP=100 (or realistic EP-layer traffic from Exp 7's generator)

This validation subset is scoped separately and does not inflate S1-S6 run
counts.

## Per-request GPU-start definition

S3 records a "GPU-kernel-start sequence," but every request launches multiple
kernels. Define the request-level sequence as:

```
request GPU-start = start of the first dtype or mm1 operation belonging to the request
```

Full per-kernel traces are retained separately. This definition is used
consistently in S3 (priority fidelity, inversion count, order preservation) and
in S2 (per-quantum `GPU_start`).

## Statistical methods

Individual requests within a run are temporally correlated by shared queues,
burst periods, GPU phases, and connection behavior. A naive request-level
bootstrap underestimates uncertainty.

Use:
- **Paired analysis** across repeated traces (same trace, different policies)
- **Trial-level confidence intervals** (treat trial as the unit for P99)
- **Block bootstrap over time windows** for long runs
- **One primary endpoint per study** declared before final runs; the rest
  labeled secondary to avoid uncorrected multi-comparison significance claims

For equivalence claims (S5 H1, S2 q=8): use TOST or paired-trial CIs against
predefined margins, not "within CI."

## Immutable run metadata

Every result row includes:

```
schema_version
git_commit
config_hash   (includes P, A, E, S, N, H, W, cell, policy, mixture, load, trial)
hostnames     (client, server)
hardware_ids  (GPU UUIDs, NIC GUIDs)
driver_version, cuda_version, pytorch_version, rdma_core_version
gpu_clocks    (recorded, not necessarily fixed)
cpu_affinity, numa_node
random_seed
trace_id
cell_id, policy_id, trial_id
start_timestamp
success_count, failure_count
```

Required because decision gates may alter later stages; metadata ensures
reproducibility and cross-phase matching (e.g., S4 all-heavy reuse by S6
requires exact config_hash match on cell/cap/trace/timing/protocol).

## Phase 3 / S6 conditionality

S6 has two stages:

```
S6-baseline:
  B3 and B5-corrected at original frozen N=8

S6-final:
  baseline plus Phase-3 selected configuration
  (e.g., if S5 selects N=4 or a different P/A, the final curve uses that)
```

If S5 changes the preferred active cap, S6-final cells differ from S6-baseline.
The S6 run count (~754) is conditional and may increase if S6-final uses a
different configuration. S4 all-heavy results can only be reused by S6 when
cell, active cap, heavy cap, trace, timing, and drain protocol match exactly
(verified via `config_hash`).

## File layout

| File | Action |
|------|--------|
| `bench_decomposition.py` | New — B0–B9 matched-contrast harness |
| `bench_first_miss.py` | Modified — dual timing domains, full accounting, all variants instrumented |
| `bench_splitting.py` | Modified — stateful client chunking, server cooperative slicing, q=8 equivalence |
| `bench_scheduling.py` | New — FIFO/Client-SJF/Server-SJF with central dispatcher |
| `bench_heavy_lane.py` | New — H sweep, Pareto, class-aware vs global-cap control |
| `bench_pool_cap.py` | New — three matched sweeps (physical-pool, active-concurrency, executor-width) |
| `bench_capacity.py` | Modified — true open-loop, three-stage admission, bracketed C, drain protocol |
| `concurrent_server.py` | Modified — persistent TCP, central dispatcher, cooperative slicing |
| `cpp/server_worker.cc` | New — C++ matched worker for B6-B9 |
| `cpp/protocol.h` | New — shared serialization schema: protocol version, byte order, struct packing/alignment, message-length framing, request-ID width — frozen identically for Python and C++ |
| `cpp/CMakeLists.txt` | New — build path for C++ matched worker |
| `tests/test_python_cpp_protocol_equivalence.py` | New — verifies equality of logical GPU operations, shapes, dtypes, stream policy, sync boundaries, allocation policy, RDMA sequence, and numerical output. Records low-level kernel sequences and flags differences, but exact kernel-name identity is NOT a pass condition (cuBLAS/framework dispatch may produce different low-level names for semantically identical ops) unless both paths intentionally call the same low-level CUDA implementation |
| `qppool.py` | Modified — physical pool / active cap separation |
| `common/instrumentation.py` | New — 19-timestamp schema, scheduler timestamps, accounting model, paired traces, immutable metadata |
| `common/load_generator.py` | New — true open-loop generator, no-op validation, drain protocol |
| `common/stats.py` | New — trial-level CIs, block bootstrap, TOST equivalence, paired-trace analysis |
| `results/decomposition/` | New output |
| `results/s1_first_miss/` | New output |
| `results/s2_splitting/` | New output |
| `results/s3_scheduling/` | New output |
| `results/s4_heavy_lane/` | New output |
| `results/s5_pool_cap/` | New output |
| `results/s6_capacity/` | New output |

If the C++ matched worker invokes different kernels or uses a materially
different allocation path, retain "runtime-stack replacement" wording rather
than "Python overhead" for B6-B9 contrasts.

## Defensible conclusions after this program

- Whether Python/TCP orchestration dominates the current prototype, and at which
  layer (transport persistence, runtime stack, active concurrency).
- Whether slicing or scheduling mitigates heterogeneous HOL blocking, and at
  which quantum/policy.
- Whether heavy-lane is class-aware or merely generic throttling.
- Whether QP count or active concurrency caused the prior degradation.
- Where each workload's real stability boundary and latency knee lie.
