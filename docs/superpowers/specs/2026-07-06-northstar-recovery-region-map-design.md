# North-Star Recovery Region Map — Design Spec

## Status

Phase 1 (N1 + N2 + N2.5, live) is implementation-specified. Phase 2 (N3–N7,
simulation-led with sparse live validation) has goals, axes, acceptance
criteria, and live-validation point counts frozen; simulator internals, exact
run counts, and specific distribution parameters are amended after Phase 1
calibration data exists. Supersedes the "North-star follow-up" section of
`2026-07-02-decomposition-confound-isolation-design.md` for the N-studies.
S1–S6 remain the remote-path mechanism authority.

## Context

The S1–S6 confound-isolation campaign established *how to build the remote
recovery path correctly*: C++ matched worker, server-side cooperative slicing,
decoupled QP provisioning/admission, bounded heavy concurrency, open-loop
capacity-based admission. The combined thesis: remote LoRA miss recovery is
not fundamentally limited by RDMA bandwidth; it is governed by software
dispatch, heterogeneous GPU work, and admission/scheduling.

That program did not establish *when* remote recovery is better than CoLoRA's
local recovery. This spec designs the next case-study layer: a measured region
map answering the north-star question.

## Goal

A measured region map answering:

> Under which workload, model, and resource conditions should a serving system
> use local recovery — `cpu_first` or `load_then_run` — versus `remote-improved`
> recovery?

Two outputs, clearly distinguished:

1. **Intrinsic recovery region map** — live N1–N2, inference removed.
2. **Deployment-adjusted region map** — sim N3–N7, calibrated by N2.5,
   validated at held-out live points.

## Guiding thesis (falsifiable, no pre-registered target numbers)

1. At low rank and low NM, `cpu_first` wins on latency because activation D2H
   + AVX compute + result H2D is shorter than RDMA round-trip + remote
   scheduling. *(N1)*
2. Above a rank- and NM-dependent crossover, `remote-improved` wins on latency
   because GPU LoRA compute dominates and per-core AVX throughput saturates.
   *(N1)*
3. We test whether `load_then_run` occupies a non-empty intermediate region:
   weights H2D is cheaper than activation round-trip, while GPU LoRA execution
   remains cheaper than AVX execution. If no statistically and practically
   distinct region exists, `load_then_run` is reported as dominated rather than
   assigned an artificial band. *(N1)*
4. Under load, `cpu_first` is bounded primarily by client CPU-worker capacity,
   host-memory bandwidth, and activation/result transfers; `load_then_run` is
   bounded by client PCIe weight transfer and local-GPU execution resources;
   `remote-improved` is bounded by remote GPU, NIC, and remote scheduling
   capacity. We expect the crossover to shift toward remote when client-side
   resources saturate earlier than the remote recovery service. *(N2)*
5. Under inference co-location, `cpu_first` additionally contends for client
   cores, DRAM bandwidth, and PCIe resources; `load_then_run` contends for
   client PCIe and inference-GPU compute or copy resources; `remote-improved`
   avoids client CPU LoRA computation and full-weight H2D, but still consumes
   client RDMA/PCIe/GPU-memory resources. We expect a further shift toward
   remote where client-side interference dominates. *(N2.5)*
6. Burstiness, EP, and multi-client sharing shift the boundary but do not
   eliminate the remote-preferred region. Absence of a remote-preferred region
   under any one interaction condition rejects that part of the thesis.
   *(N3–N5, sim)*

## Common measurement boundary (operationally precise)

- **`T0`**: the object has been classified as GPU-cold, the recovery path has
  been selected, and the required activation is ready in the inference-GPU
  input buffer. Path-selection overhead (dispatch decision, policy lookup) is
  **before** `T0`; it is not part of the measured latency.
- **`T1`**: CUDA event recorded on the **inference consumer stream** after
  that stream has waited on all path-specific completion primitives
  (copy-stream event, compute-stream event, or RDMA-completion event). `T1` is
  **not** transport completion; it is consumer-stream visibility.
- **`L_recovery = T1 − T0`** — primary reported latency, all paths.

The S1–S6 `t18` may be identified with `T1` only if the `t18` implementation
already includes NIC/GPU visibility and a consumer-stream completion event;
otherwise an outer `T1` event is added after `t18`. Same rule for `cf`/`lt`/
oracle inner endpoints.

## Five recovery paths

| Path | Adapter location | Compute | Transfer | Notes |
|------|------------------|---------|----------|-------|
| `cpu_first` | client host DRAM (resident, page-locked, NUMA-local, prefaulted) | client CPU (AVX-512 BF16, 4 pinned cores) | activation D2H; result H2D | |
| `load_then_run` | client host DRAM (resident, page-locked, NUMA-local, prefaulted) | client GPU | A/B H2D; result stays on GPU | |
| `remote-improved` | remote server HBM | remote GPU | activation RDMA READ → remote compute → result RDMA WRITE | persistent connection/QPs, pre-established buffers, non-Python per-request fast path |
| `remote-original` (ablation) | remote server HBM | remote GPU | per-request TCP + Python executor (S1–S6 B5) | diagnostic anchor points only, not full grid |
| `local-GPU oracle` (lower bound) | client GPU HBM (resident) | client GPU | none | not a competing deployment |

Distinguished memory terms (not synonyms): **resident** in host DRAM;
**page-locked/pinned** for DMA; **NUMA-local**; **prefaulted**.

## Region map semantics (two independent classifications)

- **Color — lowest-latency SLO-feasible path:** `cpu_first` / `load_then_run` /
  `remote-improved`.
- **Hatching/marker — feasibility count:** two or more paths meet the SLO.
- **Gray:** no path meets the SLO.
- **Boundary band:** where the CI of the pairwise latency difference lies
  within the practical-equivalence margin.

**Winner definition (pre-registered, mutually exclusive):**

```
Let Δ = metric(A) − metric(B), with practical margin δ.
Simultaneous CI for Δ computed via bootstrap.

A practically wins:        CI lies entirely below −δ
B practically wins:        CI lies entirely above +δ
Practically equivalent:    CI lies entirely within [−δ, +δ]
Unresolved:                all other cases
```

**Map metric by study:**
- **N1**: unloaded service-latency winner (median `L_recovery`, isolated).
- **N2 + deployment-adjusted map**: maximum sustainable arrival rate λ under
  the SLO (`C_feasible`), with latency at selected operating loads as
  supporting output.

## Axes decomposition (three groups, non-Cartesian)

### Group A — primary boundary axes

- LoRA rank R ∈ {16, 32, 64, 128, 256}
- NM (misses per logical recovery job) ∈ {1, 2, 4, 8, 16} (NM=16 = stress/
  extrapolation; typical top-k ≤ 8 per layer-projection)
- miss arrival rate λ (open-loop Poisson in N2; bracketed capacity search)
- heavy/light composition (1h9l ~10%, 1h3l ~25%, 1h1l ~50%)

Heavy class definition: `heavy = NM ≥ 4`; `light = NM = 1` (at any R;
consistent with S4/S6). Each anchor defines heavy class
`h = (R_anchor, NM_anchor)`; fixed representative light class
`l = (R=16, NM=1)` across all anchors.

### Group B — interaction axes (shift the boundary)

- burstiness: Poisson vs two-state Markov-modulated ON/OFF vs synchronized
  fan-in with common-shock mechanism
- EP interference: EP=0 vs EP=realistic (operationally defined units, frozen
  in N5)
- adapter popularity skew: affects the simulated miss stream through the
  cache/residency model and any permitted duplicate coalescing; not applied
  as a mere label distribution over an otherwise fixed independent miss stream
- multi-client sharing: 1, 2, 4 inference nodes (4 = sim extrapolation)

### Group C — deployment/resource/sensitivity axes

- CPU core budget: 4 primary; 2 and 8 sensitivity
- RDMA fabric state: idle vs contended
- remote-server sharing: dedicated (derived) vs shared (live)
- model / projection shape: one primary expert-FFN shape (H=I=2048) throughout
  N1–N6; one cross-shape held-out validation in N7 (H≠I). No dense model
  sweep.

### Non-Cartesian execution plan

- **N1** sweeps the full R × NM grid at unloaded/isolated conditions.
- **N2** uses the N1 boundary and a small set of interior anchor points, then
  performs bracketed capacity searches over λ and the frozen heavy/light
  mixtures. It does **not** repeat a dense Cartesian sweep over all four axes.
- **N2.5** measures a sparse set of inference-co-location and interference
  primitives needed to calibrate the simulator. Full burstiness, popularity,
  and multi-client composition are evaluated in N3–N5.

## Forced-cold invariant (applies to N1 and N2)

Before every measured logical request, every required expert-LoRA object is
**nonresident on the client GPU**. No object requested during the measurement
window may become a hit through reuse, deferred promotion, or residual cache
state.

```
Warm (not reset between requests):
  control-plane connections, QPs, registered buffers, CUDA contexts,
  kernels, host pages, staging buffers

Cold (invalidated before every request):
  target expert-LoRA A/B tensors on the client GPU

Cold-state preparation and cleanup occur outside [T0, T1].
```

Implementation: no-reuse object-ID stream exceeding the measurement window,
or explicit invalidation after each request, or zero-capacity recovery cache
for N1–N2.

**Path randomization:** path order randomized within paired blocks (thermal
drift, AVX downclock, GPU clocks, NIC state must not systematically favor
whichever path runs first).

## Common instrumentation

### Outer boundary (primary reported latency)

`T0` and `T1` as defined above. `L_recovery = T1 − T0`.

### Inner decomposition (per path, secondary diagnostics, single host clock)

All stage-boundary timestamps are placed in **one host monotonic clock**
(`CLOCK_MONOTONIC_RAW`) via host callbacks/completion-polling. CUDA events
independently measure device-operation durations and are **not** inserted into
the host-domain additive identity.

```
Host-domain stage intervals sum to:
  L_recovery − instrumentation_gap

CUDA-event durations are reported as separate diagnostics
(D2H engine, H2D engine, GPU compute kernel, RDMA engine).
```

`instrumentation_gap` = `L_recovery` − sum(non-overlapping host-domain stage
intervals). Flag if `instrumentation_gap_fraction > 5%` and
`instrumentation_gap_us > 50µs` (S1–S6 threshold retained).

**`cpu_first`:**
```
cf0  activation pack start
cf1  activation D2H enqueue
cf2  D2H observed complete (host poll)
cf3  AVX compute start (CPU worker)
cf4  AVX compute complete
cf5  result H2D enqueue
cf6  H2D observed complete (host poll)
cf7  consumer-stream merge/visibility → T1
```

**`load_then_run`:**
```
lt0  A/B H2D enqueue
lt1  H2D observed complete (host poll)
lt2  GPU compute enqueue
lt3  GPU compute observed complete (host poll / event)
lt4  consumer-stream merge/visibility → T1
```

**`remote-improved` / `remote-original`:** S1–S6 19-timestamp schema retained
as inner decomposition; `T0` aligns with `t1`, `T1` is the outer
consumer-stream event (added after `t18` if `t18` lacks consumer-stream
visibility).

**`local-GPU oracle`:** `T0` → grouped-GEMM launch → GPU compute complete →
result cast → merge into output buffer → consumer-stream completion event →
`T1`. Includes launch overhead and merge; not a bare arithmetic kernel.

### Resource counters (separate replay, not primary timing)

Primary timing runs use minimal instrumentation (only the timestamps above).
Counter runs are **separately replayed** at the same cell/seed with profiling
enabled; they are not used for primary latency.

| Counter family | Tool | Sampling | Scope |
|---|---|---|---|
| CPU utilization, context switches | `perf stat` | 100ms | 4 pinned cores |
| Host DRAM bandwidth, NUMA remote | `perf stat` (uncore) | 100ms | socket |
| Page faults | `perf stat` | run-total | process |
| PCIe bytes, copy-engine | nvidia-smi dmon / CUPTI | 10ms | GPU |
| GPU SM/mem utilization | nvidia-smi dmon | 10ms | GPU |
| NIC bytes, RDMA ops | mlx5 counters (`perfquery`) | 1s | port |
| Remote GPU concurrency | server-side instrumentation | per-event | server |

**Perturbation validation:** counter-enabled median latency must be within 5%
of primary median; else the counter run is discarded and rerun with lower
sampling frequency.

**Page-fault invariant:** major faults = 0 in the measured interval; minor
faults ≤ frozen tolerance (10/request). Violation invalidates the run.

## Frozen environment (N1–N2)

```
B_decode = 1                    (single-token decode; one inference request, one token)
NM       = misses per logical recovery job (distinct cold expert-LoRA objects)
N_req    = simultaneously outstanding logical recovery requests (varies in N2)
N_rows   = NM                   (activation rows processed by one job)

workload:                 single-token decode, B_decode=1
activation shape:         x = [NM, H], y = [NM, I]
hidden dim H:             2048
output dim I:             2048  (primary expert-FFN shape; H≠I held-out in N7)
rank sweep:               R ∈ {16, 32, 64, 128, 256}
NM sweep:                 NM ∈ {1, 2, 4, 8, 16}
                          (NM=16 = stress/extrapolation)
heavy/light:              heavy = NM ≥ 4; light = NM = 1
heavy/light composition:  1h9l, 1h3l, 1h1l — reuse S4/S6
EP:                       0 (N1–N2); EP=realistic enters in N5
server GPU:               RTX 6000 Ada (UM251)
client GPU:               A100-80GB (UM253)
RDMA:                     mlx5_0, HDR 200 Gbps, RC QP
```

### Batching and coalescing semantics (frozen)

- **`B_decode = 1`**: one inference request, one token, per model step.
- **`NM` = distinct cold expert-LoRA objects in one recovery job.** Each
  object has its own `(A_i, B_i)`. The activation `x = [NM, H]` has one row
  per object; rows are **not** interchangeable.
- **No dynamic cross-request coalescing in N1–N2.** NM is fixed per logical
  request by the load generator. Dynamic coalescing across independent
  requests is a scheduler-design knob evaluated in N3+.

Per-path transfer/compute semantics:

**`cpu_first`:** one `[NM, H]` activation D2H → AVX compute on `[NM, H]` with
per-object `(A_i, B_i)` → one `[NM, I]` result H2D. Pack/merge included in
`cf0`/`cf7`.

**`load_then_run`:** the NM missed objects have **distinct weights**
`A_i ∈ R^{R×H}`, `B_i ∈ R^{R×I}`. Weights transferred as contiguous
`A=[NM,R,H]` and `B=[NM,R,I]` buffers (or same-stream sequence), followed by
grouped/batched LoRA GEMMs. Weight payload =
`NM × R × (H+I) × sizeof(weight_dtype)`. No padding to `max(H,I)` in reported
bytes unless the implementation actually transfers padding.

**`remote-improved`:** one RDMA READ for `[NM, H]` activation → remote GPU
grouped GEMM → one RDMA WRITE for `[NM, I]` result. **Logical** RDMA op count
= 1 READ + 1 WRITE; server slicing with `q=4` may create multiple GPU compute
slices but must not create additional wire operations. Actual wire op count
reported per request.

### Dtype/kernel freeze (per path, reconciled)

S1–S6 `A,B f32` is a deviation: the production CPU kernel is AVX-512 BF16.
The north-star freezes A/B = BF16 across all paths for comparability.

| Path | A/B storage | A/B transfer | Kernel input | Accum | Output |
|------|-------------|--------------|--------------|-------|--------|
| `cpu_first` | BF16 (host) | — (host-local) | BF16 | FP32 | FP16 |
| `load_then_run` | BF16 (host) | BF16 (H2D) | BF16 | FP32 | FP16 |
| `remote-improved` | BF16 (remote GPU) | — (GPU-local) | BF16 | FP32 | FP16 |
| `oracle` | BF16 (client GPU) | — | BF16 | FP32 | FP16 |
| activation (all) | FP16 | FP16 | FP16/BF16 | — | — |

**Correctness criterion:** max absolute error vs FP32 reference ≤ 1e-3 (ATOL)
and max relative error ≤ 1e-2 (RTOL), per element, checked on every cell
during smoke testing.

### CPU recovery resource freeze (frozen, not merely recorded)

```
CPU model/stepping:        recorded, frozen per host
physical core IDs:         4 pinned physical cores (not SMT siblings)
SMT:                       disabled for selected cores, or sibling placement controlled
NUMA node:                 local to inference GPU PCIe root
governor:                  performance
turbo policy:              frozen (EPP=performance or disabled; recorded)
achieved per-core freq:    recorded per run; must stay within 5% of frozen target
AVX frequency downclock:   recorded (AVX-512 heavy downclock on some SKUs)
thread-library env:        OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1
                            (for one-core-per-request; custom kernel bypasses these)
memory channels:           all populated, recorded
```

**Thread-per-request policy (primary):** 1 core per request → up to 4
concurrent `cpu_first` requests. Each request uses exactly 1 core; no
intra-request multithreading.

**Sensitivity:**
- 2 cores (concurrency 2), 8 cores (concurrency 8) — core-budget sensitivity
- 4-cores-all-per-request (concurrency 1) — latency-throughput tradeoff at
  fixed budget

Core count not selected retrospectively.

### Host residency and pinning semantics

**Primary configuration:** full active-population host residency — every
`A_i/B_i` tensor for every active expert-LoRA object is host-resident,
page-locked, NUMA-local, and prefaulted. Staging buffers (preallocated pinned
tensors, reused across requests) are separate objects, also pinned and
NUMA-local.

**Cold-page sensitivity** (first-touch / cold-page behavior) is a separate
N3+ study, not the primary comparison.

### Remote-improved freeze (from S1–S6 Phase-3 architecture)

```
control transport:    persistent multiplexed TCP, request-ID matching
handler runtime:      C++ matched worker (S1 B6)
scheduling:           FIFO, central dispatcher (S3)
server slicing:       cooperative, quantum q=4 (S2)
heavy-lane:           H=2 heavy sub-cap (S4)
QP pool:              P=64 physical, A=8 active cap (S5)
admission:            open-loop, three-stage (S6)
```

### SLO: self-normalized stability vs common deployment (separated)

**Self-normalized stability SLO** (per-path capacity / queueing-collapse
detection):
```
P(L_recovery > 2 × path's own isolated median) ≤ 0.01
```
Used to define per-path capacity.

**Common deployment SLO** (region-map feasibility, cross-path):
```
P(L_recovery > SLO_common) ≤ 0.01
where SLO_common = max(10 ms, 5 × oracle isolated median)
```
The oracle anchor adapts the threshold to cell difficulty; the 10ms floor
prevents trivially loose SLOs. Factor 5 and floor 10ms frozen before final
runs. A path is **region-map feasible** if it meets the common SLO.

**Region-map semantics:**
- **Color:** lowest-latency region-map-feasible path.
- **Hatching:** two or more paths region-map-feasible.
- **Gray:** no path region-map-feasible.
- **Capacity (N2 map metric):** highest λ where the path is both
  self-normalized-stable and region-map-feasible.

### N6 memory accounting

Renamed axis: **client-host DRAM avoided** (not "local CPU memory saved").
Report both:
```
ΔM_client-host   = host DRAM freed on the inference node by using remote
ΔM_cluster-total = ΔM_client-host − remote_host_bytes − remote_gpu_bytes
                   (negative if remote replication exceeds local savings)
```

Components recorded per cell: `client_host_bytes`, `remote_host_bytes`,
`remote_gpu_bytes`, `replication_factor`, `network-side staging bytes`.

### TOST practical-equivalence margin (deterministic rule)

```
δ_{R,NM} = max(δ_absolute, ρ × calibration_median_{R,NM})
δ_absolute = 50 µs  (S1–S6 no-op control-message threshold)
ρ = 0.10  (10% of calibration median)
calibration split = 60% of N1 trials for calibration, 40% held out
```

`ρ`, `δ_absolute`, and the calibration split are frozen before final runs.

## N1 — Isolated live crossover

### Purpose

Establish the intrinsic service-time crossover without queueing or
interference. For an isolated miss, when does local recovery beat remote?

### Primary endpoint

Paired difference in trial-level **median** `L_recovery`.

### Cells

Full R × NM grid at isolated single-request conditions.
```
R  ∈ {16, 32, 64, 128, 256}   (5 values)
NM ∈ {1, 2, 4, 8, 16}         (5 values; NM=16 = stress/extrapolation)
→ 25 cells
```

### Paths

`cpu_first`, `load_then_run`, `remote-improved`, `oracle` — all 25 cells.
`remote-original` at 3 anchor points: (16,1), (64,8), (256,8).

### Methodology

- `B_decode = 1`, `N_req = 1` — one isolated request at a time, no queueing
- Pre-established normal control paths; no EP; no inference-engine background
  work
- Fixed CPU budget: 4 pinned cores, 1 core per request
- Paired adapter tensors, activations, and seeds across all paths
- Numerical-output verification per cell (ATOL 1e-3, RTOL 1e-2 vs FP32
  reference)
- Counter replay runs separately (not primary timing)
- Forced-cold invariant enforced
- Path order randomized within paired blocks

### Per-trial sampling

≥1,000 measured requests per trial; 2,000 preferred for cells used in P99
claims. 5 trials.

### Inferential vs descriptive metrics

- Primary: median (inferential)
- Secondary inferential: P95
- Secondary (adequate samples only): P99
- Descriptive only, never used for winner classification: max

### Winner classification (three-way, paired, per cell)

```
A practically wins:        CI for Δ = median(A) − median(B) lies below −δ
B practically wins:        CI lies above +δ
Practically equivalent:    CI lies entirely within [−δ, +δ]
Unresolved:                all other cases
```

A contiguous equivalence/unresolved band near the crossover is expected and
scientifically useful.

### CPU sensitivity (per-request parallelism, not worker-pool)

6 representative (R,NM) points × 3 per-request thread counts {1, 2, 4} × 5
trials. `N_req=1`; each request uses the declared thread count on the 4-core
budget. Worker-pool capacity (multiple in-flight requests) is an N2
sensitivity.

### Run count

- Primary: 25 × 4 × 5 = 500 trials
- `remote-original`: 3 × 5 = 15 trials
- Per-request threading: 6 × 3 × 5 = 90 trials
- Counter replays: ~100
- **N1 total: ~705 trials** (requests/trial increased, not trial count)

### Outputs

- Crossover curves: median `L_recovery` vs R (per NM), vs NM (per R)
- R × NM winner grid: three-way classification per cell
- Stage decomposition (secondary)

## N2 — Live loaded anchor map

### Purpose

Under open-loop load, measure per-path capacity, P99 cliff, and the
CPU-vs-remote winner. The principal measured intrinsic region map.

### Primary endpoint

`C_feasible` — highest λ satisfying both self-normalized stability and common
region-map SLO.

### Scope

With 6–8 anchor points, N2 is the **live loaded anchor map**, not the complete
principal region map. The complete region map is the N3–N7 calibrated
simulation output. Any interpolation between anchors is clearly labeled as
model-derived.

### Structured-anchor geometry (along fixed NM slices)

```
NM=1:  3 R values spanning the crossover
NM=8:  3 R values spanning the crossover
NM=16: 1–2 stress points (reported separately)
→ 7–8 anchors
```

### Anchor selection rule (deterministic)

- Forced low/low and high/high corners
- 2 points with signed N1 median difference nearest zero (boundary)
- 2 points with margin > 2×δ
- 1–2 space-filling points reserved from boundary fitting

### Heavy/light class definition

```
Each anchor defines heavy class h = (R_anchor, NM_anchor).
Fixed representative light class l = (R=16, NM=1) across all anchors.

1h9l: λ_h = 0.10λ, λ_l = 0.90λ
1h3l: λ_h = 0.25λ, λ_l = 0.75λ
1h1l: λ_h = 0.50λ, λ_l = 0.50λ
```

Latency feasibility must hold **separately for both classes**; capacity fails
if either class violates its SLO.

### Trial duration and sample-count rules

```
Each load trial:
  - fixed warm-up interval
  - steady-state measurement interval
  - ≥ N_min completed requests per class
    (target: several thousand for per-class P99)
  - arrival shutdown + complete drain
  - continue until bootstrap CI width for class P99 < frozen
    relative threshold (15% of point estimate), or max duration (120s generation)
```

For 1h9l: total run must be ~10× longer than 1h1l for the same heavy-class
sample count.

### Capacity search (bracketed, confidence-based)

```
C_lower:  highest λ confirmed feasible by frozen confidence rule
C_upper:  lowest λ confirmed infeasible
Stop when C_upper / C_lower ≤ 1.10

Repeated trials at discordant points; isotonic regression as diagnostic.
Report capacity as bracket [C_lower, C_upper] if bracket width > 1.10.
```

Two capacities reported:
```
C_stable:   highest λ satisfying open-loop stability criteria
C_feasible: highest λ satisfying stability + common region-map SLO
```

### Two winner outputs

- **Capacity winner at anchor:** path with highest `C_feasible`
- **Operational winner at particular λ:** feasible path with lowest P99
  `L_recovery`

### Separate encodings (not conflated)

- Multiple paths SLO-feasible
- Practically equivalent P99 (TOST)
- Unresolved comparison

### Paths

`cpu_first`, `load_then_run`, `remote-improved` (all anchors). `oracle` (2
anchors). `remote-original` (2 anchors).

### Worker-pool capacity sensitivity (moved from N1)

At 2 boundary anchors, sweep total one-core workers `K ∈ {2, 4, 8}` for
`cpu_first`. Multiple requests in flight.

### Mixtures

Primary 1h3l (all anchors); Secondary 1h9l, 1h1l (3-anchor subset).

### Run count (adaptive, range until anchor count frozen)

- Primary (6–8 anchors × 3 paths × 1 mixture × ~8 loads × 5 trials):
  720–960
- Secondary (3 anchors × 3 paths × 2 mixtures × ~5 loads × 5 trials): ~450
- Oracle (2 × 3 × 5): ~30
- `remote-original` (2 × 5 × 5): ~50
- Worker-pool sensitivity (2 × 3 × 5 × 5): ~150
- Counter replays: ~100
- **N2 total: ~1,500–1,800 trials** (adaptive)

### Outputs

- Capacity curves: `C_feasible` vs R (per NM slice), one curve per path, one
  panel per mixture
- R × λ anchor map: color = capacity winner; hatching = multiple feasible;
  gray = none
- P99 vs λ curves at each anchor (overlay all paths)
- Mixture-interaction summary

## N2.5 — Live co-location calibration

### Purpose

Calibrate the inference-interference model. Measure recovery and inference
metrics when a real inference engine is co-located on the client node.

### Primary endpoint

Incremental P99 TPOT relative to paired inference-only baseline.
```
Δ P99 TPOT_path = P99 TPOT_path − P99 TPOT_no-recovery
```

### Operating points (complete definition)

Each point = `(R, NM, λ_recovery, mixture, inference_intensity)`.
```
R, NM:            4 points from N2 (CPU-preferred, below crossover,
                  above crossover, remote-preferred)
λ_recovery:       common absolute rate across paths
                  = 0.7 × min(C_cpu, C_remote) from N2 results
mixture:          1h3l primary
inference_intensity: 3 levels (below)
```

The same absolute recovery arrival trace is used across paths. Path-specific
fractions of capacity are **not** compared.

### Inference intensity (operationally defined)

```
Low:        30% of inference-only stable capacity
Moderate:   60%
Near-knee:  90%, or highest preserving frozen inference SLO
```

Use offered arrival rate (requests/s), not just concurrent-sequence count.
Active sequence count recorded as outcome.

### Inference-only knee characterization (preliminary sweep)

Small capacity sweep without recovery to find inference-only stable
capacity. Included in run budget.

### Inference-only baseline (every cell)

4 points × 3 intensities × 5 trials = 60 additional runs. Allows Δ metrics.
Oracle is **not** a substitute (it still performs LoRA GPU work and consumes
GPU resources).

### Miss injection (token-coupled, not background)

```
Each injected miss is attached to an actual decode token and routed
expert-LoRA access.

The faulting request cannot advance beyond the affected layer until T1.
Other GPU-ready requests continue per real scheduler.
No synthetic recovery job is counted unless its result is consumed by
the corresponding inference request.

Paired comparisons: replay identical exogenous request arrivals,
adapter IDs, and pre-recorded routing decisions.
```

### Inference metrics definitions (frozen)

```
TPOT:        time per output token, excluding first output token (decode-only)
TTFT:        time to first decode output, excludes prefill (secondary)
Request P99: P99 of per-request TPOT across completed requests
Inference SLO: P99 TPOT ≤ 2× inference-only baseline (frozen)
```

### Paths

`cpu_first`, `remote-improved` (all points). `load_then_run` (2 boundary
points). `oracle` (1 point).

### Calibration/validation split (all 12 cells accounted)

```
Calibration:
  3 operating points × {low, moderate} = 6 cells

Held-out validation:
  Spatial:         4th operating point × {low, moderate} = 2 cells
  Intensity extrap: all 4 × {near-knee}                  = 4 cells
  Total validation = 6 cells
```

No simulator tuning after viewing validation outcomes without declaring a
new split.

### Winner validation (equivalence-aware)

- If live `cpu_first` and `remote-improved` practically equivalent: either
  predicted path acceptable only if predicted difference within margin.
- If live data establishes unique winner: simulator must predict that winner.

### Error criteria (combined absolute and relative)

```
|ŷ − y| ≤ max(ε_abs, ε_rel × |y|)

Recovery P99:  ε_abs = 2 ms,   ε_rel = 0.20
TPOT:          ε_abs = 5 ms,   ε_rel = 0.15
Throughput:    ε_abs = 50 req/s, ε_rel = 0.10
Boundary shift within one sampled grid step
```

Failure → expand live coverage, declare new validation split.

### Run count

- Primary: 4 × 3 × 2 × 5 = 120
- `load_then_run` boundary: 2 × 2 × 5 = 20
- Oracle: 1 × 2 × 5 = 10
- Inference-only baseline: 4 × 3 × 5 = 60
- Inference-knee characterization: ~30
- Counter replays: ~30
- **N2.5 total: ~270 trials**

## Phase 2 — Simulator components

Four materially different submodels, each with its own calibration source and
acceptance criteria:

| Component | Calibrated by | Required by | Component-level acceptance |
|-----------|--------------|-------------|---------------------------|
| **C1: Single-client service + interference** | N1, N2, N2.5 | N3–N5 baseline | Recovery P99, TPOT, throughput (Chunk 3 criteria) |
| **C2: Cache/residency + duplicate-coalescing** | N3 live validation | N3 | Miss-rate error, coalescing-rate error, reuse-distance distribution error, recovery-arrival-rate error |
| **C3: Shared remote-server queueing** | N4 live validation | N4 | Per-client queue wait, server active-concurrency distribution, fairness across clients, aggregate throughput |
| **C4: EP/network coexistence** | N5 live validation | N5 | Recovery traffic delay, EP traffic delay, NIC utilization, throughput/TPOT shift |

C1 is calibrated by Phase 1. C2–C4 require their own live validation; C1
calibration alone does not establish them.

### Simulator locking and failure handling

```
Before any N3/N4/N5 live-validation result is inspected:
  - simulator code version is locked
  - parameters are locked
  - validation cells and seeds are locked
  - predicted metrics and winners are written to an immutable prediction file

If acceptance fails:
  - failed cells moved into a new calibration set
  - simulator version incremented
  - new, previously unseen validation cells selected
  - same cells cannot be reused to claim validation of the amended model
```

### Three-stage data split (prevents N7 validation leakage)

```
A. Simulator calibration:
   N1/N2/N2.5 calibration cells.

B. Simulator validation:
   designated N2.5 and N3–N5 validation cells.

C. Final deployment-rule test:
   live cells sealed until both the simulator AND the N7 rule are frozen.
   2 final blind configurations × 5 trials = 10 new trials.
```

N3–N5 validation cells used for simulator acceptance cannot also be used for
N7. Cells reserved for N7 are sealed before N3–N5 validation begins.

## N3 — Burstiness and popularity

### Purpose

Quantify how burstiness and adapter popularity shift the local-versus-remote
recovery boundary.

### Axes (frozen, both live-validated)

- Burstiness: Poisson (baseline) vs two-state Markov-modulated ON/OFF vs
  synchronized fan-in with common-shock mechanism
- Adapter popularity: uniform vs Zipf (one representative parameter live;
  additional parameters simulation-only)
- Boundary-adjacent points from N2

### Distribution families frozen now

Poisson baseline; two-state Markov-modulated or ON/OFF burst process;
synchronized fan-in with defined common-shock mechanism. Parameters deferred
to Phase 1 completion.

### Controlled across burstiness comparisons

Mean offered recovery rate, heavy/light composition, per-object popularity,
total request count. Otherwise "burstiness" may partly be an average-load
change.

### Live validation

2 boundary points × 2 burstiness levels × 2 popularity levels × 5 trials =
**40 trials**

### Acceptance criteria (uncertainty-aware)

C2 component-level: miss-rate error, coalescing-rate error, reuse-distance
distribution error. Plus C1 criteria.

## N4 — Multi-client scaling

### Purpose

Determine how client count, server sharing, and load imbalance shift the
local-versus-remote recovery boundary.

### Axes (frozen)

- Client count: 1, 2, 4 (4 = simulation extrapolation only)
- Server sharing: shared (live-validated) vs dedicated (derived from
  replicated single-client, labeled "derived dedicated baseline")
- Load allocation: symmetric vs asymmetric

### Fairness metrics

Per-client P99, per-client throughput, Jain fairness index, starvation/
rejection rate, server queue occupancy.

### Live validation

1-client baseline (5) + 2-client shared symmetric (5) + 2-client shared
asymmetric (5) = **15 trials**. Four-client behavior is simulation
extrapolation, not live-validated.

### Acceptance criteria (uncertainty-aware)

C3 component-level: per-client queue wait, server active-concurrency
distribution, fairness across clients, aggregate throughput.

## N5 — EP and network coexistence

### Purpose

Quantify how EP traffic and fabric contention shift the region boundary.

### Axes (frozen, factorial)

- EP state: EP=0 (idle fabric) vs EP=realistic (contended fabric)
- If independent fabric contention is tested: separate factor → factorial
  design

### EP operational definition (frozen units)

```
EP bytes/s; messages/s; average and peak active RDMA operations;
message-size distribution; burst duration; fraction of NIC bandwidth;
shared or separate QPs/traffic classes.
```

### Live validation

- Collapsed axes (EP state = fabric state): 2 points × 2 EP levels × 5 =
  **20 trials**
- Factorial (independent contention): 2 × 2 × 2 × 5 = **40 trials**

Choice frozen before N5 execution based on whether independent fabric
contention is achievable on the testbed.

### Acceptance criteria (uncertainty-aware)

C4 component-level: recovery traffic delay, EP traffic delay, NIC
utilization, throughput/TPOT shift.

## N6 — Resource/cost Pareto

### Purpose

At representative workloads, report multi-objective Pareto: P99, throughput,
client-host DRAM avoided, PCIe bytes, network bytes, remote-GPU utilization,
cluster-total memory.

### Axes (frozen)

- 2–3 representative workloads from N2 anchor map (CPU-preferred, boundary,
  remote-preferred)
- Resource accounting: measured from Phase 1 counter replays
- Cost model: frozen before N6 execution

### Uncertainty propagation

Bootstrap measured inputs, recompute frontier. Classify points:
- **Definitely dominated** (CI excludes non-dominated region)
- **Possibly dominated** (CI crosses boundary)
- **Non-dominated with confidence** (CI excludes dominated region)

### Point provenance labels

Live measured / derived from counters / simulator predicted / cost-model
transformed.

### Cost model frozen before N6 execution

```
pricing date and region; reserved/on-demand assumptions;
GPU allocation fraction; remote-server sharing factor;
DRAM and network charges; depreciation horizon for owned hardware;
utilization assumption.
```

Cloud list price vs deployment-specific cost = **separate scenarios**, not
interchangeable alternatives.

## N7 — Derived deployment rule

### Three-stage split

Calibration (A) / simulator validation (B) / final deployment-rule test (C).
C is sealed until both simulator and N7 rule are frozen.

### Rule-selection procedure (frozen now)

```
Candidate families:
  1. axis-aligned threshold table
  2. decision tree, depth ≤ 3
  (logistic regression permitted as candidate only if thresholding
   procedure is frozen)

Selection criterion:
  lowest cross-validated decision regret
  ties resolved in favor of the simpler model

Complexity limit:
  depth ≤ 3; ≤ 8 leaf rules

Feature transformations:
  frozen before fitting (no post-hoc feature engineering)

Cross-validation:
  5-fold on calibration + simulator-validation cells (A ∪ B)
  one-time evaluation on sealed test set (C)
```

### Abstention region

```
If predicted path differences lie inside the equivalence margin
or uncertainty crosses the boundary:
    return "either / benchmark locally / adaptive choice"
```

### Reported metrics

- Classification accuracy
- Equivalence-aware accuracy
- Abstention coverage
- Decision regret (latency/resource cost of wrong choice)
- Unsafe-error rate (selection of an SLO-infeasible path)

### H≠I held-out validation (restored)

```
Hold out 2–3 realistic expert-FFN projection shapes with H ≠ I.
Sealed until both simulator and N7 rule are frozen.
Evaluate: predicted winner, P99 error, boundary error, numerical correctness.
```

### Final test

2 final blind configurations × 5 trials = 10 new trials.

## Statistical methods

### Primary endpoint and multiplicity

Each study has one endpoint type designated as primary. Because that endpoint
is tested across multiple cells and path contrasts, the frozen family-wise
correction applies to **all confirmatory cell-level claims**.

### Classification rule (mutually exclusive, simultaneous-CI based)

```
Let Δ = metric(A) − metric(B), with practical margin δ.
Simultaneous CI for Δ computed via bootstrap.

A practically wins:        CI lies entirely below −δ
B practically wins:        CI lies entirely above +δ
Practically equivalent:    CI lies entirely within [−δ, +δ]
Unresolved:                all other cases
```

Mutually exclusive and interpretable.

### Multiple-comparison families (formulas)

```
N1:  25 cells × 3 primary path pairs = 75
N2:  n_anchors × 3 primary path pairs
N2.5: n_confirmatory_cells × declared path pairs
N3:  n_validation_cells × 2 path pairs
N4:  n_validation_cells × 2 path pairs
N5:  n_validation_cells × 2 path pairs
N7:  n_sealed_test_cells × 2 path pairs
```

`load_then_run` boundary cells in N2.5 either included in the N2.5 family or
explicitly labeled exploratory.

Holm-Bonferroni step-down within each family (α = 0.05). Simultaneous
bootstrap intervals (max-statistic method) for capacity brackets.

### Full adaptive capacity bootstrap (N2)

```
For each bootstrap replicate:
  1. resample paired trials/blocks (hierarchical: trials first, then blocks)
  2. recompute feasibility at every sampled λ
  3. reapply monotonicity handling (isotonic regression as diagnostic)
  4. recompute C_lower and C_upper
  5. derive path-capacity contrasts
```

Record all attempted λ values; do not interpolate silently across unsampled
gaps.

### Block-length selection (justified, not fixed)

```
Estimate correlation scale from pilot/calibration runs.
Freeze block-length selection rule before final runs:
  - moving-block or stationary bootstrap
  - block length from integrated autocorrelation time
  - require ≥ 20–30 effective blocks per trial
Preserve pairing: same block indices resampled jointly across paths.
Hierarchical paired bootstrap: trials first, then temporal blocks within
trials.
```

### Uncertainty-aware acceptance criteria

```
Pass:          entire CI for prediction error within [−tolerance, +tolerance]
Inconclusive:  point estimate within tolerance, CI crosses tolerance
Fail:          point estimate outside tolerance or CI excludes acceptable range
```

Applied to: recovery P99, TPOT, throughput, winner classification, boundary
shift.

### "One grid step" definition

```
For R × NM boundary:
  one step = one adjacent value in the ordered R set or NM set.

For 1D fixed-NM slice:
  one step = one adjacent R value.

For irregular anchor points:
  no grid-step claims; report nearest-anchor disagreement
  or continuous-coordinate error instead.
```

## Program execution order (decision-gated)

```
Phase 1a: N1 (isolated crossover)
           ↓ N1 gate
Phase 1b: N2 (loaded anchor map) — anchors from N1
           ↓ N2 gate
Phase 1c: N2.5 (co-location calibration) — operating points from N2
           ↓ N2.5 gate (C1 simulator acceptance)
Phase 1d: Phase 1 calibration complete; C2–C4 internals amended
           ↓
Phase 2a: N3 (burstiness/popularity) — C2 calibrated + validated
Phase 2b: N4 (multi-client) — C3 calibrated + validated
Phase 2c: N5 (EP coexistence) — C4 calibrated + validated
Phase 2d: N6 (Pareto, derived from measured + sim)
Phase 2e: N7 (deployment rule, sealed test set C)
```

### Decision gates

**N1 gate:** For every non-stress cell, paired primary comparison classified
(unique winner / practically equivalent / unresolved). Unresolved cells
confined to a contiguous crossover band; else additional trials. NM=16
reported separately as stress/extrapolation.

**N2 gate:** Every reported capacity has confirmed feasible and infeasible
endpoints; frozen bracket-width criterion (`C_upper/C_lower ≤ 1.10`) met;
class-level sample requirements and stability checks pass. Otherwise report
capacity interval rather than point.

**N2.5 gate:** Simulator passes all held-out validation cells under
uncertainty-aware tolerances. Unique live winners predicted correctly;
live-equivalent cells predicted as equivalent or within frozen practical
margin. Failure triggers expanded live calibration and newly declared
validation split.

## File layout

```
configs/
  n1.yaml, n2.yaml, n2_5.yaml
  n3.yaml, n4.yaml, n5.yaml, n6.yaml, n7.yaml

schemas/
  request_record.schema.json
  trial_summary.schema.json
  simulator_prediction.schema.json

bench_northstar_crossover.py        N1 driver
bench_northstar_loaded.py           N2 driver
bench_northstar_colocation.py       N2.5 driver

common/forced_cold.py               forced-cold invariant enforcement
common/miss_injection.py            token-coupled miss injection (N2.5)
common/inference_baseline.py        inference-only knee characterization
common/path_randomization.py        paired-block path randomization

sim/
  calibrate.py                      C1–C4 calibration
  validate.py                       locked-prediction validation
  version.py                        simulator versioning + locking
  northstar_simulator.py            Phase 2 simulator
  interference_model.py             C1: single-client service + interference
  cache_residency.py                C2: cache/residency + coalescing
  multi_client.py                   C3: shared server queueing
  ep_coexistence.py                 C4: EP/network coexistence
  pareto.py                         N6 resource/cost Pareto
  deployment_rule.py                N7 derived rule

analysis/
  analyze_n1.py                     crossover curves + winner grid
  analyze_n2.py                     capacity brackets + region map
  analyze_n2_5.py                   interference calibration
  analyze_phase2.py                 N3–N7 sim + validation

tests/
  test_forced_cold.py
  test_timestamp_accounting.py
  test_capacity_search.py
  test_simulator_replay.py
  test_stats_classification.py

results/
  n1_crossover/
  n2_loaded/
  n2_5_colocation/
  n3_burstiness/
  n4_multi_client/
  n5_ep/
  n6_pareto/
  n7_rule/
  sim/
    calibration_inputs/
    locked_predictions/
    live_validation/
    post_validation_amendments/
```

### Run-directory structure (immutable manifest)

```
raw/          # collected data only
derived/      # analysis outputs
figures/
logs/
manifest.json
```

```
manifest.json:
  run_id; spec_version; git_commit; host_identifiers;
  hardware_topology; driver/runtime_versions; config_hash;
  random_seeds; start/end_timestamps; clock_domain_metadata;
  path_order; validation/calibration_role
```

Benchmark drivers collect data only. Analysis scripts generate capacity
estimates, significance classifications, and figures from immutable raw
results.

Reuse from S1–S6: `common/instrumentation.py` (19-timestamp schema, extended),
`common/load_generator.py` (open-loop generator, extended for forced-cold),
`common/stats.py` (trial CIs, TOST, block bootstrap, Holm correction),
`qppool.py` (P/A separation), `cpp/server_worker.cc` (C++ matched worker),
`cpp/protocol.h`.

## Defensible conclusions after this program

- **N1:** The intrinsic service-time crossover between local recovery
  (`cpu_first`, `load_then_run`) and `remote-improved`, as a function of
  (R, NM), without queueing.
- **N2:** The loaded capacity crossover **at the structured live anchors and
  mixtures tested in N2**, with interpolation outside those anchors clearly
  labeled model-derived.
- **N2.5 + N3–N5:** The deployment-adjusted crossover, distinguishing:
  - live-validated points
  - simulated region between those points
  - extrapolated regions (e.g., four-client operation)
- **N6:** Pareto frontiers **under the frozen resource-accounting and
  cost-model assumptions**.
- **N7:** An interpretable decision rule **within the frozen workload,
  hardware, and model-shape domain**, including an abstention region near
  uncertain boundaries.

## Explicit non-claims

- No claim about R > 256 without separate studies.
- **Decode batch clarification:** Each inference sequence contributes one
  token per decode iteration. N1/N2 recovery microbenchmarks use one logical
  inference request at a time unless explicitly loaded through `N_req`. N2.5
  may use engine-level continuous batching across multiple sequences. No
  claim about multi-token-per-sequence decode or recovery-job coalescing
  beyond the frozen `N_rows`/NM semantics.
- No claim about models other than the primary expert-FFN shape; N7 H≠I cells
  test **limited shape generalization within the frozen runtime and
  hardware**.
- No claim that the simulated region map is a measurement — it is a
  simulation prediction validated at sparse live points.
- No claim about absolute cost ($/req) without a frozen cost model.
- No causal attribution of co-location interference to specific resources
  without dedicated N2.5 ablations.
- No extrapolation beyond the frozen CPU core budget (4 cores primary; 2/8
  sensitivity).
- **No cross-model or cross-hardware claim.**

## Program totals

### Phase 1 live

```
N1:   ~705 trials   (≥1,000–2,000 requests/trial)
N2:   ~1,500–1,800  (adaptive bracketed search)
N2.5: ~270          = 150 path trials + 60 inference-only baseline
                     + 30 inference-knee characterization + 30 counter replays
─────────────────────
Phase 1: ~2,475–2,775 live trials
```

### Phase 2 live validation (range depending on axis coverage)

```
N3: 40 (both axes) or 20 (burstiness only)
N4: 15
N5: 20 (collapsed) or 40 (factorial)
N6: 0 (derived)
N7: 10 (new blind) or 0 (sealed subset of prior cells)
─────────────────────
Phase 2: ~55–105 live trials
```

### Total live

~2,530–2,880 trials (adaptive; Phase 2 range depends on frozen axis coverage).

Phase 2 simulation run counts amended after Phase 1 calibration; not included
in live total.
