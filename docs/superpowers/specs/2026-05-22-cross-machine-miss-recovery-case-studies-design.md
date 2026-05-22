# Cross-Machine Miss-Recovery Case Studies (A, B, D) — Design Spec

Date: 2026-05-22
Status: Awaiting user review

## 1. Purpose and Scope

This spec defines three analytical case studies that extend CoLoRA's single-machine miss-recovery story into a distributed, cross-machine setting. The case studies are SwiftEP-inspired: each one establishes a theoretical contrast (lower-bound or crossover) rather than a measured system result, then projects that contrast onto the existing CoLoRA trace.

Out of scope for this spec: case studies C (remote placement), E (RPC/RDMA granularity), F (oracle policy gap). These remain on the longer roadmap and may reuse the cost-model library defined here.

Out of scope entirely: any change to `lightllm/`, `S-LoRA/`, the CPU fallback kernels, or the runtime. This is analysis-only tooling under `tools/`.

## 2. The Three Case Studies at a Glance

| ID | Question | Headline figure |
|----|----------|-----------------|
| A | In what fraction of misses does activation/residual movement have a strictly lower theoretical floor than full-object movement? | Lower-bound latency vs. LoRA rank; trace fraction in exec-first regime |
| B | Where does promotion-first become more cost-effective than remote-execution-first once future reuse is considered? | Phase diagram over (rank, reuse count k) with trace-projected operating points |
| D | Does miss-recovery traffic compete with MoE EP dispatch/combine, and which policy survives contention? | Per-miss recovery latency vs. EP load rho, plus foreground RDMA bytes |

## 3. Shared Cost Model (Section 1, approved)

### 3.1 Module

`tools/case_study/cross_machine/cm_cost_model.py` — pure functions over a `Params` dataclass. No I/O, no plotting.

### 3.2 Parameters

```
Network
  B_rdma            RDMA bandwidth between local and remote node (default 400 Gbps = 50 GB/s)
  alpha_rdma        Per-message RDMA setup latency (default 2 us; reserved for case E, not used here)
  B_pcie_h2d        CPU->GPU bandwidth (default 25 GB/s; PCIe Gen4 x16 effective)
  B_pcie_d2h        GPU->CPU bandwidth (default 25 GB/s)

Object sizes (bytes; fp16 throughout)
  W_lora(r, d)      = 2 * r * d * 2          # LoRA A + B matrices
  A_act(b, d)       = b * d * 2              # per-step activation
  R_res(b, d)       = b * d * 2              # post-up-projection residual (default)

Compute (microseconds; calibrated from CoLoRA, see Section 6)
  T_gpu_lora(r, b)
  T_cpu_lora(r, b)
  T_remote_gpu_lora(r, b)   # equal to T_gpu_lora (assumption: same GPU class)
  T_merge(b, d)
  T_staging(b, d)
```

### 3.3 Recovery-path formulas

```
T_promote_min(r, b)
    = W_lora(r, d) / B_rdma
    + T_gpu_lora(r, b)
    + T_merge(b, d)

T_remote_exec_min(r, b)
    = A_act(b, d) / B_rdma
    + T_remote_gpu_lora(r, b)
    + R_res(b, d) / B_rdma
    + T_merge(b, d)
    + T_staging(b, d)
```

### 3.4 Contention overlay (used only by D)

Scalar `rho` in `[0, 1]` representing the fraction of `B_rdma` consumed by MoE EP dispatch/combine traffic.

```
Mode A (no-contention oracle):     B_rdma_eff = B_rdma
Mode B (shared-network realistic): B_rdma_eff = (1 - rho) * B_rdma
Mode C (priority-aware):           recovery flows only in EP idle slots;
                                   effective latency = T_recovery / (1 - rho)
```

### 3.5 Explicit assumptions (must be carried into paper text)

1. Remote GPU compute equals local GPU compute (same hardware class).
2. No remote-side queueing term (deferred to case study C).
3. Residual is hidden-sized post-up-projection `R_res(b, d)`, matching CoLoRA's existing merge step.
4. Compute scales linearly in `r` and `b` from the calibration anchor.
5. `T_remote_exec_min` does not include alpha-term per-RPC overhead (deferred to case E).
6. `rho` is a scalar load fraction, not a time-series of EP bursts.

## 4. Case Study A — Payload-asymmetry upper bound (Section 2, approved)

### 4.1 Script

`tools/case_study/cross_machine/case_a_payload_lower_bound.py`

### 4.2 Input

- `Params` from the cost model, populated via `calibrate_from_colora()`.
- The same joined trace that feeds `notes/storyline_case_study.md` (`artifacts/case_study/router_lora_case_v1/`).

### 4.3 Sweep

- Rank `r in {4, 8, 16, 32, 64}`
- Batch `b in {1, 8}`

### 4.4 Outputs

**Fig A.1** — Lower-bound latency vs. LoRA rank.
Two panels (batch=1, batch=8). Three curves: `T_promote_min`, `T_remote_exec_min`, `min(...)` oracle.

**Fig A.2** — Fraction of trace misses where `T_remote_exec_min < T_promote_min`.
x = rank; y = fraction (bar chart). Computed by replaying the trace, attaching `(b, r)` to each miss, evaluating both formulas.

**CSV** `artifacts/cross_machine/csv/case_a_results.csv` — per-(rank, batch) row with both latencies, gap, and fraction.

**Stub** `artifacts/cross_machine/stubs/case_a.md` — auto-generated paragraph for paper insertion.

## 5. Case Study B — Crossover phase diagram (Section 3, approved)

### 5.1 Script

`tools/case_study/cross_machine/case_b_phase_diagram.py`

### 5.2 Model extension (reuse-aware cost)

```
C_promote(r, b, k)   = T_promote_min(r, b) + k * (T_gpu_lora(r, b) + T_merge(b, d))
C_remote_exec(r, b, k) = (k + 1) * T_remote_exec_min(r, b)
```

Crossover `k*(r, b)` solves `C_promote == C_remote_exec`.

### 5.3 Sweep

- Rank `r in {4, 8, 16, 32, 64}`
- Reuse count `k in {1, ..., 1024}` (log scale on x-axis)
- Batch `b in {1, 8}` (two panels)

### 5.4 k definition

`k` is trace-projected from the reuse-distance distribution under the same cache budget used in `notes/storyline_case_study.md` (object count = 2048, LRU). The adapter reads the existing cached-replay output from `artifacts/case_study/router_lora_case_v1/` and extracts per-miss reuse distance → expected `k` mapping.

### 5.5 Promotion cost note

Promotion is modeled WITHOUT staging (no `T_staging` term in promotion path). Promo writes the full LoRA object into the GPU cache slot directly; no per-call activation staging. This is consistent with CoLoRA's single-machine model.

### 5.6 Outputs

**Fig B.1** — Phase diagram.
- x: expected reuse count `k` (log scale, 1 … 1024)
- y: LoRA rank `r`
- Color: binary exec-first region vs. promote-first region, with `k*` contour overlaid.
- Two panels: batch=1, batch=8.

**Fig B.2** — Trace-projected operating points.
- Scatter of real trace-miss `(k, r)` points on the same axes, colored by which regime they fall in.
- Marginal: histogram of `k` for misses falling in exec-first region.

**CSV** `artifacts/cross_machine/csv/case_b_results.csv`

**Stub** `artifacts/cross_machine/stubs/case_b.md`

## 6. Case Study D — Network contention with MoE EP traffic (Section 4, approved)

### 6.1 Script

`tools/case_study/cross_machine/case_d_network_contention.py`

### 6.2 Modes

```
1. No-contention oracle    B_rdma_eff = B_rdma
2. Shared-network          B_rdma_eff = (1 - rho) * B_rdma
3. Priority-aware          effective latency = T_recovery / (1 - rho)
```

### 6.3 Sweep

- EP load `rho in {0, 0.3, 0.5, 0.7, 0.9}`
- Rank panels: `r in {4, 16, 64}`
- Batch fixed at 8 (CoLoRA representative)

### 6.4 Outputs

**Fig D.1** — Recovery latency vs. EP load.
- x: `rho`
- y: per-miss recovery latency (us)
- Four curves: promote-first/shared, promote-first/priority, exec-first/shared, exec-first/priority.
- Three panels (rank=4, 16, 64).

**Fig D.2** — Foreground RDMA bytes per decode step.
- x: `rho`
- y: bytes (stacked or grouped bars)
- Two bars per `rho`: promote-first vs. exec-first.
- Three panels (rank=4, 16, 64).

**CSV** `artifacts/cross_machine/csv/case_d_results.csv`

**Stub** `artifacts/cross_machine/stubs/case_d.md`

### 6.5 No per-RPC alpha term

D models bytes-on-wire only. The granularity question (many small RPCs vs. coalesced) is deferred to case study E.

## 7. Parameter Calibration (Section 5, approved)

### 7.1 Module

`tools/case_study/cross_machine/params.py`

### 7.2 Anchors

The anchor pair is `T_promote_measured = 1020 us`, `T_exec_measured = 786 us` at rank=16, batch=8. These numbers come from the user's summary; the spec requires the script to source them from a named, verifiable location (e.g., a specific `.csv` in `results/` or a `.log` line). If no such file exists, the script must raise a clear error instructing the user which file to provide.

### 7.3 Calibration procedure

```
1. T_promote_measured(16, 8) = W_lora(16, 4096) / B_pcie_h2d + T_gpu_lora(16, 8) + T_merge(8, 4096)
   => Solve for T_gpu_lora(16, 8) + T_merge(8, 4096)

2. T_exec_measured(16, 8) = A_act(8, 4096) / B_pcie_d2h + T_cpu_lora(16, 8) + R_res(8, 4096) / B_pcie_h2d + T_merge(8, 4096) + T_staging(8, 4096)
   => Solve for T_cpu_lora(16, 8) + T_staging(8, 4096)

3. Extrapolate linearly in r and b from these anchors.
```

### 7.4 Hard-coded constants

| Symbol | Value | Source |
|--------|-------|--------|
| `B_rdma` | 400 Gbps (50 GB/s) | SwiftEP NIC class |
| `B_pcie_h2d` | 25 GB/s | CoLoRA single-machine calibration |
| `B_pcie_d2h` | 25 GB/s | same |
| `d` (hidden) | 4096 | Mixtral/DeepSeek class |
| `bytes_per_elem` | 2 | fp16 |

## 8. File Layout (Section 6, approved)

```
tools/case_study/cross_machine/
├── __init__.py
├── params.py                       # Params dataclass + calibrate_from_colora()
├── cm_cost_model.py                # T_promote_min, T_remote_exec_min, contention overlay
├── trace_adapter.py                # loads existing joined trace; emits (miss, rank, batch, reuse_k)
├── case_a_payload_lower_bound.py   # -> Fig A.1, A.2, CSV
├── case_b_phase_diagram.py         # -> Fig B.1, B.2, CSV
├── case_d_network_contention.py    # -> Fig D.1, D.2, CSV
└── README.md

artifacts/cross_machine/
├── figures/
├── csv/
└── stubs/
```

### 8.1 Trace adapter

`trace_adapter.py` loads the joined trace from `artifacts/case_study/router_lora_case_v1/` (the same trace used by the existing motivation case study). If the trace does not already contain per-miss `(rank, batch)` tuples, the adapter attaches them:
- `rank` = adapter's LoRA rank (from adapter metadata / map file);
- `batch` = decode-step batch size (from request-level replay data in the existing trace).
If batch size is not recorded per-step in the current trace, the adapter defaults to a fixed scalar (configurable, default=8).

### 8.2 artifacts/ git status

`artifacts/` is NOT currently in `.gitignore`. The spec prescribes it should be. If the user prefers a different output location, only this section needs changing.

## 9. Edge cases and error handling

1. **Missing calibration anchors:** If the 1020 us / 786 us pair is not extractable from a named file, `calibrate_from_colora()` must raise `FileNotFoundError` with instructions for which file to provide. No hard-coded fallback.
2. **Trace format drift:** If the joined trace format has changed since `notes/storyline_case_study.md` was written, `trace_adapter.py` must fail early with a schema error, not silently produce wrong figures.
3. **Degenerate divisions:** `1 - rho = 0` when rho = 1 — the contention overlay must handle rho = 1 by returning `+inf` for priority-aware mode and `0` B_rdma_eff for shared-network mode.
4. **Negative crossover k*:** If `k*` solves to a negative value (exec-first is ALWAYS cheaper), the phase diagram should show exec-first across the entire domain.
5. **Zero batch:** b=0 is not swept; all batch parameters must be >= 1. If a trace event has b=0, it is skipped with a warning.

## 10. Self-review (post-write checklist)

- [x] No TBD/TODO placeholders
- [x] Sections are internally consistent: formulas in Section 3 match the sweeps in 4/5/6
- [x] Scope is focused: exactly A, B, D. C/E/F deferred.
- [x] Ambiguity resolved: residual shape (post-up-projection), k source (trace-projected), staging exclusion (promotion), per-RPC alpha (deferred)
- [x] Calibration anchor explicitly flagged as requiring a verified source file
