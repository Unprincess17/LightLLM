# COLoRA Runtime Audit — 2026-03-31

## Status legend
- **Implemented** — code path exists, tests cover the semantics
- **Partial** — code path exists but incomplete or untested
- **Missing** — paper claims it, code does not implement it
- **Unverified** — code exists but no test or measurement confirms the semantics

## A. Execution-first hot/cold architecture

| # | Claim | Paper ref | Code path | Tests | Status | Notes | Next action |
|---|-------|-----------|-----------|-------|--------|-------|-------------|
| A1 | Cold miss defaults to execution-first recovery, not blocking promotion | SS3.1 | `lora_dispatch.py:39` `MISS_POLICY_CPU_FIRST` is default; `lora_dispatch.py:1995-2000` branches on policy | `test_colora_no_blocking_miss.py` | Implemented | `cpu_first` executes on CPU immediately, queues promotion async | -- |
| A2 | GPU hot path and CPU cold path coexist under hybrid MoE mode | SS3.1 | `lora_dispatch.py:1929-2231` `_batch_apply_moe_lora_hybrid` splits hits (GPU BGMV) from misses (CPU AVX) | `test_colora_no_blocking_miss.py` | Implemented | Hit adapters use BGMV kernel on GPU cache; miss adapters fall back to CPU | -- |
| A3 | Expert-LoRA joint object is the runtime management key | SS2 | `lora_dispatch.py:136-139` `JointObjectKey(layer_id, adapter_bin, expert_id)` | `test_colora_background_optimizations.py` | Implemented | All promotion/prefetch/spec use joint keys | -- |
| A4 | Miss handling and later promotion are decoupled | SS3.1, SS3.5 | `lora_dispatch.py:924-963` `note_decode_joint_access` queues promotion async; miss proceeds independently via CPU | `test_colora_background_optimizations.py::test_deferred_promotion_rejection_does_not_block_cpu_fallback` | Implemented | Deferred promotion is background; CPU miss completion is foreground | -- |

## B. Request-level skip-and-reinsert scheduler

| # | Claim | Paper ref | Code path | Tests | Status | Notes | Next action |
|---|-------|-----------|-----------|-------|--------|-------|-------------|
| B1 | A faulting request can be paused without blocking unrelated ready requests | SS3.1 | `infer_batch.py:412-416` `colora_continuation`, `colora_paused`; `lora_dispatch.py:162-172` `ColoraCompletionTask`; `lora_dispatch.py:723` completion queue | `test_colora_request_skip_runtime.py` | Implemented | Paused request activation staged for CPU; GPU continues with ready requests | -- |
| B2 | Continuation resumes from the next execution point / resume layer | SS3.1 | `infer_batch.py:412` `colora_continuation` holds layer state for resumption | `test_colora_request_skip_runtime.py` | Implemented | Resume layer stored in continuation object | -- |
| B3 | Continuation concurrency is explicitly bounded | SS3.1 | `start_args_type.py` `colora_max_continuations=8`; `base_backend.py:1289` passes to dispatcher | `test_colora_request_skip_runtime.py`, `test_colora_cli_args.py` | Implemented | Default 8, configurable via CLI | -- |
| B4 | `overlap_mode=no_overlap` disables request skip semantics | Design | `base_backend.py:1259-1265` forces `request_skip_enabled=False` when `overlap_mode="no_overlap"` | `test_colora_request_skip_runtime.py::test_init_batched_lora_adapters_disables_request_skip_and_async_fallback_only_in_no_overlap`, `test_colora_background_optimizations.py::test_no_overlap_disables_async_overlap_gate` | Implemented | Also disables `async_fallback_enabled` | -- |

## C. CPU cold-path realization

| # | Claim | Paper ref | Code path | Tests | Status | Notes | Next action |
|---|-------|-----------|-----------|-------|--------|-------|-------------|
| C1 | Packed / coalesced activation roundtrip exists | SS3.4 | `transformer_layer_infer.py:409-508` `_dispatch_lora_with_coalesced_cpu_roundtrip`: packs -> D2H -> CPU compute -> H2D -> scatter | -- | Unverified | Code exists but no unit test exercises the full roundtrip in isolation | test addition |
| C2 | CPU-side MoE kernels are used for strict cpu/hybrid paths | SS3.4 | `lora_dispatch.py:1154-1271` `_strict_moe_cpu_batch_lora` uses `moe_batch_lora_gate_avx` (stage 1) + projection-specific AVX kernel (stage 2) | `test_colora_no_blocking_miss.py`, `test_colora_background_optimizations.py` | Implemented | AVX-optimized BF16 kernels; thread binding via dispatcher config | -- |
| C3 | Asynchronous transfers on separate CUDA streams allow overlap | SS3.4 | `transformer_layer_infer.py:460-502` uses pinned memory and stream sync for D2H/H2D; `lora_dispatch.py:2048-2072` submits to thread pool for async CPU compute | -- | Unverified | Stream-level overlap exists in code but no isolated measurement confirms it | measurement run |

## D. Background optimizations

| # | Claim | Paper ref | Code path | Tests | Status | Notes | Next action |
|---|-------|-----------|-----------|-------|--------|-------|-------------|
| D1 | Deferred promotion admission depends on reuse-distance-like signals (EMA interval) | SS3.5 | `lora_dispatch.py:924-963` `note_decode_joint_access`: EMA interval <= `delta_steps` -> admit; otherwise reject | `test_colora_background_optimizations.py::test_deferred_promotion_rejection_does_not_block_cpu_fallback` | Implemented | Static delta (default 4); adaptive delta is future work per paper | -- |
| D2 | Temporal prefetch is step-scoped and not correctness-critical | SS3.5, App. A | `lora_dispatch.py:964-1127`: `begin_temporal_prefetch_step`, `maybe_submit_temporal_prefetch_job`, FIFO hot cache with generation counters; fallback on miss/not-ready | `test_colora_background_optimizations.py::test_temporal_prefetch_missing_or_not_ready_falls_back_cleanly` | Implemented | Prefetch never blocks demand path; stats track false positives | -- |
| D3 | Speculative dispatch is implemented or clearly still MVP/unverified | SS3.5 | `lora_dispatch.py:1393-1799`: full spec infrastructure with `begin_spec_step`, `maybe_submit_fused_gate_up_spec_job`, `try_bind_gate_up_job_with_status`; CLI default `colora_speculative_dispatch=False` | `test_colora_cli_args.py` (default=False only) | Partial | Infrastructure complete but disabled by default; no runtime test exercises the bind/retire cycle | test addition |

## E. Observability and paper evidence

| # | Claim | Paper ref | Code path | Tests | Status | Notes | Next action |
|---|-------|-----------|-----------|-------|--------|-------|-------------|
| E1 | Runtime emits enough counters for paper claims (hit/miss, queueing, overlap, promotion, prefetch) | SS3 | `lora_dispatch.py:656-706` defines 39+ counters; `transformer_layer_infer.py:270-387` aggregates via `_merge_colora_stats` | `test_colora_no_blocking_miss.py` | Implemented | Full surface: hit/miss tokens, promotion breakdown, prefetch stats, overlap ratio, kernel calls, timing, bytes | -- |
| E2 | Case-study scripts can consume counters or calibrated artifacts | Tools | `analyze_system_tpot.py` reads calibration JSON + cache replay; `live_validate_small.py` parses `[COLoRA]` debug lines from server log | `test_case_study_system_tpot_tools.py`, `test_case_study_live_validation_tools.py` | Implemented | CSV merge preserves conditions; live parser aggregates all counter fields | -- |
| E3 | System-TPOT tooling covers the paper's comparison modes | SS3, eval | `system_tpot_core.py:59-67` defines `MISS_HANDLING_MODE_ORDER`: load_then_run, execution_first, no_cpu_path, no_deferred_sync | -- | Unverified | Constants exist but no test asserts the set matches the paper's intended comparison | test addition |

## Summary

| Category | Implemented | Partial | Missing | Unverified |
|----------|------------|---------|---------|------------|
| A. Execution-first | 4 | 0 | 0 | 0 |
| B. Request-skip | 4 | 0 | 0 | 0 |
| C. CPU cold-path | 1 | 0 | 0 | 2 |
| D. Background opts | 2 | 1 | 0 | 0 |
| E. Observability | 2 | 0 | 0 | 1 |
| **Total** | **13** | **1** | **0** | **3** |

## Test gate — 2026-03-31

All 25 COLoRA-targeted unit tests passed:
- `test_colora_no_blocking_miss.py` (3 tests)
- `test_colora_background_optimizations.py` (6 tests)
- `test_colora_request_skip_runtime.py` (5 tests)
- `test_case_study_system_tpot_tools.py` (3 tests)
- `test_case_study_live_validation_tools.py` (6 tests)
- `test_colora_cli_args.py` (2 tests)

## Measured evidence — 2026-03-31

### System-TPOT replay
- **Modes available:** load_then_run (baseline only — execution_first requires `cold_path_profile` in calibration, which is absent)
- **Conditions:** expert_only, joint_indep, joint_corr
- **Budgets:** 0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
- Verified by `tpot_quantiles.csv`: P99.9 TPOT reaches 7.06 ms under joint conditions at budget=2048 vs 1.35 ms for expert-only (confirms paper's tail-blowout claim)
- Verified by `background_policy_summary.csv`: deferred promotion and prefetch counters present and zero under load_then_run (expected — background opts only active in execution_first)
- **Blocker:** execution_first/no_deferred_sync replays blocked on missing `cold_path_profile` in calibration JSON — requires recalibration with COLoRA cold-path measurement enabled
- Still unverified: speculative dispatch contribution remains unisolated (D3)
- Still unverified: coalesced activation roundtrip in isolation (C1)
- Still unverified: stream-level overlap measurement (C3)

### Motivation figures
- Panel A (root cause): practical-limit coverage at rank=1024: C0=89.5%, C1=9.1%, C2=13.5%
- Panel B (capacity illusion): floor budgets: C0=2048, C2=65536
- Panel C (tail blowout): budget=2048 P99 TPOT: C0=1.20 ms, C1=2.31 ms, C2=2.16 ms; P99.9: C0=1.35 ms, C1=7.06 ms, C2=7.06 ms
- All panels regenerated without missing-input errors
