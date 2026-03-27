# Case Study Workflow

This directory contains the entrypoints for the trace-driven MoE x LoRA case study.

The current implementation covers:

- `B0` repository conventions and schemas
- `B1` narrow literature framing notes
- `B2` fixed prompt corpus generation
- `B3` Azure trace preprocessing
- `B4` tenant-to-adapter mapping
- `B5` real router-trace collection entrypoint
- `B6` trace join and quality checks
- `B7` offline replay locality analysis
- `B8` offline replay cache analysis
- `B9` micro-benchmark calibrated system TPOT analysis
- `B10` synthetic control sweeps on calibrated TPOT
- `B11` small live validation
- `B13` figure and story assembly

## Inputs

Default paths are defined in [configs/global.yaml](/home/shufan/LightLLM-integrate-to-SLoRA/configs/global.yaml):

- ShareGPT V3 JSON
- Azure Functions trace
- base model path
- server launcher path
- dummy LoRA directories

## Run Order

### B2. Prepare Prompt Corpus

Build a deterministic request list from ShareGPT V3.

```bash
python tools/case_study/prepare_prompt_corpus.py
```

Outputs:

- `artifacts/case_study/<run_id>/prompt_corpus/fixed_requests.jsonl`
- `artifacts/case_study/<run_id>/prompt_corpus/prompt_stats.json`
- `artifacts/case_study/<run_id>/prompt_corpus/prompt_length_hist.csv`
- `artifacts/case_study/<run_id>/prompt_corpus/request_manifest.csv`

### B3. Preprocess Azure Trace

Normalize Azure Functions arrivals into an adapter-oriented arrival trace.

```bash
python tools/case_study/preprocess_azure_trace.py
```

Outputs:

- `artifacts/case_study/<run_id>/adapter_trace/adapter_trace_raw.jsonl`
- `artifacts/case_study/<run_id>/adapter_trace/adapter_trace_summary.json`
- `artifacts/case_study/<run_id>/adapter_trace/tenant_popularity.csv`
- `artifacts/case_study/<run_id>/adapter_trace/interarrival_stats.csv`

### B4. Map Tenants to Adapters

Emit deterministic independent and correlated adapter traces.

```bash
python tools/case_study/map_adapter_trace.py \
  --run_id router_lora_case_v1 \
  --raw_trace_path artifacts/case_study/router_lora_case_v1/adapter_trace/adapter_trace_raw.jsonl \
  --output_dir artifacts/case_study/router_lora_case_v1/adapter_trace \
  --cardinalities 8,32,128
```

This is the explicit form of the default `B4` mapping run; it writes the `mapping_policy.md`,
`mapping_summary.json`, and `adapter_trace_mapped_{indep,corr}*.jsonl` files under
`artifacts/case_study/router_lora_case_v1/adapter_trace`.

Outputs:

- `artifacts/case_study/<run_id>/adapter_trace/mapping_policy.md`
- `artifacts/case_study/<run_id>/adapter_trace/mapping_summary.json`
- `artifacts/case_study/<run_id>/adapter_trace/adapter_trace_mapped_indep*.jsonl`
- `artifacts/case_study/<run_id>/adapter_trace/adapter_trace_mapped_corr*.jsonl`

### B5. Collect Real Router Trace

This is the main entrypoint for the real-router half of the case study.

Default behavior:

- starts the local LightLLM server using `test/lora/start_server.sh`
- launches that server with `--no_lora` by default, because `B5` is a pure real-router trace stage
- enables router tracing
- replays `fixed_requests.jsonl` sequentially
- canonicalizes the raw runtime hook output

Command:

```bash
python tools/case_study/collect_router_trace.py
```

If you explicitly want the traced server to load adapters too:

```bash
python tools/case_study/collect_router_trace.py --with_lora
```

If the server is already running and tracing into a known file:

```bash
python tools/case_study/collect_router_trace.py \
  --reuse_server \
  --raw_router_trace_path /tmp/moe_router_trace.jsonl
```

Outputs:

- `artifacts/case_study/<run_id>/router_trace/router_trace_raw.jsonl`
- `artifacts/case_study/<run_id>/router_trace/router_trace.jsonl`
- `artifacts/case_study/<run_id>/router_trace/router_summary.json`
- `artifacts/case_study/<run_id>/router_trace/expert_popularity.csv`
- `artifacts/case_study/<run_id>/router_trace/per_request_expert_stats.csv`
- `artifacts/case_study/<run_id>/router_trace/router_request_log.jsonl`
- `artifacts/case_study/<run_id>/router_trace/collect_router_trace_summary.json`
- `artifacts/case_study/<run_id>/router_trace/router_server.log`

## What `collect_router_trace.py` Does

Input:

- prompt corpus from `B2`
- base model from config
- server launcher from config

Behavior:

1. removes any stale raw trace file for this run
2. launches the server unless `--reuse_server` is set
3. waits for `/healthz`
4. replays the fixed request list through `/v1/chat/completions`
5. writes a request log with latency and status
6. reads raw router events emitted by the model hook
7. canonicalizes them into the schema in `spec/schema.md`
8. emits expert-popularity and per-request summaries

## Expected B5 Result

If B5 succeeds, you should get:

- one canonical `router_trace.jsonl` row per traced token event
- preserved `req_idx` identities matching `fixed_requests.jsonl`
- `event_idx` contiguous within each request
- possible request-interleaved file order from the runtime hook; `B6` re-canonicalizes by `req_idx` before emitting joined traces
- phase labels such as `prefill` and `decode`
- `topk_experts` per event

The most important deliverables are:

- `router_trace.jsonl`: replay-ready real router events
- `router_summary.json`: quick sanity summary
- `expert_popularity.csv`: H1 figure input
- `per_request_expert_stats.csv`: request-level diversity summary

## Common Flags

- `--run_id <id>`: isolate outputs under another run directory
- `--limit <N>`: run a smaller slice first
- `--reuse_server`: do not start a server
- `--server_url http://host:port`: point to an existing server
- `--raw_router_trace_path <path>`: explicit raw hook output path
- `--tee_server_output`: mirror launched server logs to the terminal while still writing `router_server.log`

Example smoke run:

```bash
python tools/case_study/collect_router_trace.py --limit 16
```

If you want to watch the launched server logs live while still saving them:

```bash
python tools/case_study/collect_router_trace.py --limit 16 --tee_server_output
```

## Suggested First Use

For a light first pass:

```bash
python tools/case_study/collect_router_trace.py --limit 16
```

Then inspect:

- `router_summary.json`
- `expert_popularity.csv`
- `router_request_log.jsonl`

If the output looks sane, rerun without `--limit`.

### B6. Join Router Trace with Adapter Mapping

Attach mapped adapter identities to the fixed-request router trace and emit replay-ready joined traces.

Command:

```bash
python tools/case_study/join_traces.py
```

This stage:

- reads `router_summary.json` to get the router request count
- selects the first `request_count` mapped adapter arrivals in sorted `arrival_idx` order, matching the truncation semantics already used by `replay_fixed_requests.py`
- canonicalizes router rows by `req_idx` before emission because the traced router events can contain request-interleaved blocks even though `event_idx` is monotonic within each request
- explodes `topk_experts` so one joined row equals one selected expert event
- writes per-mode joined traces plus a QC report and compact projection summary

Outputs:

- `artifacts/case_study/<run_id>/joined_trace/joined_trace_indep.jsonl`
- `artifacts/case_study/<run_id>/joined_trace/joined_trace_corr.jsonl`
- `artifacts/case_study/<run_id>/joined_trace/join_qc_report.json`
- `artifacts/case_study/<run_id>/joined_trace/join_projection_summary.csv`

The joined JSONL keeps the required replay keys:

- `arrival_idx`
- `req_idx`
- `adapter_id`
- `mapping_mode`
- `event_idx`
- `layer_id`
- `token_pos` or `chunk_idx`
- `phase`
- `expert_id`

If present, it also preserves:

- `start_ts`
- `duration_ms`
- `model_name`
- `trace_run_id`

### B7. Offline Replay Study, Locality Layer

Compute figure-ready locality metrics for:

- `B0` expert-only
- `B1` expert x LoRA independent
- `B2` expert x LoRA correlated

Command:

```bash
python tools/case_study/analyze_locality.py --run_id router_lora_case_v1
```

Outputs:

- `artifacts/case_study/<run_id>/replay/locality/expert_only_locality.json`
- `artifacts/case_study/<run_id>/replay/locality/joint_indep_locality.json`
- `artifacts/case_study/<run_id>/replay/locality/joint_corr_locality.json`
- `artifacts/case_study/<run_id>/replay/locality/popularity_rank.csv`
- `artifacts/case_study/<run_id>/replay/locality/reuse_distance_cdf.csv`
- `artifacts/case_study/<run_id>/replay/locality/topk_coverage.csv`

Object keys:

- `B0`: `(layer_id, expert_id)`, derived from `joined_trace_indep.jsonl` by dropping `adapter_id` while preserving the exact B6 row order
- `B1`: `(layer_id, expert_id, adapter_id)` from `joined_trace_indep.jsonl`
- `B2`: `(layer_id, expert_id, adapter_id)` from `joined_trace_corr.jsonl`

Edge cases and audit rules:

- `B7` reads `joined_trace_indep.jsonl` and `joined_trace_corr.jsonl` in lockstep and refuses to proceed if any non-adapter event field differs. The checked alignment fields are `arrival_idx`, `req_idx`, `event_idx`, `layer_id`, `phase`, `expert_id`, and `token_pos`.
- `reuse_distance_cdf.csv` uses `reuse_distance = -1` for cold first touches. Finite reuse summaries in the JSON files are conditioned on non-cold accesses only.
- Per-request unique-object counts are identical across `B0`, `B1`, and `B2` in this run because each request carries one fixed adapter. The fragmentation signal appears in the global working set and reuse-distance metrics, not inside a single request block.

High-level findings for `router_lora_case_v1`:

- The aligned replay stream contains `44,012,136` expert events across `512` requests.
- Total distinct objects grow from `4,559` in `B0` to `42,806` in `B1` and `59,800` in `B2`.
- Entropy-based effective working-set size grows from `1,138.3` in `B0` to `7,619.7` in `B1` and `10,763.7` in `B2`.
- Top-`1000` coverage drops from `89.1%` in `B0` to `40.7%` in `B1` and `32.6%` in `B2`, showing a much flatter popularity curve once adapters are part of the access key.
- Mean finite reuse distance rises from `463.4` in `B0` to `549.3` in `B1` and `589.5` in `B2`; the median stays at `383`, which is consistent with most locality loss coming from cross-request fragmentation rather than within-request structure changes.

### B8. Offline Replay Study, Cache Layer

Replay the aligned B6 expert-event stream through one identical cache model per condition:

- `B0` expert-only
- `B1` expert x LoRA independent
- `B2` expert x LoRA correlated

Command:

```bash
python tools/case_study/analyze_cache_replay.py --run_id router_lora_case_v1
```

Outputs:

- `artifacts/case_study/<run_id>/replay/cache/cache_metrics.json`
- `artifacts/case_study/<run_id>/replay/cache/cache_curve.csv`
- `artifacts/case_study/<run_id>/replay/cache/per_request_miss_count.csv`
- `artifacts/case_study/<run_id>/replay/cache/eviction_stats.csv`

Cache object keys:

- `B0`: `(layer_id, expert_id)` from `joined_trace_indep.jsonl`, preserving exact B6 event order
- `B1`: `(layer_id, expert_id, adapter_id)` from `joined_trace_indep.jsonl`
- `B2`: `(layer_id, expert_id, adapter_id)` from `joined_trace_corr.jsonl`

Replay rules:

- The default baseline policy is `LRU`, loaded from `case_study.replay_cache.policy` in `configs/global.yaml`.
- `B8` uses one explicit shared budget grid for every condition. For `router_lora_case_v1`, the grid is `[0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]` cache objects.
- `B8` reuses the exact B7 alignment checks, so `B0`, `B1`, and `B2` are replayed under identical event ordering and the tool refuses to proceed if `joined_trace_indep.jsonl` and `joined_trace_corr.jsonl` diverge on any non-adapter field.

High-level findings for `router_lora_case_v1`:

- At tiny budgets the cache is saturated for all three conditions, so the curves are nearly identical. At `512` objects, miss rate is `0.2190` for `B0`, `0.2198` for `B1`, and `0.2201` for `B2`.
- Once the cache reaches the locality scale of `B0`, the joint key expansion dominates. At `2048` objects, miss rate is `0.00159` in `B0`, `0.01095` in `B1`, and `0.01476` in `B2`, so `B1` is `6.9x` worse than `B0` and `B2` is `9.3x` worse.
- At `8192` objects, `B0` has already fallen to its cold-miss floor of `0.000104`, while `B1` remains at `0.00636` and `B2` at `0.00810`. That is `61x` and `78x` higher miss rate than `B0`, and `B2` is still about `27%` worse than `B1`.
- The per-request miss counts show the same amplification. At `2048` objects, median request misses rise from `80` in `B0` to `1,188` in `B1` and `1,577` in `B2`. At `8192` objects, the median request is already at `0` misses in `B0`, but still at `151` misses in `B1` and `254` in `B2`.

### B8.5. Calibrate System Baseline

Measure the transfer, launch, compute, and overlap terms that the TPOT simulator uses.

Command:

```bash
python tools/case_study/calibrate_system_baseline.py \
  --run_id router_lora_case_v1 \
  --base_tpot_ms 1:1.20,2:1.45,4:1.90
```

Outputs:

- `artifacts/case_study/<run_id>/calibration/system_baseline_calibration.json`

Calibration notes:

- The calibration manifest stores packed pinned H2D curves, staged pageable-to-pinned gather plus H2D curves, and fragmented direct pageable H2D curves.
- Both `idle` and `stressed` host profiles are benchmarked so the simulator can model realistic pageable-pool staging overheads instead of assuming an infinite pinned host pool.
- `gate_lora` is excluded from the fixed object-byte accounting because its footprint is negligible relative to the MLP LoRA matrices.
- `base_tpot_ms` is treated as an externally measured hot-cache constant and is serialized into the same JSON alongside the micro-benchmark curves.

### B9. Offline Replay Study, Calibrated System TPOT

Convert B8 miss bitmaps into calibrated token-level TPOT under a causal per-layer system baseline.

Command:

```bash
python tools/case_study/analyze_system_tpot.py \
  --run_id router_lora_case_v1 \
  --calibration_path artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calibration.json
```

Outputs:

- `artifacts/case_study/<run_id>/replay/system_tpot/token_tpot.csv`
- `artifacts/case_study/<run_id>/replay/system_tpot/tpot_quantiles.csv`
- `artifacts/case_study/<run_id>/replay/system_tpot/tail_token_breakdown.csv`
- `artifacts/case_study/<run_id>/replay/system_tpot/layer_barrier_breakdown.csv`
- `artifacts/case_study/<run_id>/replay/system_tpot/request_decode_summary.csv`
- `artifacts/case_study/<run_id>/replay/system_tpot/scheduler_sensitivity.csv`
- `artifacts/case_study/<run_id>/replay/system_tpot/system_tpot_manifest.json`

Replay and audit rules:

- `B9` reads the exact B8 budget grid from `replay/cache/cache_curve.csv` unless `--cache_budgets` overrides it.
- It rebuilds the aligned LRU replay stream from `joined_trace_indep.jsonl` and `joined_trace_corr.jsonl`.
- It validates every `(condition, cache_budget, req_idx)` miss count against `replay/cache/per_request_miss_count.csv` before writing TPOT artifacts.
- The miss-to-latency model is causal: packing is legal only within one layer step, never across future layers of the same token.
- The default physical baseline is `pageable host pool -> pinned staging buffer -> PCIe H2D -> GPU compute`, and the transfer mode and load profile are recorded in the manifest.
- `scheduler_sensitivity.csv` is a second-stage sweep over shared-batch service with one PCIe service queue; its `service_tpot` columns match stage-1 TPOT when `system_batch=1`, and its scheduler wait columns isolate queueing effects separately.

### B10. Synthetic Control Sweeps

Run robustness sweeps over synthetic adapter-assignment knobs while reusing the same object keys, LRU replay kernels, and calibrated token-TPOT path as `B8` and `B9`.

Command:

```bash
python tools/case_study/run_synthetic_control_sweeps.py \
  --run_id router_lora_case_v1 \
  --calibration_path artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calibration.json
```

The default run:

- loads the first `128` requests from `joined_trace_indep.jsonl` as replay templates
- synthesizes `192` requests per sweep point using the seed from `configs/seeds.yaml`
- sweeps:
  - `num_loras in {8, 32, 64, 128}`
  - `skew in {0.0, 0.8, 1.4}`
  - `burstiness in {1.0, 2.5, 6.0}` as mean adapter run length
  - `corr_strength in {0.0, 0.5, 0.95}`
  - `cache_budget in {256, 1024, 4096}`
- writes run-scoped artifacts under `artifacts/case_study/<run_id>/sweeps`

Useful overrides:

- `--joined_indep_path <path>`: source a different joined replay stream for request templates
- `--template_request_count <N>`: change how many real requests seed the template bank
- `--synthetic_request_count <N>`: change how many synthetic requests are replayed per point
- `--num_loras`, `--skew_levels`, `--burstiness_levels`, `--corr_strength_levels`, `--cache_budgets`: override the sweep grid
- `--transfer_mode`, `--load_profile`, `--calibration_stat`, `--system_batch`: change which calibrated TPOT baseline the sweep uses

Outputs:

- `artifacts/case_study/<run_id>/sweeps/sweep_manifest.json`
- `artifacts/case_study/<run_id>/sweeps/sweep_results.csv`
- `artifacts/case_study/<run_id>/sweeps/num_loras_vs_p99.csv`
- `artifacts/case_study/<run_id>/sweeps/skew_burst_corr_grid.csv`

Output contracts:

- `sweep_results.csv` has one row per `(run_id, condition)` sweep point with:
  `run_id, condition, num_loras, skew, burstiness, corr_strength, cache_budget, miss_rate, mean_tpot_ms, p95, p99`
- `num_loras_vs_p99.csv` is the plotting projection:
  `condition, num_loras, cache_budget, p99`
- `skew_burst_corr_grid.csv` is the grid projection:
  `condition, skew, burstiness, corr_strength, cache_budget, miss_rate, p99`
- The `p95` and `p99` columns are token-level TPOT in milliseconds, not the retired request-level latency proxy.

Replay and reproducibility rules:

- `B10` keeps the `B0` / `B1` / `B2` condition definitions unchanged.
- The sweep path reuses the shared `replay_core.py` LRU kernels plus the same calibrated per-token TPOT logic that powers `analyze_system_tpot.py`.
- Every sweep point is deterministic because the point id, adapter schedules, and template resampling order are all derived from serialized seeds and written into `sweep_manifest.json`.

### B11. Small Live Validation

Replay one explicit adapter schedule through the live serving path and collect:

- exact request order and adapter assignment
- per-request latency
- any exposed live COLoRA counters or diagnostics

Command:

```bash
python tools/case_study/live_validate_small.py --run_id router_lora_case_v1
```

Default behavior:

- uses `prompt_corpus/fixed_requests.jsonl`
- auto-selects a small reuse-heavy contiguous request window from the mapped correlated adapter trace
- chooses the largest mapped adapter cardinality that fits the loaded dummy-LoRA count
  - for the current `10` dummy-LoRA setup this resolves to `adapter_trace_mapped_corr_c008.jsonl`
- caps live decode to `16` tokens per request so the run stays narrow and robust
- launches `test/lora/start_server.sh` with the configured dummy LoRAs unless `--reuse_server` is set
- mirrors launched server logs to stdout by default while still writing `live_server.log`
- reads `[COLoRA]` debug lines from the server log in per-request byte windows to assign live counters back to each request

Useful overrides:

- `--request_count <N>`: change the live subset size
- `--max_decode_tokens <N>`: adjust the decode cap
- `--selection_strategy first_n`: replay the first aligned requests instead of the auto reuse window
- `--selection_strategy explicit_indices --request_indices 409,410,411,412`: force one exact aligned subset
- `--mapping_mode indep`: replay the independent mapped schedule instead of the correlated one
- `--reuse_server --server_log_path <path>`: use an already-running server while still collecting counter windows from its log
- `--no_tee_server_output`: keep the launched server quiet on stdout and only write `live_server.log`

Outputs:

- `artifacts/case_study/<run_id>/live_validation/live_requests.jsonl`
- `artifacts/case_study/<run_id>/live_validation/live_latency.csv`
- `artifacts/case_study/<run_id>/live_validation/live_counters.json`
- `artifacts/case_study/<run_id>/live_validation/live_summary.md`
- `artifacts/case_study/<run_id>/live_validation/live_server.log`

Output contracts:

- `live_requests.jsonl` records the exact live replay order, aligned `req_idx`, offline adapter label, resolved live adapter name, and capped decode length actually used.
- `live_latency.csv` records one row per live request with:
  `req_idx, adapter_id, submit_order, start_time, end_time, latency_ms, status`
- `live_counters.json` stores:
  - aggregate COLoRA counters parsed from the server log
  - per-request counter windows when `[COLoRA]` lines are present
  - optional `/metrics` snapshots and deltas
  - `/v1/lora/adapters` validation output
- `live_summary.md` is the short writeup for the paper appendix:
  - setup
  - limitations
  - qualitative agreement or disagreement with the offline replay mechanism

Scope note:

- `B11` is intentionally small and diagnostic. It is not a throughput benchmark and it is not the primary evidence for the paper.
- The default replay uses dummy LoRAs, one server configuration, and a short decode cap to check whether the offline cold-miss then reuse mechanism is visible at all in live serving behavior.

### B13. Figure and Story Assembly

Assemble the paper-facing figures, a traceable manifest, and the short main-text storyline directly from the B7-B10 artifacts.

Command:

```bash
python tools/case_study/assemble_paper_figures.py \
  --run_id router_lora_case_v1
```

Notebook companion:

- `tools/case_study/assemble_paper_figures.ipynb`
- The notebook calls the same `plot_fig*`, `write_manifest`, `write_storyline`, and `assemble_paper_figures(...)` helpers as the CLI script, so interactive Jupyter edits stay on the same reproducible code path.

This stage:

- reads the existing B7 locality CSVs for popularity and reuse-distance summaries
- reads the existing B8 cache replay outputs for the miss-rate curve and request-level miss counts
- reads the existing B9 tail breakdown to attach the request-level tail mechanism
- reads a B10 `num_loras_vs_p99.csv` sweep projection that includes the low-LoRA onset regime needed for the plateau-style `Fig. 4`
- writes all figure PDFs, `figure_manifest.md`, and `notes/storyline_case_study.md` without any manual editing step

`Fig. 4` note:

- The updated `Fig. 4` is designed to show that the tail penalty rises quickly from the single-LoRA regime and then plateaus.
- For that reason, make sure the run-scoped B10 sweep under `artifacts/case_study/<run_id>/sweeps` includes at least `num_loras=1,2,3,4,5,6,7,8` and a cache budget of `8192` objects or larger.
- The figure code prefers the `8192`-object slice when available because it shows the onset-to-plateau transition more clearly than the older `4096` default.
- The notebook exposes one cell per figure, so you can inspect or tweak the rendered plot inline and still save the final PDF plus manifest without any manual post-processing step.

Example B10 command for the plateau-oriented `Fig. 4` input:

```bash
python tools/case_study/run_synthetic_control_sweeps.py \
  --run_id router_lora_case_v1 \
  --num_loras 1,2,3,4,5,6,7,8,16,32,64,128 \
  --skew_levels 0.8 \
  --burstiness_levels 2.5 \
  --corr_strength_levels 0.5 \
  --cache_budgets 256,1024,4096,8192,16384,32768,65536
```

Outputs:

- `artifacts/case_study/<run_id>/figures/fig1_popularity_rank.pdf`
- `artifacts/case_study/<run_id>/figures/fig2_reuse_distance.pdf`
- `artifacts/case_study/<run_id>/figures/fig3_cache_curve.pdf`
- `artifacts/case_study/<run_id>/figures/fig4_num_loras_vs_p99.pdf`
- `artifacts/case_study/<run_id>/figures/fig5_tail_breakdown.pdf`
- `artifacts/case_study/<run_id>/figures/figure_manifest.md`
- `notes/storyline_case_study.md`

Reproducibility note:

- `B13` is a pure assembly stage. It does not require any undocumented manual edits, and every figure in the manifest lists the exact source artifact paths plus the plotting script that produced it.
- If you want the onset-to-plateau `Fig. 4`, regenerate the richer `B10` sweep above into the run-scoped default sweep directory before running `B13`.
