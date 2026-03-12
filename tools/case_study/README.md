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
