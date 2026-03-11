# Case Study Workflow

This directory contains the entrypoints for the trace-driven MoE x LoRA case study.

The current implementation covers:

- `B0` repository conventions and schemas
- `B1` narrow literature framing notes
- `B2` fixed prompt corpus generation
- `B3` Azure trace preprocessing
- `B4` tenant-to-adapter mapping
- `B5` real router-trace collection entrypoint

## Why `artifacts/case_study` Is Ignored

Generated run outputs live under `artifacts/case_study/<run_id>/...`.

That tree is intentionally ignored by [artifacts/.gitignore](/home/shufan/LightLLM-integrate-to-SLoRA/artifacts/.gitignore) because:

- the Azure-derived outputs are large
- router traces can become very large very quickly
- these files are reproducible from local inputs and config
- keeping them tracked would make routine iteration noisy and expensive

Tracked files remain the source of truth:

- schemas in [spec/schema.md](/home/shufan/LightLLM-integrate-to-SLoRA/spec/schema.md)
- configs in [configs/global.yaml](/home/shufan/LightLLM-integrate-to-SLoRA/configs/global.yaml) and [configs/seeds.yaml](/home/shufan/LightLLM-integrate-to-SLoRA/configs/seeds.yaml)
- scripts in this directory
- notes in `notes/`

If you want specific outputs tracked, whitelist only those files explicitly rather than the whole generated tree.

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
python tools/case_study/map_adapter_trace.py
```

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
- preserved `req_idx` ordering matching `fixed_requests.jsonl`
- `event_idx` contiguous within each request
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
