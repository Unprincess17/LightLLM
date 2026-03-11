# Generated Artifacts

Tracked files in `artifacts/` document the output contract. Generated run outputs remain untracked by default.

## Layout

Generated outputs go under:

`artifacts/case_study/<run_id>/<stage>/...`

Default run id:

`router_lora_case_v1`

## First-Stage Deliverables

- `prompt_corpus/fixed_requests.jsonl`
- `prompt_corpus/prompt_stats.json`
- `prompt_corpus/prompt_length_hist.csv`
- `prompt_corpus/request_manifest.csv`
- `adapter_trace/adapter_trace_raw.jsonl`
- `adapter_trace/adapter_trace_summary.json`
- `adapter_trace/tenant_popularity.csv`
- `adapter_trace/interarrival_stats.csv`
- `adapter_trace/mapping_policy.md`
- `adapter_trace/adapter_trace_mapped_indep*.jsonl`
- `adapter_trace/adapter_trace_mapped_corr*.jsonl`
- `adapter_trace/mapping_summary.json`
- `router_trace/collect_router_trace_summary.json`
- `router_trace/router_request_log.jsonl`

## Reproducibility

- Schemas: `spec/schema.md`
- Global config: `configs/global.yaml`
- Seeds: `configs/seeds.yaml`

All scripts should be invoked with an explicit or default `run_id` so output paths stay deterministic.
