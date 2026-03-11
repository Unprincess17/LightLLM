# Case Study Schema Contract

This document locks the canonical schemas, directory layout, and naming used by the MoE x LoRA case-study pipeline.

## Run Layout

All generated outputs live under:

`artifacts/case_study/<run_id>/<stage>/...`

Default `run_id`: `router_lora_case_v1`

Stage directories used in the first implementation:

- `prompt_corpus`
- `adapter_trace`
- `router_trace`

## Naming Conventions

- Request index: `req_idx`
- Arrival order index: `arrival_idx`
- Router event index within one request: `event_idx`
- Adapter namespace: `lora_<slot>`
- Mapping modes: `indep`, `corr`
- Router phases: `prefill`, `decode`

## Seed Policy

- All scripts must read `configs/seeds.yaml`.
- Every artifact summary must record the effective seed values.
- If a stage is deterministic without sampling, it still records the seed that would control tie-breaking or hashing.

## Canonical JSONL Schemas

### `fixed_requests.jsonl`

One line per replay request.

```json
{
  "req_idx": 0,
  "prompt_text": "...",
  "prompt_len_tokens": 128,
  "target_decode_len": 64,
  "source_id": "sharegpt_xxx",
  "split_tag": null,
  "messages": [
    {"role": "user", "content": "..."}
  ]
}
```

Required keys:

- `req_idx`
- `prompt_text`
- `prompt_len_tokens`
- `target_decode_len`
- `source_id`

Optional keys:

- `split_tag`
- `messages`

### `adapter_trace_raw.jsonl`

One line per parsed Azure invocation record after timestamp normalization.

```json
{
  "arrival_idx": 0,
  "start_ts": 1234567890,
  "end_ts": 1234567990,
  "duration_ms": 100,
  "app_id": "app_42",
  "func_id": "func_7",
  "session_id": "app_42:1",
  "raw_trace_source": "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
}
```

Required keys:

- `arrival_idx`
- `start_ts`
- `end_ts`
- `duration_ms`
- `app_id`
- `raw_trace_source`

Optional keys:

- `func_id`
- `session_id`

### `adapter_trace_mapped_*.jsonl`

One line per mapped adapter invocation record.

```json
{
  "arrival_idx": 0,
  "start_ts": 1234567890,
  "end_ts": 1234567990,
  "duration_ms": 100,
  "app_id": "app_42",
  "func_id": "func_7",
  "session_id": "app_42:1",
  "adapter_id": "lora_3",
  "mapping_mode": "indep",
  "cardinality": 32
}
```

Required keys:

- `arrival_idx`
- `adapter_id`
- `mapping_mode`
- `cardinality`
- `start_ts`
- `end_ts`
- `duration_ms`
- `app_id`

Optional keys:

- `func_id`
- `session_id`

### `router_trace.jsonl`

One line per router event emitted by the model trace hook.

```json
{
  "req_idx": 0,
  "event_idx": 17,
  "layer_id": 5,
  "token_pos": 12,
  "phase": "decode",
  "topk_experts": [3, 18],
  "topk_scores": [0.77, 0.21],
  "num_selected_experts": 2,
  "model_name": "qwen3_vl_30b_a3b",
  "trace_run_id": "router_lora_case_v1"
}
```

The runtime logger currently emits `topk_weights`; downstream tooling should accept `topk_scores` or `topk_weights`.

Required keys:

- `req_idx`
- `layer_id`
- `token_pos`
- `phase`
- `topk_experts`

Optional keys:

- `event_idx`
- `topk_scores`
- `topk_weights`
- `num_selected_experts`
- `model_name`
- `trace_run_id`

### `joined_trace_*.jsonl`

Exploded form, one line per selected expert after attaching one adapter to each request.

```json
{
  "arrival_idx": 0,
  "req_idx": 0,
  "adapter_id": "lora_3",
  "mapping_mode": "indep",
  "event_idx": 17,
  "layer_id": 5,
  "token_pos": 12,
  "phase": "decode",
  "expert_id": 3
}
```

Required keys:

- `arrival_idx`
- `req_idx`
- `adapter_id`
- `mapping_mode`
- `layer_id`
- `token_pos`
- `phase`
- `expert_id`

## Determinism Rules

- `req_idx` is contiguous from `0`.
- `arrival_idx` is monotonic from `0`.
- Sorting ties use lexical order of original identifiers.
- Adapter remapping must be deterministic under the configured cardinality and seed.
