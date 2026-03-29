# Motivation Microbenchmarks

Scripts for generating motivation figures used in the paper.

## Temporal Locality of MoE Expert Routing

**Script**: `plot_temporal_locality.py`

Measures how much expert-set overlap exists between consecutive decode tokens at each MoE layer. For each sequence, for each layer L, the script iterates through decoded tokens (indexed by position `t = 1, 2, ..., T`) and computes:

```
Hit_Rate(L, t) = |E(L,t) ∩ E(L,t-1)| / |E(L,t)|
```

where `E(L,t)` is the set of top-k experts selected at layer `L` for decode token at position `t`. The figure plots the per-layer average of this metric (averaged over all sequences and all valid token transitions) as a single mean trend line.

### Usage

```bash
# Quick test with synthetic data:
python tools/motivation/plot_temporal_locality.py --use_mock

# Real data from B5 router trace:
python tools/motivation/plot_temporal_locality.py \
    --trace_path artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl \
    --output temporal_locality_hit_rate.pdf
```

### Input

Canonical `router_trace.jsonl` produced by `tools/case_study/collect_router_trace.py` (B5 stage). Each JSONL line must contain:

| Field | Type | Description |
|-------|------|-------------|
| `req_idx` | int | Sequence / request identifier |
| `layer_id` | int | MoE layer index |
| `token_pos` | int | Token position within the sequence |
| `phase` | str | `"prefill"` or `"decode"` (only decode is used) |
| `topk_experts` | list[int] | Selected expert IDs, e.g. `[3,7,9,12,15,18,21,24]` |

### Output

- `temporal_locality_hit_rate.pdf` -- publication-ready figure
- `temporal_locality_hit_rate.csv` -- per-layer statistics (mean, std, p5, p95, num_transitions)

### Results (Qwen3-VL-30B-A3B, top-8, 128 experts, 48 layers, ShareGPT v3)

Generated from the real B5 router trace at `artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl`.

| Metric | Value |
|--------|-------|
| Decode events analyzed | 5,297,088 |
| Token transitions | 5,272,512 |
| Requests | 512 |
| Global mean hit rate | **72.2%** |
| Highest per-layer mean (L=0) | 95.3% |
| Lowest per-layer mean (L=22) | 56.8% |

The per-layer hit rate follows a U-shaped curve: early layers (L=0 to L=3) exhibit very high locality (85-95%), middle layers (L=20 to L=35) dip to 58-70%, and the final layers (L=44 to L=47) recover to 80-87%. The mean trend remains well above random chance (6.25% for top-8 out of 128 experts).
