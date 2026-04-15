# Live E2E Evaluation for COLoRA

This module provides a manifest-driven runner for live end-to-end evaluation of COLoRA configurations against a baseline LoRA all-GPU configuration. It follows the design in `docs/superpowers/specs/2026-04-01-live-e2e-evaluation-design.md`.

## Quick Start

### Run the canonical paper suite

```bash
python -m tools.evaluation.live_e2e --manifest configs/live_e2e/paper_suite_example.yaml
```

This will:

1. Execute each of the three canonical configurations through `test/lora/benchmark_lora.sh`
2. Create output directories under `artifacts/evaluation/live_e2e/canonical_paper_suite_01/`
3. Automatically run summarization after all runs complete
4. Emit `live_e2e_comparison.csv` with p50/p95/p99 latency and throughput for paper use

### Manual smoke commands

Paper suite single-run smoke command:

```bash
python -m tools.evaluation.live_e2e --manifest configs/live_e2e/paper_suite_example.yaml
```

Diagnostic nsys smoke command:

```bash
python -m tools.evaluation.live_e2e --manifest path/to/diagnostic_manifest.yaml
```

Use a diagnostic manifest containing at least one run with:

```yaml
suite_kind: diagnostic
nsys_enabled: true
nsys_output_prefix: smoke_profile
```

### Run summarization only on existing results

```bash
python -m tools.evaluation.live_e2e --manifest configs/live_e2e/paper_suite_example.yaml --summarize-only
```

### Resume after partial failure (skip valid runs)

By default, re-running the same manifest **does not** re-execute a run whose output directory already contains `run_result.json` with `"valid": true`. Invalid or missing results are always re-run. Use **`--overwrite`** to execute every run regardless (for example after changing the manifest or when you want fresh benchmarks).

```bash
python -m tools.evaluation.live_e2e --manifest configs/live_e2e/your_suite.yaml
python -m tools.evaluation.live_e2e --manifest configs/live_e2e/your_suite.yaml --overwrite
```

## Real Trace Replay (Alibaba)

Live e2e now supports adapter-trace replay from Alibaba-derived JSONL files.
`tools/case_study/preprocess_gentd26_trace.py` outputs mapped traces that can be
used directly via manifest fields.

Use:

- `measurement_adapter_trace_path` for measured phase replay
- `warmup_adapter_trace_path` for optional warmup replay
- legacy `adapter_trace_path` is still accepted as a measurement-phase alias

Expected JSONL row schema:

- required: `adapter_id`
- optional ordering keys: `arrival_idx`, `req_idx`

Example manifest:

```yaml
run_id: live_e2e_alibaba_trace_smoke
runs:
  - run_label: exec_first_real_trace
    suite_kind: diagnostic
    mode_label: execution_first
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: cpu_first
    overlap_policy: request_skip
    warmup_adapter_trace_path: /path/to/adapter_trace_mapped_corr_warmup.jsonl
    measurement_adapter_trace_path: /path/to/adapter_trace_mapped_corr.jsonl
    warmup_requests: 0
    measurement_requests: 128
```

### Fake-server smoke (avoid busy default port)

For non-GPU smoke validation with a fake endpoint on a non-default port, run:

```bash
pytest test/lora/test_live_e2e_real_trace_fake_server.py -v
```

### Run a diagnostic nsys profiling run

Create a manifest with one or more runs where `nsys_enabled: true`, e.g.:

```yaml
run_id: diagnostic_run_01
runs:
  - run_label: diag_execution_first
    suite_kind: diagnostic
    mode_label: execution_first
    compute_device: "..."
    miss_handling_mode: execution_first
    nsys_enabled: true
    nsys_output_prefix: exec_first_profile
    ...
```

Then run:

```bash
python -m tools.evaluation.live_e2e --manifest path/to/your/diagnostic_manifest.yaml
```

The nsys `.nsys-rep` artifact will be created in the diagnostic run directory.

## Output Layout

```
artifacts/evaluation/live_e2e/<run_id>/
├── paper_runs/<run_label>/
│   ├── config_snapshot.json
│   ├── run_result.json
│   ├── benchmark_stdout.log
│   ├── benchmark_stderr.log
│   └── per_request_metrics.jsonl
├── diagnostic_runs/<run_label>/
│   ├── config_snapshot.json
│   ├── run_result.json
│   ├── benchmark_stdout.log
│   ├── benchmark_stderr.log
│   ├── per_request_metrics.jsonl
│   └── <prefix>.nsys-rep
└── summaries/
    ├── manifest_results.json
    ├── live_e2e_summary.json
    └── live_e2e_comparison.csv
```

## Naming schema (stable)

Use this scheme for `run_label` values, artifact directory names, and cross-manifest references so runs stay comparable and grep-friendly.

**Core fields** (concatenate in order; each segment uses `__` separators):

```
colora__miss-{load_then_run|cpu_first|no_deferred_sync|no_cpu_path}
__reqskip-{0|1}
__ovmode-{full|no_overlap}
__asynccpu-{0|1}
__workers-{N}
__kernel-{avx|naive}
__packer-{0|1}
__tprefetch-{0|1}
__moe-{gpu|hybrid}
```

**Prefetch detail** (optional; add when temporal prefetch is enabled and you want the name to expose the full prefetch configuration):

```
__deferdelta-{N}
__ema-{F}
__tpwl-{none|0-7|...}
__thot-{N}
```

For many paper-suite artifacts, `overlap_mode` is effectively `full` and temporal prefetch is effectively off (`tprefetch-0`). The clean scheduler pair is then:

- `colora__miss-load_then_run__reqskip-0__ovmode-full__asynccpu-0__workers-0__kernel-avx__packer-1__tprefetch-0__moe-gpu`
- `colora__miss-load_then_run__reqskip-1__ovmode-full__asynccpu-0__workers-0__kernel-avx__packer-1__tprefetch-0__moe-gpu`

A representative `no_deferred_sync` example:

- `colora__miss-no_deferred_sync__reqskip-0__ovmode-full__asynccpu-0__workers-1__kernel-naive__packer-0__tprefetch-0__moe-hybrid`

## Running Tests

```bash
# Manifest parsing and command construction tests
pytest test/lora/test_live_e2e_manifest_parsing.py -v

# Summary parsing tests
pytest test/lora/test_live_e2e_summary_parsing.py -v
```

## Notes

- All outputs under `artifacts/evaluation/` are untreated artifacts and are not tracked in git.
- The `live_e2e_comparison.csv` contains only valid paper-suite runs for direct use in the paper table.
- Diagnostic nsys runs are not included in the comparison CSV by default.

