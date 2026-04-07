# Live E2E Evaluation for COLoRA

This module provides a manifest-driven runner for live end-to-end evaluation of COLoRA configurations against a baseline LoRA all-GPU configuration. It follows the design in `docs/superpowers/specs/2026-04-01-live-e2e-evaluation-design.md`.

## Quick Start

### Run the canonical paper suite

```bash
python -m tools.evaluation.live_e2e --manifest configs/live_e2e/paper_suite_example.yaml
```

This will:
1.  Execute each of the three canonical configurations through `test/lora/benchmark_lora.sh`
2.  Create output directories under `artifacts/evaluation/live_e2e/canonical_paper_suite_01/`
3.  Automatically run summarization after all runs complete
4.  Emit `live_e2e_comparison.csv` with p50/p95/p99 latency and throughput for paper use

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
