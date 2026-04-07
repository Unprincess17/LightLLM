# Live E2E Evaluation Design

Date: 2026-04-01

## Goal

Design a live end-to-end evaluation flow for COLoRA that launches a real server, sends real requests, and produces paper-facing latency evidence first, with optional nsys-based diagnosis second.

The design should reuse the existing server and benchmark entrypoints instead of introducing a replacement harness.

## Context

Relevant existing entrypoints:
- `test/lora/start_server.sh` — server launcher with COLoRA and baseline knobs
- `test/lora/benchmark_lora.sh` — current orchestrator for launching the server, sending traffic, collecting logs, and emitting request-level metrics
- `tools/case_study/live_validate_small.py` — prior live validation path in the case-study workflow
- `tools/case_study/collect_router_trace.py` — existing server lifecycle and request replay helper patterns

The synthetic evaluation plan is already in progress separately. This design covers the live E2E side only.

## Decisions

### Evaluation priority
- Primary goal: paper evidence
- Secondary goal: breakdown analysis for live E2E behavior

### Comparison matrix
Use a two-stage matrix.

#### Stage A: paper suite
A fixed, small, reproducible request slice with three canonical configurations:
1. LoRA GPU/GPU baseline
2. COLoRA `execution_first`
3. COLoRA `load_then_run`

Primary outputs:
- request success rate
- throughput
- request latency p50/p95/p99

#### Stage B: diagnosis suite
Selective reruns of chosen configurations with nsys enabled.

Primary outputs:
- raw nsys profiling artifacts
- metadata linking the profile back to the exact benchmark configuration and request slice

Diagnosis outputs are not mixed into the headline paper table by default.

### Operational model
Reuse the existing benchmark flow.

`test/lora/benchmark_lora.sh` remains the canonical orchestrator for:
- launching the server
- sending requests
- collecting server and benchmark logs
- writing per-request metrics
- cleanup

The new design adds:
1. a thin nsys wrapper around `benchmark_lora.sh`
2. a minimal post-processing summary layer

## Proposed architecture

### 1. Experiment manifest/config layer
Add a small declarative configuration format for live E2E runs.

Each run definition should include:
- run label
- suite kind (`paper` or `diagnostic`)
- server mode / baseline label
- compute device string
- relevant COLoRA knobs
- request slice / adapter trace inputs
- output root
- nsys enabled flag
- optional nsys output prefix

Purpose:
- make paper runs reproducible
- avoid long hand-written shell invocations
- keep the comparison matrix explicit

This layer should describe experiments, not execute them.

### 2. Thin live-eval runner
Add a small runner that reads the manifest/config and executes runs by invoking `test/lora/benchmark_lora.sh`.

Responsibilities:
- select the requested run set
- map run definitions to benchmark arguments
- create output directories
- snapshot effective config/metadata per run
- optionally wrap the benchmark invocation with nsys for diagnostic runs

Non-responsibilities:
- do not replace `benchmark_lora.sh`
- do not reimplement server lifecycle logic
- do not compute paper metrics directly inside the runner

### 3. nsys wrapper
The runner should support a profiling mode that wraps the benchmark invocation with nsys only for diagnostic runs.

Behavior:
- invoke `nsys profile` around `benchmark_lora.sh`
- standardize output naming by run label
- write nsys artifacts into the diagnostic run directory
- record the exact command used in run metadata

This keeps profiling optional and isolated from the paper suite.

### 4. Minimal summary post-processor
Add a small post-processing step that consumes existing benchmark outputs and emits stable machine-readable summaries.

Inputs:
- per-request log emitted by `benchmark_lora.sh`
- run metadata/config snapshot
- optionally benchmark stdout if needed for throughput extraction

Outputs per valid run:
- summary JSON containing:
  - run label
  - suite kind
  - mode/baseline label
  - request count
  - success rate
  - throughput
  - latency p50/p95/p99
  - pointers to raw artifacts
  - validity flag

Merged outputs:
- comparison CSV across Stage A paper runs
- top-level summary JSON for all discovered runs

Recommendation:
- keep this as a separate parser instead of extending `benchmark_lora.sh` itself with paper-specific formatting

## Data flow

### Stage A: paper suite
1. Select the fixed small request slice.
2. Execute the three canonical configurations through `benchmark_lora.sh`.
3. Persist raw run artifacts.
4. Run the summary post-processor.
5. Emit merged comparison outputs for paper use.

### Stage B: diagnosis suite
1. Reuse one or more Stage A configurations.
2. Execute the same benchmark flow with nsys wrapping enabled.
3. Persist raw profiling artifacts and matching metadata.
4. Optionally summarize the request-level results, but do not use profiled runs as default headline paper numbers.

## Artifact layout

All new outputs should live under `artifacts/evaluation/...`.

Proposed structure:

- `artifacts/evaluation/live_e2e/<run_id>/paper_runs/<run_label>/`
- `artifacts/evaluation/live_e2e/<run_id>/diagnostic_runs/<run_label>/`
- `artifacts/evaluation/live_e2e/<run_id>/summaries/`

Per-run artifacts:
- benchmark stdout/stderr log
- server log
- per-request log
- effective config snapshot
- run metadata
- optional nsys artifact for diagnostic runs

Summary artifacts:
- `artifacts/evaluation/live_e2e/<run_id>/summaries/live_e2e_summary.json`
- `artifacts/evaluation/live_e2e/<run_id>/summaries/live_e2e_comparison.csv`

## Boundaries

### `test/lora/benchmark_lora.sh`
Owns:
- server launch
- request traffic generation
- cleanup
- raw logs
- per-request metrics generation

### New live-eval runner
Owns:
- experiment selection
- argument assembly
- output layout
- optional nsys wrapping
- config snapshotting

### New summary post-processor
Owns:
- stable paper-facing summaries
- validation status per run
- merged comparison tables

This keeps each layer focused and rerunnable.

## Failure handling

A run is valid only if:
- the server starts successfully
- the benchmark phase completes
- the per-request log exists
- enough request results exist to compute the summary

A run is failed if:
- server startup times out
- request execution fails or is incomplete
- required artifacts are missing
- nsys was requested but the expected profiling artifact is absent

Failed runs should still write metadata including:
- run label
- effective config
- failure stage
- paths to partial logs
- validity flag set to false

The merged paper comparison should include only valid Stage A runs.

## Verification strategy

For Stage A paper suite:
- verify that all three canonical configurations produced valid summary records
- verify that merged comparison outputs exist
- verify that request count and latency fields are populated for each included run

For Stage B diagnosis suite:
- verify that the matching nsys artifact exists
- verify that the run metadata records the wrapped command and output path

## Testing strategy

Focus tests on the new layers, not on re-testing the server implementation.

### Unit tests
- manifest/config parsing for the canonical paper suite
- command construction for benchmark invocations
- command construction for nsys-wrapped invocations
- summary parsing from per-request logs
- failure classification when required artifacts are missing
- output path generation under `artifacts/evaluation/live_e2e/<run_id>/...`

### Smoke/integration tests
- narrow dry-run or command-generation test for one paper run
- narrow dry-run or command-generation test for one diagnostic nsys run

### Manual validation
Document one explicit local smoke command for:
- paper suite single-run execution
- diagnostic suite single-run execution with nsys

## Recommended implementation sequence

1. Add the experiment manifest/config shape.
2. Add the thin live-eval runner that invokes `benchmark_lora.sh`.
3. Add optional nsys wrapping for diagnostic runs.
4. Add the summary post-processor for per-run JSON and merged CSV.
5. Add focused tests for manifest parsing, command construction, output layout, and summary parsing.
6. Run one manual paper-suite smoke test and one diagnostic nsys smoke test.

## Out of scope

- Replacing `benchmark_lora.sh` as the orchestrator
- Expanding the first paper matrix beyond baseline + `execution_first` + `load_then_run`
- Prometheus-first observability
- Large-window live replay as the canonical paper path
- Mixing diagnostic profiled runs into the default paper table

## Open assumptions made explicit

To keep this design concrete, the following assumptions are fixed for the first version:
- baseline is LoRA GPU/GPU
- the fixed small request slice is the canonical paper workload
- nsys is diagnosis-only by default
- summaries are generated in a separate post-processing layer rather than inside `benchmark_lora.sh`
- all outputs land under `artifacts/evaluation/live_e2e/<run_id>/...`
