# Live E2E Config Guide (`configs/live_e2e`)

This guide explains what each manifest field does in practice for
`python -m tools.evaluation.live_e2e`.

## How a manifest run is translated

Each `runs[]` item is turned into `test/lora/benchmark_lora.sh` args by
`tools/evaluation/live_e2e/runner.py`.

Key mapping:

- `miss_handling_mode` -> `--colora_miss_policy <value>`
- `overlap_policy`:
  - `no_overlap` -> `--colora_request_skip 0`
  - any other value -> `--colora_request_skip 1`
- `async_fallback` -> `--colora_async_fallback 0|1`
- `cpu_workers` -> `--colora_cpu_workers N`
- `cpu_queue_depth` -> `--colora_cpu_queue_depth N`
- `cpu_batch_timeout_us` -> `--colora_cpu_batch_timeout_us N`
- `speculative_dispatch` -> `--colora_speculative_dispatch` / `--no_colora_speculative_dispatch`

Important: the current runner does **not** pass `--colora_overlap_mode` directly.
It approximates overlap policy with request-skip on/off.

## `miss_handling_mode` values (runtime truth)

From `lightllm/server/api_cli.py`, valid values are:

- `cpu_first`
- `load_then_run`
- `no_cpu_path`
- `no_deferred_sync`

### Meaning of each mode

- `load_then_run`
  - Promotion-first style for misses.
  - Try to promote/cache then run hot path.
  - Common baseline.

- `cpu_first`
  - Execution-first behavior in runtime terms.
  - Misses can execute on CPU fallback path first; promotion can be deferred.
  - Use this as the practical replacement for "execution_first" naming in manifests.

- `no_deferred_sync`
  - Misses may execute CPU path, then force blocking promotion sync path.
  - Useful ablation to isolate value of deferred promotion/synchronization.

- `no_cpu_path`
  - Disallow CPU fallback for unresolved misses.
  - Promotion/hot-path must satisfy misses or run fails.
  - Diagnostic mode, usually not primary paper baseline.

## Naming mismatch: `execution_first` vs `cpu_first`

Some older examples/docs use `execution_first` in manifests. Current CLI expects
`cpu_first`. If you use `execution_first` in `miss_handling_mode`, server arg
validation may fail.

Recommended convention:

- Keep `mode_label: execution_first_*` for readability in reports.
- Set `miss_handling_mode: cpu_first` for correctness.

## `overlap_policy` semantics in this repo

Current effective behavior in runner:

- `no_overlap` => batch-wait style, no skip-and-reinsert (`colora_request_skip=0`).
- non-`no_overlap` (e.g. `calibrated`, `skip_reinsert`) => skip-and-reinsert enabled (`colora_request_skip=1`).

Because runner uses only request-skip toggling, treat `overlap_policy` as
"scheduler mode selector" unless runner is extended to pass `--colora_overlap_mode`.

## `async_fallback` and CPU knobs

- `async_fallback: true`
  - Allows async CPU miss worker path (overlap potential).
- `async_fallback: false`
  - Forces synchronous CPU miss handling (naive/batch-wait style).

`cpu_workers`, `cpu_queue_depth`, `cpu_batch_timeout_us` tune async path.

Notes:

- Runtime clamps `colora_cpu_workers` to at least 1 internally.
- In sync mode (`async_fallback: false`), workers have limited impact.
- Use `cpu_workers: 1` in manifests to avoid confusion.

## Recommended config patterns

- Locked load-then-run baseline:
  - `miss_handling_mode: load_then_run`
  - `overlap_policy: no_overlap`
  - `async_fallback: false`

- Skip-and-reinsert stability run:
  - `miss_handling_mode: load_then_run`
  - `overlap_policy: skip_reinsert_stability` (or any non-`no_overlap`)
  - `async_fallback: false`

- Execution-first overlap run:
  - `miss_handling_mode: cpu_first`
  - `overlap_policy: calibrated` (non-`no_overlap`)
  - `async_fallback: true`
  - `cpu_workers: 2` (or 4), queue/depth timeout aligned with baseline.
