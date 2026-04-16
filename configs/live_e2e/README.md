# Live E2E Config Guide (`configs/live_e2e`)

This guide explains what each manifest field does in practice for
`python -m tools.evaluation.live_e2e`.

## How a manifest run is translated

Each `runs[]` item is turned into `test/lora/benchmark_lora.sh` args by
`tools/evaluation/live_e2e/runner.py`.

Set `cpu_kernel_mode` and `coalescing_packer` on every run so MoE CPU behavior is explicit and reproducible (repository manifests use `avx` + `true` except naive ablations, which use `naive` + `false`).

Key mapping:

- `miss_handling_mode` -> `--colora_miss_policy <value>`
- `overlap_mode` -> `--colora_overlap_mode <value>` (`full` or `no_overlap`)
- `overlap_policy`:
  - `no_overlap` -> `--colora_request_skip 0` (batch-wait scheduling)
  - any other value -> `--colora_request_skip 1` (skip-reinsert scheduling); prefer the label `request_skip` for clarity
- `async_fallback` -> `--colora_async_fallback 0|1`
- `cpu_workers` -> `--colora_cpu_workers N`
- `cpu_queue_depth` -> `--colora_cpu_queue_depth N`
- `cpu_batch_timeout_us` -> `--colora_cpu_batch_timeout_us N`
- `max_continuations` -> `--colora_max_continuations N`
- `temporal_prefetch` -> `--colora_temporal_prefetch | --no_colora_temporal_prefetch`
- `temporal_prefetch_layer_whitelist` -> `--colora_temporal_prefetch_layer_whitelist CSV`
- `temporal_hot_cache_slots` -> `--colora_temporal_hot_cache_slots N`
- `speculative_dispatch` (optional, deprecated): only add if you need to override the server default. Default is **off** (`colora_speculative_dispatch=False`); when omitted, the runner does not pass either `--colora_speculative_dispatch` or `--no_colora_speculative_dispatch`.
- `spec_layer_whitelist` -> `--colora_spec_layer_whitelist CSV`
- `cache_budget_mb` -> `--colora_cache_budget_mb MB`
- `promote_min_hits` -> `--colora_promote_min_hits N`
- `promote_window` -> `--colora_promote_window N`
- `max_promote_per_step` -> `--colora_max_promote_per_step N`
- `decay` -> `--colora_decay F`
- `deferred_promotion_delta_steps` -> `--colora_deferred_promotion_delta_steps N`
- `promotion_ema_alpha` -> `--colora_promotion_ema_alpha F`
- `cpu_kernel_mode` -> prepends `COLORA_CPU_KERNEL_MODE=<value>` (e.g. `avx`, `naive`) before the benchmark script
- `coalescing_packer` -> prepends `MOE_COALESCING_PACKER=0|1` before the benchmark script
- `warmup_adapter_trace_path` -> `--warmup_adapter_trace_path <jsonl>`
- `measurement_adapter_trace_path` -> `--measure_adapter_trace_path <jsonl>`
- `adapter_trace_path` (legacy alias) -> mapped to `measurement_adapter_trace_path` when explicit measurement field is not set
- `server_host` -> `--server_host <host>` (for fake server or custom endpoint)
- `server_port` -> `--server_port <port>` (for fake server or avoiding default-port collisions)

Note: `overlap_mode` and `overlap_policy` are now both available. Use `overlap_mode` for server-native overlap semantics and `overlap_policy` for explicit request-skip scheduling control.

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
- any other value => skip-reinsert path (`colora_request_skip=1`).

Recommended straightforward labels:

- `no_overlap` — counterpart to batch-wait scheduling (request skip **off**).
- `request_skip` — counterpart to skip-reinsert scheduling (request skip **on**); maps directly to `--colora_request_skip 1`.

Other strings (e.g. `skip_reinsert_stability`) still work: anything except `no_overlap` turns request skip **on**.

Because runner uses only request-skip toggling, treat `overlap_policy` as
"scheduler mode selector" and use `overlap_mode` for runtime overlap semantics.

## Real trace replay configuration

Real-trace replay currently reuses adapter-assignment traces (request prompts are
still synthetic in `test_moe_lora_api.py`).

Expected adapter trace JSONL row format:
- required: `adapter_id`
- optional: `arrival_idx`, `req_idx` (used for stable ordering)

Alibaba mapped traces from `tools/case_study/preprocess_gentd26_trace.py`
already match this schema. See:
- `configs/live_e2e/real_trace_alibaba_example.yaml`

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

## CPU kernel mode and MoE coalescing packer

These fields are optional. When set, the runner prepends shell-style env assignments
to the benchmark command (same process as the shell script).

| Manifest field | Environment variable | Effect |
|----------------|---------------------|--------|
| `cpu_kernel_mode: avx` | `COLORA_CPU_KERNEL_MODE=avx` | Strict MoE CPU path uses the MoE AVX-512 BF16 kernels (default when the variable is unset). |
| `cpu_kernel_mode: naive` | `COLORA_CPU_KERNEL_MODE=naive` | Strict MoE CPU path uses PyTorch `matmul` on CPU as a reference implementation (no AVX kernel requirement). |
| `coalescing_packer: true` | `MOE_COALESCING_PACKER=1` | Enables the coalescing packer in the MoE layer when supported. |
| `coalescing_packer: false` | `MOE_COALESCING_PACKER=0` | Disables it. |

Single-run ablation manifests: `ablation_ExecuteFirst_Overlap_RequestSkip_AsyncDefault.yaml`,
`ablation_ExecuteFirst_Avx_CoalescingPackerOn.yaml`, and
`ablation_ExecuteFirst_Naive_CoalescingPackerOff.yaml`.

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
  - `overlap_policy: request_skip`
  - `async_fallback: true`
  - `cpu_workers: 2` (or 4), queue/depth timeout aligned with baseline.
