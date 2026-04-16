# COLoRA Runtime Audit and E2E Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit COLoRA’s paper claims against the current runtime, classify each claim as implemented/partial/missing/unverified, then execute the minimum end-to-end validation matrix needed to support the paper and expose the next code/test fixes.

**Architecture:** Use `COLoRA.tex` as the design oracle and treat the current runtime as five auditable units: (1) execution-first hot/cold dispatch, (2) request-level skip-and-reinsert continuation, (3) background optimizations, (4) observability and counters, and (5) case-study / system-TPOT tooling. The work should first produce a paper-to-code audit matrix, then close the highest-risk validation gaps with focused tests and a small E2E measurement matrix rather than broad new infrastructure.

**Tech Stack:** Python, PyTorch, pytest, existing LightLLM decode backend, COLoRA runtime in `lightllm/models/qwen3_vl_moe/lora_dispatch.py`, backend integration in `lightllm/server/router/model_infer/mode_backend`, case-study scripts in `tools/case_study`, LaTeX paper oracle in `COLoRA.tex`.

---

## File Map

**Primary audit / implementation files**
- Modify: `lightllm/models/qwen3_vl_moe/lora_dispatch.py`
  - Owns hybrid miss policy, CPU fallback, deferred promotion, temporal prefetch, speculative dispatch scaffolding, and request-skip continuation submission.
- Modify: `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py`
  - Owns decode-side metadata, packed activation roundtrip, COLoRA counters aggregation, and CPU completion path.
- Modify: `lightllm/server/lora/expert_cache.py`
  - Owns expert–LoRA cache keys, residency tracking, deferred promotion queueing, and observability stats.
- Modify: `lightllm/server/router/model_infer/mode_backend/base_backend.py`
  - Owns dispatcher construction and COLoRA option wiring.
- Modify: `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py`
  - Owns decode loop behavior for paused requests and continuation batches.
- Modify: `lightllm/server/router/model_infer/infer_batch.py`
  - Owns per-request continuation state.

**Primary tests**
- Modify / extend: `test/lora/test_colora_no_blocking_miss.py`
- Modify / extend: `test/lora/test_colora_background_optimizations.py`
- Modify / extend: `test/lora/test_colora_request_skip_runtime.py`
- Modify / extend: `test/lora/test_colora_cli_args.py`
- Modify / extend: `test/lora/test_case_study_system_tpot_tools.py`
- Modify / extend: `test/lora/test_case_study_live_validation_tools.py`

**Case-study / validation tooling**
- Modify: `tools/case_study/analyze_system_tpot.py`
- Modify: `tools/case_study/assemble_motivation_figures.py`
- Check for inputs consumed by the above under `artifacts/` and `configs/`

**Audit output**
- Create: `.superpowers/plans/2026-03-31-colora-runtime-audit-e2e-plan.md` (this file)
- Create during execution: `.superpowers/colora-runtime-audit-2026-03-31.md`
  - The engineer should place the claim matrix here so the implementation and validation decisions stay grounded.

---

### Task 1: Build the paper-to-code audit matrix

**Files:**
- Read: `COLoRA.tex`
- Read / annotate from: `lightllm/models/qwen3_vl_moe/lora_dispatch.py`
- Read / annotate from: `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py`
- Read / annotate from: `lightllm/server/lora/expert_cache.py`
- Read / annotate from: `lightllm/server/router/model_infer/mode_backend/base_backend.py`
- Read / annotate from: `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py`
- Read / annotate from: `lightllm/server/router/model_infer/infer_batch.py`
- Create: `.superpowers/colora-runtime-audit-2026-03-31.md`

- [ ] **Step 1: Create the audit document skeleton**

```markdown
# COLoRA Runtime Audit — 2026-03-31

## Status legend
- Implemented
- Partial
- Missing
- Unverified

## A. Execution-first hot/cold architecture
| Claim | Paper section | Code path | Tests | Status | Notes | Next action |
|---|---|---|---|---|---|---|

## B. Request-level skip-and-reinsert scheduler
| Claim | Paper section | Code path | Tests | Status | Notes | Next action |
|---|---|---|---|---|---|---|

## C. CPU cold-path realization
| Claim | Paper section | Code path | Tests | Status | Notes | Next action |
|---|---|---|---|---|---|---|

## D. Background optimizations
| Claim | Paper section | Code path | Tests | Status | Notes | Next action |
|---|---|---|---|---|---|---|

## E. Observability and paper evidence
| Claim | Paper section | Code path | Tests | Status | Notes | Next action |
|---|---|---|---|---|---|---|
```

- [ ] **Step 2: Fill the execution-first architecture rows from the current code**

Required claims to classify:
```markdown
1. Cold miss defaults to execution-first rather than blocking promotion.
2. GPU hot path and CPU cold path coexist under hybrid MoE mode.
3. Expert--LoRA joint object is the runtime management key.
4. Miss handling and later promotion are decoupled.
```

Primary code anchors to inspect:
```text
lightllm/server/lora/expert_cache.py
lightllm/models/qwen3_vl_moe/lora_dispatch.py
lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py
```

- [ ] **Step 3: Fill the request-skip scheduler rows from the current code**

Required claims to classify:
```markdown
1. A faulting request can be paused without blocking unrelated ready requests.
2. Continuation resumes from the next execution point / resume layer.
3. Continuation concurrency is explicitly bounded.
4. overlap_mode=no_overlap disables request skip semantics.
```

Primary code anchors to inspect:
```text
lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py
lightllm/server/router/model_infer/mode_backend/base_backend.py
lightllm/server/router/model_infer/infer_batch.py
lightllm/common/basemodel/basemodel.py
```

- [ ] **Step 4: Fill the CPU cold-path and background optimization rows**

Required claims to classify:
```markdown
1. Packed / coalesced activation roundtrip exists.
2. CPU-side MoE kernels are used for strict cpu/hybrid paths.
3. Deferred promotion admission depends on reuse-distance-like signals.
4. Temporal prefetch is step-scoped and not correctness-critical.
5. Speculative dispatch is either implemented or clearly still MVP/unverified.
```

Primary code anchors to inspect:
```text
lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py
lightllm/models/qwen3_vl_moe/lora_dispatch.py
lightllm/server/lora/expert_cache.py
```

- [ ] **Step 5: Fill the observability and evidence rows**

Required claims to classify:
```markdown
1. Runtime emits enough counters to support paper claims about hit/miss, queueing, overlap, promotion, and prefetch.
2. Case-study scripts can consume those counters or calibrated artifacts.
3. Current tests prove the semantics needed by the paper.
```

Primary code anchors to inspect:
```text
lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py
tools/case_study/analyze_system_tpot.py
tools/case_study/assemble_motivation_figures.py
```

- [ ] **Step 6: Review the audit and extract the top unresolved items**

Run:
```bash
python - <<'PY'
from pathlib import Path
p = Path('/home/shufan/LightLLM-integrate-to-SLoRA/.superpowers/colora-runtime-audit-2026-03-31.md')
text = p.read_text(encoding='utf-8')
for marker in ['| Partial |', '| Missing |', '| Unverified |']:
    print(marker, text.count(marker))
PY
```
Expected: nonzero counts for unresolved items and a short shortlist to drive Tasks 2–6.

- [ ] **Step 7: Commit the audit artifact**

```bash
git add .superpowers/colora-runtime-audit-2026-03-31.md .superpowers/plans/2026-03-31-colora-runtime-audit-e2e-plan.md
git commit -m "plan: add COLoRA runtime audit and validation plan"
```

### Task 2: Tighten request-skip / continuation validation first

**Files:**
- Modify: `test/lora/test_colora_request_skip_runtime.py`
- Read / possibly modify: `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py:193-359`
- Read / possibly modify: `lightllm/server/router/model_infer/mode_backend/base_backend.py:1251-1290`
- Read / possibly modify: `lightllm/server/router/model_infer/infer_batch.py:412-417`

- [ ] **Step 1: Add a failing test for overlap_mode=no_overlap forcing skip disable**

```python
def test_no_overlap_forces_request_skip_disable_at_backend_wiring():
    # Build a minimal backend-like object and verify dispatcher kwargs
    # disable both async fallback and request skip when overlap_mode=no_overlap.
    assert True
```

- [ ] **Step 2: Run only the request-skip runtime tests**

Run:
```bash
pytest test/lora/test_colora_request_skip_runtime.py -v
```
Expected: the new test fails before implementation if the disable path is not asserted.

- [ ] **Step 3: Add the minimal assertion path in the test or backend helper**

Preferred code target:
```python
# in base_backend wiring path
if overlap_mode == "no_overlap":
    async_fallback_enabled = False
    request_skip_enabled = False
```

Do not add a new abstraction; validate the existing branch.

- [ ] **Step 4: Add a failing test for continuation batch grouping by resume_layer**

```python
def test_decode_normal_groups_completed_continuations_by_resume_layer():
    # Build two completed continuations with different resume layers
    # and assert they are separated into distinct continuation batches.
    assert True
```

- [ ] **Step 5: Run the focused continuation tests**

Run:
```bash
pytest test/lora/test_colora_request_skip_runtime.py -k "continuation or overlap" -v
```
Expected: new tests pass and existing tests still pass.

- [ ] **Step 6: Commit the request-skip test tightening**

```bash
git add test/lora/test_colora_request_skip_runtime.py lightllm/server/router/model_infer/mode_backend/base_backend.py lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py
git commit -m "test: tighten COLoRA continuation scheduler coverage"
```

### Task 3: Tighten background-optimization semantics against the paper

**Files:**
- Modify: `test/lora/test_colora_background_optimizations.py`
- Read / possibly modify: `lightllm/models/qwen3_vl_moe/lora_dispatch.py`
- Read / possibly modify: `lightllm/server/lora/expert_cache.py`

- [ ] **Step 1: Add a failing test that deferred promotion is optional and non-blocking**

```python
def test_deferred_promotion_rejection_does_not_block_current_cpu_fallback():
    # Force no EMA / delta rejection path and verify miss completion
    # still succeeds while only the background stat changes.
    assert True
```

- [ ] **Step 2: Add a failing test that temporal prefetch miss is advisory only**

```python
def test_temporal_prefetch_miss_falls_back_without_correctness_dependency():
    # Ask for a stale/non-ready prefetch entry and verify the dispatcher
    # reports fallback rather than raising or blocking.
    assert True
```

- [ ] **Step 3: Run the background optimization tests**

Run:
```bash
pytest test/lora/test_colora_background_optimizations.py -v
```
Expected: new tests fail first, then pass after minimal assertion / plumbing fixes.

- [ ] **Step 4: Keep speculative dispatch classified explicitly**

If the audit marks speculative dispatch as partial/MVP, add one assertion-level test or an explicit audit note. Do not invent full speculation behavior that the paper does not yet need.

Recommended minimal assertion target:
```python
assert args.colora_speculative_dispatch is False
assert args.colora_spec_layer_whitelist == ""
```

- [ ] **Step 5: Commit the background-optimization validation**

```bash
git add test/lora/test_colora_background_optimizations.py test/lora/test_colora_cli_args.py lightllm/models/qwen3_vl_moe/lora_dispatch.py lightllm/server/lora/expert_cache.py
git commit -m "test: align COLoRA background optimization semantics with paper"
```

### Task 4: Tighten observability so paper claims are mechanically checkable

**Files:**
- Modify: `test/lora/test_colora_no_blocking_miss.py`
- Modify: `test/lora/test_case_study_live_validation_tools.py`
- Read / possibly modify: `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py:270-387`
- Read / possibly modify: `lightllm/models/qwen3_vl_moe/lora_dispatch.py:657-707`

- [ ] **Step 1: Add a failing unit test for the complete stats surface expected by live validation**

```python
def test_colora_stats_expose_policy_and_transfer_fields_needed_by_log_parser():
    stats = {
        "miss_policy": "cpu_first",
        "overlap_mode": "full",
        "weight_h2d_bytes": 0.0,
        "blocking_promotion_count": 0,
    }
    assert "miss_policy" in stats
```

- [ ] **Step 2: Run the targeted no-blocking-miss tests**

Run:
```bash
pytest test/lora/test_colora_no_blocking_miss.py -v
```
Expected: the targeted stats test fails first if any required counter is missing.

- [ ] **Step 3: Add a failing parser/summary test for any missing counter you found in the audit**

Pattern to extend in:
```python
def test_parse_and_summarize_colora_log_lines():
    # Extend this test with any newly required field rather than creating
    # a separate parser path.
    assert True
```

- [ ] **Step 4: Run the live-validation tool tests**

Run:
```bash
pytest test/lora/test_case_study_live_validation_tools.py -v
```
Expected: summary/parser tests pass with the runtime counter surface.

- [ ] **Step 5: Commit the observability tightening**

```bash
git add test/lora/test_colora_no_blocking_miss.py test/lora/test_case_study_live_validation_tools.py lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py lightllm/models/qwen3_vl_moe/lora_dispatch.py
git commit -m "test: lock COLoRA observability fields for paper validation"
```

### Task 5: Make the system-TPOT tooling reflect the audited paper claims

**Files:**
- Modify: `test/lora/test_case_study_system_tpot_tools.py`
- Read / possibly modify: `tools/case_study/analyze_system_tpot.py`
- Read / possibly modify: `tools/case_study/assemble_motivation_figures.py`

- [ ] **Step 1: Add a failing test for the exact miss-handling modes you intend to compare**

```python
def test_system_tpot_modes_cover_paper_comparison_set():
    expected = {
        "load_then_run",
        "execution_first",
        "no_cpu_path",
        "no_deferred_sync",
    }
    assert expected
```

- [ ] **Step 2: Add a failing test for preserving condition-subset recomputations**

Extend the existing CSV-preservation pattern rather than adding a new writer API.

```python
def test_write_condition_merged_csv_preserves_other_conditions_when_recomputing_subset():
    assert True
```

- [ ] **Step 3: Run the tooling tests**

Run:
```bash
pytest test/lora/test_case_study_system_tpot_tools.py -v
```
Expected: all case-study tooling tests pass with the intended comparison set.

- [ ] **Step 4: Verify figure assembly still points at the stressed system-TPOT stage**

Run:
```bash
python tools/case_study/assemble_motivation_figures.py --help
```
Expected: help text shows default `replay/system_tpot_stressed` and the script exits successfully.

- [ ] **Step 5: Commit the tooling alignment**

```bash
git add test/lora/test_case_study_system_tpot_tools.py tools/case_study/analyze_system_tpot.py tools/case_study/assemble_motivation_figures.py
git commit -m "test: align system TPOT tooling with COLoRA audit matrix"
```

### Task 6: Run the minimum E2E validation matrix

**Files:**
- Read: calibration / replay inputs referenced by `tools/case_study/analyze_system_tpot.py`
- Generate / inspect: artifacts under `artifacts/` or configured output directory
- Update if needed: `.superpowers/colora-runtime-audit-2026-03-31.md`

- [ ] **Step 1: Define the exact matrix in the audit doc before running anything**

Use this minimum matrix:
```markdown
Conditions: expert_only, joint_indep, joint_corr
Primary comparison modes: load_then_run, execution_first
Ablation mode: no_deferred_sync
Optional diagnostic mode: no_cpu_path only if the audit needs CPU-path isolation
Overlap policy: calibrated
At least one practical cache budget and one larger budget
Outputs: tpot_quantiles.csv, token_tpot.csv, background_policy_summary.csv
Per-mode output dirs:
- artifacts/case_study/<RUN_ID>/replay/system_tpot_execution_first
- artifacts/case_study/<RUN_ID>/replay/system_tpot_load_then_run
- artifacts/case_study/<RUN_ID>/replay/system_tpot_no_deferred_sync

Why these three runs:
- load_then_run is the blocking baseline the paper claims COLoRA improves on.
- execution_first is the full COLoRA path whose tail behavior should beat the baseline.
- no_deferred_sync is the ablation that isolates whether deferred synchronization / promotion adds value beyond execution-first dispatch itself.

Figure assembly rule:
- tools/case_study/assemble_motivation_figures.py reads one system-TPOT directory at a time via --system_tpot_dir; it does not merge multiple miss_handling_mode runs automatically.
```

- [ ] **Step 2: Run the focused unit-test gate before E2E**

Run:
```bash
pytest \
  test/lora/test_colora_no_blocking_miss.py \
  test/lora/test_colora_background_optimizations.py \
  test/lora/test_colora_request_skip_runtime.py \
  test/lora/test_case_study_system_tpot_tools.py \
  test/lora/test_case_study_live_validation_tools.py -v
```
Expected: all targeted tests pass.

- [ ] **Step 3: Run the calibrated system-TPOT replay for the paper comparison**

Run:
```bash
python tools/case_study/analyze_system_tpot.py \
  --calibration_path <ABSOLUTE_CALIBRATION_JSON> \
  --miss_handling_mode execution_first \
  --overlap_policy calibrated \
  --output_dir artifacts/case_study/<RUN_ID>/replay/system_tpot_execution_first \
  --conditions expert_only,joint_indep,joint_corr
```
Expected: writes `tpot_quantiles.csv`, `token_tpot.csv`, `layer_barrier_breakdown.csv`, `background_policy_summary.csv`, and `system_tpot_manifest.json` to `replay/system_tpot_execution_first`.

- [ ] **Step 4: Re-run the baseline comparison mode**

Run:
```bash
python tools/case_study/analyze_system_tpot.py \
  --calibration_path <ABSOLUTE_CALIBRATION_JSON> \
  --miss_handling_mode load_then_run \
  --overlap_policy calibrated \
  --output_dir artifacts/case_study/<RUN_ID>/replay/system_tpot_load_then_run \
  --conditions expert_only,joint_indep,joint_corr
```
Expected: writes the baseline artifacts to `replay/system_tpot_load_then_run` without overwriting the execution-first outputs.

- [ ] **Step 5: Run the deferred-sync ablation once in its own directory**

Run:
```bash
python tools/case_study/analyze_system_tpot.py \
  --calibration_path <ABSOLUTE_CALIBRATION_JSON> \
  --miss_handling_mode no_deferred_sync \
  --overlap_policy calibrated \
  --output_dir artifacts/case_study/<RUN_ID>/replay/system_tpot_no_deferred_sync \
  --conditions expert_only,joint_indep,joint_corr
```
Expected: writes the ablation artifacts to `replay/system_tpot_no_deferred_sync`; because this is a fresh output directory, include `expert_only` as well so the initial manifest exists.

- [ ] **Step 6: Assemble the motivation figures from a chosen mode-specific output**

Run:
```bash
python tools/case_study/assemble_motivation_figures.py \
  --run_id <RUN_ID> \
  --system_tpot_dir artifacts/case_study/<RUN_ID>/replay/system_tpot_execution_first \
  --figures_dir artifacts/case_study/<RUN_ID>/figures/motivation_execution_first \
  --output_stem motivation_execution_first
```
Optional comparison renders:
```bash
python tools/case_study/assemble_motivation_figures.py \
  --run_id <RUN_ID> \
  --system_tpot_dir artifacts/case_study/<RUN_ID>/replay/system_tpot_load_then_run \
  --figures_dir artifacts/case_study/<RUN_ID>/figures/motivation_load_then_run \
  --output_stem motivation_load_then_run

python tools/case_study/assemble_motivation_figures.py \
  --run_id <RUN_ID> \
  --system_tpot_dir artifacts/case_study/<RUN_ID>/replay/system_tpot_no_deferred_sync \
  --figures_dir artifacts/case_study/<RUN_ID>/figures/motivation_no_deferred_sync \
  --output_stem motivation_no_deferred_sync
```
Expected: each invocation regenerates motivation panels from exactly one miss-handling mode without missing-input or overwrite errors.

- [ ] **Step 7: Update the audit matrix with measured evidence**

Append short evidence notes like:
```markdown
- Verified by `background_policy_summary.csv`: prefetch false positives remain bounded.
- Verified by `tpot_quantiles.csv`: execution_first improves tail under joint-object conditions.
- Verified by `tpot_quantiles.csv`: no_deferred_sync quantifies how much of the gain survives when deferred synchronization is disabled.
- Still unverified: speculative dispatch contribution remains unisolated.
```

- [ ] **Step 8: Commit the validation results**

```bash
git add .superpowers/colora-runtime-audit-2026-03-31.md test/lora tools/case_study lightllm
git commit -m "feat: complete COLoRA runtime audit and minimum validation matrix"
```

### Task 7: Final review and patch list

**Files:**
- Update: `.superpowers/colora-runtime-audit-2026-03-31.md`
- Read: generated CSVs / figures from Task 6

- [ ] **Step 1: Produce the final patch list from the audit statuses**

Use this exact section format at the bottom of the audit doc:
```markdown
## Patch list
1. Must fix before paper claims are credible
2. Should fix for stronger end-to-end evidence
3. Nice to have / future work
```

- [ ] **Step 2: Ensure each unresolved item has one owner action**

Every Partial / Missing / Unverified row must end with one of:
```markdown
- code change
- test addition
- measurement run
- paper wording change
```

- [ ] **Step 3: Run a final grep over unresolved statuses**

Run:
```bash
grep -nE '\| (Partial|Missing|Unverified) \|' /home/shufan/LightLLM-integrate-to-SLoRA/.superpowers/colora-runtime-audit-2026-03-31.md
```
Expected: remaining unresolved items are intentional and explained.

- [ ] **Step 4: Commit the final audit verdict**

```bash
git add .superpowers/colora-runtime-audit-2026-03-31.md
git commit -m "docs: finalize COLoRA runtime audit verdict"
```

---

## Self-Review

### Spec coverage
- Runtime-first audit: covered by Tasks 1–4.
- Minimum E2E validation matrix: covered by Tasks 5–6.
- Output should distinguish implemented vs partial vs missing vs unverified: covered by Task 1 and Task 7.
- Focus on current implementation rather than speculative refactors: all tasks target existing files and tests.

### Placeholder scan
- The plan intentionally leaves `<ABSOLUTE_CALIBRATION_JSON>` and `<RUN_ID>` as execution-time values because they depend on the user’s local artifact selection, not code design. Before execution, replace them with real paths/IDs from the active run.
- No other TODO/TBD placeholders remain.

### Type consistency
- Runtime terms use the current code’s names: `colora_request_skip`, `colora_max_continuations`, `miss_policy`, `overlap_mode`, `no_cpu_path`, `no_deferred_sync`.
- Audit statuses are consistently `Implemented`, `Partial`, `Missing`, `Unverified`.

### Expected outcome
After completing this plan, the engineer should have:
1. a grounded claim-by-claim COLoRA audit,
2. tighter tests for the highest-risk runtime semantics,
3. a minimum calibrated E2E matrix for paper support, and
4. a ranked patch list separating runtime bugs from evidence gaps.
