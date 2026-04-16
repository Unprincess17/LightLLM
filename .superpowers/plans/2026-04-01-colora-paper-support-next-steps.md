# COLoRA Paper-Support Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate a real execution-first calibration manifest with `cold_path_curves`, rerun the blocked `execution_first` / `no_deferred_sync` system-TPOT evaluations, and close the highest-value remaining evidence gaps with focused runtime tests and overlap measurement.

**Architecture:** Treat the remaining paper-support work as four linked units: (1) calibration generation, (2) replay/evaluation, (3) targeted runtime tests, and (4) evidence consolidation. The highest-priority path is to use the existing calibration producer in `tools/case_study/calibrate_system_baseline.py` to emit a manifest containing `cold_path_curves`, because `tools/case_study/system_tpot_core.py:214-287` hard-blocks `execution_first` and `no_deferred_sync` replays without that structure.

**Tech Stack:** Python, PyTorch, pytest, CUDA timing/NVTX, COLoRA runtime in `lightllm/models/qwen3_vl_moe/lora_dispatch.py`, cold-path roundtrip in `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py`, case-study tooling in `tools/case_study/`.

---

## File Map

**Calibration / evaluation**
- Read / run: `tools/case_study/calibrate_system_baseline.py:47-69`
  - Produces the real calibration manifest. `cold_path_curves` are written at `tools/case_study/calibrate_system_baseline.py:467-706`.
- Read / validate against: `tools/case_study/system_tpot_core.py:214-287`
  - `validate_execution_first_calibration()` requires `cold_path_curves.profiles.<load_profile>.{early,late}.{pack_rows,d2h_rows,cpu_rows,h2d_rows,merge_rows}`.
- Read / run: `tools/case_study/analyze_system_tpot.py`
  - Consumes calibration and writes `tpot_quantiles.csv`, `token_tpot.csv`, `background_policy_summary.csv`, and related replay outputs.
- Read / run: `tools/case_study/assemble_motivation_figures.py`
  - Regenerates the motivation figure panels from replay outputs.
- Inspect / overwrite: `artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calibration.json`
- Inspect / overwrite: `artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json`

**Runtime test targets**
- Read / possibly modify: `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py:409-508`
  - `_dispatch_lora_with_coalesced_cpu_roundtrip` target for the packed/coalesced CPU roundtrip unit test.
- Read / possibly modify: `lightllm/models/qwen3_vl_moe/lora_dispatch.py:1393-1772`
  - Speculative dispatch lifecycle target.
- Modify / extend: `test/lora/test_colora_background_optimizations.py`
  - Existing home for dispatcher-level async/background semantics tests.
- Modify / extend: `test/lora/test_case_study_system_tpot_tools.py`
  - Existing home for calibration/replay validation tests.

**Evidence / audit artifact**
- Update: `.superpowers/colora-runtime-audit-2026-03-31.md`
  - Append measured evidence and convert unresolved rows where possible.
- Create: `.superpowers/plans/2026-04-01-colora-paper-support-next-steps.md` (this file)

---

### Task 1: Regenerate a real execution-first calibration manifest with `cold_path_curves`

**Files:**
- Read / run: `tools/case_study/calibrate_system_baseline.py:47-69`
- Read / validate against: `tools/case_study/system_tpot_core.py:214-287`
- Overwrite: `artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json`
- Optionally also overwrite: `artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calibration.json`
- Test: `test/lora/test_case_study_system_tpot_tools.py`

- [ ] **Step 1: Add a failing test that the v2 calibration fixture exposes execution-first cold-path curves**

```python
from pathlib import Path
import json


def test_case_study_v2_calibration_contains_execution_first_cold_path_curves():
    calibration_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    system_tpot_mod.validate_execution_first_calibration(
        calibration=calibration,
        load_profile="stressed",
    )
```

- [ ] **Step 2: Run the single calibration-validation test and confirm it fails on the current artifact**

Run:
```bash
pytest test/lora/test_case_study_system_tpot_tools.py::test_case_study_v2_calibration_contains_execution_first_cold_path_curves -v
```
Expected: FAIL with `execution-first replay requires a populated execution-first calibration manifest`.

- [ ] **Step 3: Run the real calibration producer against the v2 output path**

Run:
```bash
python tools/case_study/calibrate_system_baseline.py \
  --config configs/global.yaml \
  --output_path /home/shufan/LightLLM-integrate-to-SLoRA/artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json \
  --base_tpot_ms 1:1.20,2:1.45,4:1.90 \
  --host_profiles idle,stressed
```
Expected: console ends with `wrote calibration manifest:` and the new JSON contains `cold_path_curves` plus `cold_path_config`.

- [ ] **Step 4: Verify the regenerated manifest contains the required cold-path structure**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/home/shufan/LightLLM-integrate-to-SLoRA/artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json')
payload = json.loads(p.read_text(encoding='utf-8'))
profile = payload['cold_path_curves']['profiles']['stressed']
print(sorted(payload['cold_path_curves']['profiles'].keys()))
print(sorted(profile['early'].keys()))
print(sorted(profile['late'].keys()))
PY
```
Expected:
```text
['idle', 'stressed']
['cpu_rows', 'd2h_rows', 'h2d_rows', 'merge_rows', 'pack_rows']
['cpu_rows', 'd2h_rows', 'h2d_rows', 'merge_rows', 'pack_rows']
```

- [ ] **Step 5: Re-run the failing test and confirm the artifact now validates**

Run:
```bash
pytest test/lora/test_case_study_system_tpot_tools.py::test_case_study_v2_calibration_contains_execution_first_cold_path_curves -v
```
Expected: PASS.

- [ ] **Step 6: If the baseline JSON is still used elsewhere, regenerate it too**

Run:
```bash
python tools/case_study/calibrate_system_baseline.py \
  --config configs/global.yaml \
  --output_path /home/shufan/LightLLM-integrate-to-SLoRA/artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calibration.json \
  --base_tpot_ms 1:1.20,2:1.45,4:1.90 \
  --host_profiles idle,stressed
```
Expected: both canonical calibration files share the same execution-first schema.

- [ ] **Step 7: Commit the regenerated calibration and validation lock**

```bash
git add \
  artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json \
  artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calibration.json \
  test/lora/test_case_study_system_tpot_tools.py
git commit -m "test: lock execution-first calibration manifest schema"
```

### Task 2: Rerun the blocked system-TPOT replay modes using the regenerated calibration

**Files:**
- Read / run: `tools/case_study/analyze_system_tpot.py`
- Read / validate against: `tools/case_study/system_tpot_core.py:214-287`
- Read / regenerate: `tools/case_study/assemble_motivation_figures.py`
- Inspect outputs under: `artifacts/case_study/router_lora_case_v1/replay/`
- Update: `.superpowers/colora-runtime-audit-2026-03-31.md`

- [ ] **Step 1: Add a failing test that execution-first calibration validation is expected to pass for the checked-in v2 artifact**

```python
from pathlib import Path
import json


def test_validate_execution_first_calibration_accepts_checked_in_v2_manifest():
    calibration_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    system_tpot_mod.validate_execution_first_calibration(
        calibration=calibration,
        load_profile="stressed",
        tool_name="pytest",
    )
```

- [ ] **Step 2: Run the system-TPOT tooling test file**

Run:
```bash
pytest test/lora/test_case_study_system_tpot_tools.py -v
```
Expected: all tooling tests pass before starting replay.

- [ ] **Step 3: Run the `execution_first` replay on the stressed profile**

Run:
```bash
python tools/case_study/analyze_system_tpot.py \
  --calibration_path /home/shufan/LightLLM-integrate-to-SLoRA/artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json \
  --miss_handling_mode execution_first \
  --overlap_policy calibrated \
  --conditions expert_only,joint_indep,joint_corr
```
Expected: replay completes without calibration error and writes updated CSVs including `tpot_quantiles.csv`, `token_tpot.csv`, and `background_policy_summary.csv`.

- [ ] **Step 4: Run the `no_deferred_sync` diagnostic replay**

Run:
```bash
python tools/case_study/analyze_system_tpot.py \
  --calibration_path /home/shufan/LightLLM-integrate-to-SLoRA/artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json \
  --miss_handling_mode no_deferred_sync \
  --overlap_policy calibrated \
  --conditions joint_indep,joint_corr
```
Expected: replay completes and isolates the value of deferred promotion relative to execution-first.

- [ ] **Step 5: Re-run the `load_then_run` baseline if needed to refresh merged CSVs on the same artifact**

Run:
```bash
python tools/case_study/analyze_system_tpot.py \
  --calibration_path /home/shufan/LightLLM-integrate-to-SLoRA/artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json \
  --miss_handling_mode load_then_run \
  --overlap_policy calibrated \
  --conditions expert_only,joint_indep,joint_corr
```
Expected: merged outputs preserve all selected conditions and allow apples-to-apples comparison.

- [ ] **Step 6: Regenerate the motivation figures from the refreshed replay outputs**

Run:
```bash
python tools/case_study/assemble_motivation_figures.py --run_id router_lora_case_v1
```
Expected: motivation panel PDFs regenerate without missing-input errors.

- [ ] **Step 7: Record the concrete evidence in the audit document**

```markdown
## Measured evidence — 2026-04-01

### Execution-first replay
- Verified by `tpot_quantiles.csv`: execution_first now runs on the checked-in calibration manifest with populated `cold_path_curves`.
- Verified by `background_policy_summary.csv`: deferred promotion / temporal prefetch counters are non-zero only in modes that admit them.
- Verified by `no_deferred_sync`: removing deferred promotion support changes the joint-object tail relative to execution_first.
```

- [ ] **Step 8: Commit the replay outputs and audit update**

```bash
git add \
  .superpowers/colora-runtime-audit-2026-03-31.md \
  artifacts/case_study/router_lora_case_v1 \
  test/lora/test_case_study_system_tpot_tools.py
git commit -m "feat: rerun execution-first system TPOT evaluation"
```

### Task 3: Add a focused unit test for the packed/coalesced CPU roundtrip

**Files:**
- Read / possibly modify: `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py:409-508`
- Modify: `test/lora/test_colora_background_optimizations.py`

- [ ] **Step 1: Add a failing unit test that exercises pack -> CPU dispatch -> scatter on the CPU path**

```python
import torch


def test_coalesced_cpu_roundtrip_scatter_preserves_token_positions():
    layer = object.__new__(dispatch_layer_mod.Qwen3MOETransformerLayerInfer)
    token_positions = torch.tensor([2, 0], dtype=torch.long)
    packed_bins = torch.tensor([1, 3], dtype=torch.long)
    packed_input = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    input_tensor = torch.zeros((4, 2), dtype=torch.float32)

    pack_meta = {
        "packed_activations": packed_input,
        "packed_bins": packed_bins,
        "token_positions": token_positions,
    }

    def fake_dispatch(x, layer_id, bins, expert_id):
        assert layer_id == 5
        assert expert_id == 7
        assert bins.tolist() == [1, 3]
        return x + 1.0

    out = layer._dispatch_lora_with_coalesced_cpu_roundtrip(
        dispatch_fn=fake_dispatch,
        input_tensor=input_tensor,
        layer_id=5,
        expert_id=7,
        pack_meta=pack_meta,
        reuse_packed_input=True,
        phase_name="gate",
        study2_prefix=None,
    )

    expected = torch.zeros((4, 2), dtype=torch.float32)
    expected[2] = torch.tensor([11.0, 21.0])
    expected[0] = torch.tensor([31.0, 41.0])
    assert torch.equal(out, expected)
```

- [ ] **Step 2: Run just the new roundtrip test**

Run:
```bash
pytest test/lora/test_colora_background_optimizations.py -k coalesced_cpu_roundtrip -v
```
Expected: FAIL first if the test harness needs helper binding or scatter setup.

- [ ] **Step 3: Add the minimal test harness support without changing runtime semantics**

```python
layer._scatter_lora_from_packed = dispatch_layer_mod.Qwen3MOETransformerLayerInfer._scatter_lora_from_packed.__get__(
    layer,
    dispatch_layer_mod.Qwen3MOETransformerLayerInfer,
)
```

Keep the runtime code unchanged unless the test exposes a real bug.

- [ ] **Step 4: Extend coverage to the non-reused pack path if the first test passes cleanly**

```python
pack_meta = {
    "packed_bins": torch.tensor([1, 3], dtype=torch.long),
    "token_positions": torch.tensor([2, 0], dtype=torch.long),
}
out = layer._dispatch_lora_with_coalesced_cpu_roundtrip(
    dispatch_fn=fake_dispatch,
    input_tensor=torch.tensor([[30.0, 40.0], [0.0, 0.0], [10.0, 20.0], [0.0, 0.0]]),
    layer_id=5,
    expert_id=7,
    pack_meta=pack_meta,
    reuse_packed_input=False,
    phase_name="gate",
    study2_prefix=None,
)
```

- [ ] **Step 5: Run the full background optimization test file**

Run:
```bash
pytest test/lora/test_colora_background_optimizations.py -v
```
Expected: all existing background tests still pass.

- [ ] **Step 6: Commit the roundtrip test**

```bash
git add test/lora/test_colora_background_optimizations.py
git commit -m "test: cover COLoRA coalesced cpu roundtrip"
```

### Task 4: Add a runtime test for speculative dispatch lifecycle

**Files:**
- Read / possibly modify: `lightllm/models/qwen3_vl_moe/lora_dispatch.py:1393-1772`
- Modify: `test/lora/test_colora_background_optimizations.py`

- [ ] **Step 1: Add a failing runtime test for submit -> bind -> retire on one speculative step**

```python
import torch


def test_speculative_gate_up_job_lifecycle_runtime():
    lora_mem_pool = _build_dummy_lora_mem_pool(dtype=torch.bfloat16)
    dispatcher = dispatch_mod.Qwen3VLMoELoRADispatcher(
        num_layers=1,
        gate_lora_rank=2,
        up_lora_rank=2,
        down_lora_rank=2,
        lora_compute_config=LoRAComputeConfig(moe_storage="cpu", moe_compute="hybrid"),
        colora_speculative_dispatch=True,
    )
    dispatcher.init_batched_mode(
        lora_mem_pool=lora_mem_pool,
        req_bins=torch.tensor([0, 0], dtype=torch.long),
        expert_cache_manager=MoEExpertCacheManager(MoEExpertCacheConfig(cache_budget_mb=1)),
    )

    dispatcher.begin_spec_step(11)
    key = dispatch_mod.SpecJobKey(
        layer_id=0,
        decode_step_id=11,
        op_kind="gate_up",
        adapter_bin=0,
        expert_id=0,
        row_group_sig=((0, 2),),
    )
    outcome = dispatcher.maybe_submit_fused_gate_up_spec_job(
        key=key,
        input_tensor=torch.ones((2, 4), dtype=torch.bfloat16),
        req_bins=torch.tensor([0, 0], dtype=torch.long),
        row_indices=torch.tensor([0, 1], dtype=torch.long),
    )
    assert outcome.status == "submitted"
```

- [ ] **Step 2: Run the single speculative lifecycle test**

Run:
```bash
pytest test/lora/test_colora_background_optimizations.py -k speculative_gate_up_job_lifecycle_runtime -v
```
Expected: FAIL first because the test still needs deterministic executor / cache-manager setup.

- [ ] **Step 3: Make the test deterministic by stubbing only the expensive internals**

```python
monkeypatch.setattr(dispatcher, "_is_joint_gate_up_ready", lambda **kwargs: False)
monkeypatch.setattr(dispatcher, "_get_moe_buffer_layer_id", lambda pool, layer_id, expert_id: layer_id)
monkeypatch.setattr(dispatcher, "_strict_moe_cpu_batch_lora", fake_strict_cpu_batch_lora)
```

Where `fake_strict_cpu_batch_lora` returns predictable gate/up tensors without changing production code.

- [ ] **Step 4: Extend the test to bind success and retire leftovers**

```python
bind = dispatcher.try_bind_gate_up_job_with_status(key)
assert bind.status == "bound"
assert isinstance(bind.result, tuple)

cleanup = dispatcher.retire_unbound_gate_up_jobs(layer_id=0, decode_step_id=11, keep_keys=set())
assert cleanup["retired_active"] == 0
```

- [ ] **Step 5: Add one not-ready or stale-path assertion in the same test file**

```python
dispatcher.begin_spec_step(12)
second_key = key._replace(decode_step_id=12, row_group_sig=((0, 1),))
outcome = dispatcher.try_bind_gate_up_job_with_status(second_key)
assert outcome.status in {"missing", "not_ready", "stale"}
```

- [ ] **Step 6: Run the full background optimization suite**

Run:
```bash
pytest test/lora/test_colora_background_optimizations.py -v
```
Expected: all tests pass with the new speculative coverage.

- [ ] **Step 7: Commit the speculative runtime test**

```bash
git add test/lora/test_colora_background_optimizations.py
git commit -m "test: cover COLoRA speculative dispatch lifecycle"
```

### Task 5: Add explicit measurement evidence for async overlap between GPU hot-path and CPU cold-path

**Files:**
- Read / possibly modify: `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py:443-502`
- Read / possibly modify: `lightllm/models/qwen3_vl_moe/lora_dispatch.py`
- Optionally create or extend under: `tools/case_study/`
- Update: `.superpowers/colora-runtime-audit-2026-03-31.md`

- [ ] **Step 1: Add a minimal measurement helper instead of instrumenting the production path broadly**

```python
start = torch.cuda.Event(enable_timing=True)
mid = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# launch representative GPU hot-path work
mid.record()
# run representative packed D2H -> CPU -> H2D path
end.record()
torch.cuda.synchronize()
print({
    "gpu_window_ms": start.elapsed_time(mid),
    "combined_window_ms": start.elapsed_time(end),
})
```

Place the helper in a case-study script or an isolated benchmark path, not in the steady-state runtime.

- [ ] **Step 2: Run the overlap helper once on the target host and save the raw output**

Run:
```bash
python <overlap_measurement_script>.py
```
Expected: one raw report showing GPU-only window, combined window, and an overlap-derived delta.

- [ ] **Step 3: If NVTX is easier to interpret on this machine, collect an NVTX trace instead**

Run:
```bash
nsys profile --trace=cuda,nvtx -o /tmp/colora_overlap_trace python <overlap_measurement_script>.py
```
Expected: trace shows the `MoE_COLoRA_D2H_Activation`, `MoE_COLoRA_CPU_AVX_Compute`, and `MoE_COLoRA_H2D_Activation` ranges from `transformer_layer_infer.py:443-502`.

- [ ] **Step 4: Convert the result into one audit note with a quantitative conclusion**

```markdown
- Verified by CUDA event timing / NVTX trace: the cold-path pack+D2H+CPU+H2D window overlaps with the hot-path GPU window by <measured amount>, so C3 moves from Unverified to Implemented/Measured Evidence.
```

- [ ] **Step 5: Commit the measurement helper or captured summary if it belongs in-repo**

```bash
git add .superpowers/colora-runtime-audit-2026-03-31.md tools/case_study
git commit -m "bench: capture COLoRA hot-cold overlap evidence"
```

### Task 6: Final audit consolidation and patch-list refresh

**Files:**
- Update: `.superpowers/colora-runtime-audit-2026-03-31.md`
- Read: generated calibration JSONs, replay CSVs, and any overlap evidence

- [ ] **Step 1: Update the audit statuses using the new evidence**

```markdown
- C1: move to Implemented if the roundtrip unit test passes.
- C3: move to Implemented or Measured Evidence if the overlap run succeeds.
- D3: move from Partial to Implemented if the speculative lifecycle test proves submit/bind/retire semantics.
- E2 / replay blocker: remove the cold_path_curves blocker once execution_first and no_deferred_sync complete.
```

- [ ] **Step 2: Refresh the patch list so only real remaining gaps stay open**

```markdown
## Patch list

### 1. Must fix before paper claims are credible
| Item | Category | Action |
|------|----------|--------|
| <leave empty if Task 1+2 close the replay blocker> | | |

### 2. Should fix for stronger end-to-end evidence
| Item | Category | Action |
|------|----------|--------|
| <only items still unresolved after Tasks 3-5> | | |
```

- [ ] **Step 3: Run the final focused validation gate**

Run:
```bash
pytest \
  test/lora/test_case_study_system_tpot_tools.py \
  test/lora/test_colora_background_optimizations.py -v
```
Expected: both files pass and cover the remaining evidence-critical semantics.

- [ ] **Step 4: Commit the final paper-support verdict**

```bash
git add \
  .superpowers/colora-runtime-audit-2026-03-31.md \
  .superpowers/plans/2026-04-01-colora-paper-support-next-steps.md
git commit -m "docs: finalize COLoRA paper support follow-up plan"
```

---

## Self-Review

### Spec coverage
- Highest-priority blocker (`cold_path_curves` missing from checked-in calibration) is covered by Task 1.
- Required replays (`execution_first`, `no_deferred_sync`) are covered by Task 2.
- Remaining should-fix evidence gaps (C1, C3, D3) are covered by Tasks 3–5.
- Final audit / patch-list refresh is covered by Task 6.

### Placeholder scan
- The only intentional placeholder is `<overlap_measurement_script>.py` because the measurement helper may be added as a new script or folded into an existing benchmark script after inspecting the preferred local workflow. All other commands use concrete repo paths.
- No TODO/TBD placeholders remain in the execution-critical calibration and replay path.

### Type consistency
- Calibration field names are consistent with `tools/case_study/calibrate_system_baseline.py:610-705` and `tools/case_study/system_tpot_core.py:214-287`: `cold_path_curves`, `profiles`, `early`, `late`, `pack_rows`, `d2h_rows`, `cpu_rows`, `h2d_rows`, `merge_rows`.
- Speculative dispatch method names match `lightllm/models/qwen3_vl_moe/lora_dispatch.py:1393-1772`.
- Cold-path roundtrip target matches `lightllm/models/qwen3_moe/layer_infer/transformer_layer_infer.py:409-508`.

### Expected outcome
After completing this plan, the repo should have:
1. a real checked-in calibration manifest that unblocks execution-first replay,
2. refreshed system-TPOT evidence for `execution_first` and `no_deferred_sync`,
3. direct unit-test coverage for the packed CPU roundtrip and speculative lifecycle, and
4. a smaller, sharper post-audit patch list focused only on any evidence still missing.
