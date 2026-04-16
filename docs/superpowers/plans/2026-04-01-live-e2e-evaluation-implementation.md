# Live E2E Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a live end-to-end evaluation runner for COLoRA that orchestrates paper-facing benchmark runs through the existing `benchmark_lora.sh` orchestrator, supports optional nsys profiling for diagnosis, and produces machine-readable summary artifacts for the paper comparison table.

**Architecture:** Follows the thin-wrapper design from the approved spec: reuse `test/lora/benchmark_lora.sh` as the canonical orchestrator for server lifecycle and traffic generation. Add four new components in `tools/evaluation/live_e2e/`: (1) a manifest schema for describing experiments, (2) a thin runner that invokes the shell script with appropriate arguments and organizes outputs into `artifacts/evaluation/live_e2e/<run_id>/`, (3) an optional nsys wrapper for diagnostic runs, and (4) a summary post-processor that parses the per-request JSONL logs and produces merged JSON/CSV summaries for the paper. All outputs follow the artifact layout defined in the design spec.

**Tech Stack:** Python 3.10+, JSON/YAML manifest format, `subprocess` for shell invocation, `pytest` for unit tests, `nsys` (NVIDIA Nsight Systems) for optional profiling, existing `tools/case_study/common.py` utilities.

---

## File Map

**New evaluation tooling:**
- Create: `tools/evaluation/live_e2e/__init__.py` — empty package init
- Create: `tools/evaluation/live_e2e/manifest.py` — manifest dataclasses and parsing
- Create: `tools/evaluation/live_e2e/runner.py` — main runner orchestration, output layout, nsys wrapping
- Create: `tools/evaluation/live_e2e/summarize.py` — post-processor that parses per-request logs and emits summaries
- Create: `tools/evaluation/live_e2e/__main__.py` — CLI entrypoint

**Tests:**
- Create: `test/lora/test_live_e2e_manifest_parsing.py` — unit tests for manifest loading and validation
- Create: `test/lora/test_live_e2e_summary_parsing.py` — unit tests for summary extraction from per-request logs

**Example manifest:**
- Create: `configs/live_e2e/paper_suite_example.yaml` — example manifest with the three canonical paper configurations

---

## Task 1: Create package structure and manifest schema

**Files:**
- Create: `tools/evaluation/live_e2e/__init__.py`
- Create: `tools/evaluation/live_e2e/manifest.py`
- Create: `test/lora/test_live_e2e_manifest_parsing.py`
- Create: `configs/live_e2e/paper_suite_example.yaml`

- [ ] **Step 1: Create package directories and empty init**

```bash
mkdir -p tools/evaluation/live_e2e
mkdir -p configs/live_e2e
touch tools/evaluation/live_e2e/__init__.py
```

- [ ] **Step 2: Write manifest dataclasses**

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class LiveE2ERun:
    """Single run definition for live E2E evaluation."""
    run_label: str
    suite_kind: str  # "paper" or "diagnostic"
    mode_label: str  # "baseline", "execution_first", "load_then_run", etc.
    compute_device: str
    miss_handling_mode: Optional[str] = None
    overlap_policy: Optional[str] = None
    async_fallback: Optional[bool] = None
    cpu_workers: Optional[int] = None
    cpu_queue_depth: Optional[int] = None
    cpu_batch_timeout_us: Optional[int] = None
    speculative_dispatch: Optional[bool] = None
    requests_path: str = "fixed_requests.jsonl"
    adapter_trace_path: str = "adapter_trace.jsonl"
    output_root: Path = Path("artifacts/evaluation/live_e2e")
    nsys_enabled: bool = False
    nsys_output_prefix: Optional[str] = None
    warmup_requests: int = 0
    measurement_requests: Optional[int] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for snapshotting."""
        return {
            "run_label": self.run_label,
            "suite_kind": self.suite_kind,
            "mode_label": self.mode_label,
            "compute_device": self.compute_device,
            "miss_handling_mode": self.miss_handling_mode,
            "overlap_policy": self.overlap_policy,
            "async_fallback": self.async_fallback,
            "cpu_workers": self.cpu_workers,
            "cpu_queue_depth": self.cpu_queue_depth,
            "cpu_batch_timeout_us": self.cpu_batch_timeout_us,
            "speculative_dispatch": self.speculative_dispatch,
            "requests_path": self.requests_path,
            "adapter_trace_path": self.adapter_trace_path,
            "output_root": str(self.output_root),
            "nsys_enabled": self.nsys_enabled,
            "nsys_output_prefix": self.nsys_output_prefix,
            "warmup_requests": self.warmup_requests,
            "measurement_requests": self.measurement_requests,
        }


@dataclass
class LiveE2EManifest:
    """Top-level manifest for a collection of live E2E runs."""
    run_id: str
    runs: List[LiveE2ERun] = field(default_factory=list)
    description: Optional[str] = None
    benchmark_script: str = "test/lora/benchmark_lora.sh"


def load_manifest(manifest_path: Path) -> LiveE2EManifest:
    """Load and validate a manifest from YAML."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    runs = []
    for run_data in data.get("runs", []):
        if "output_root" in run_data:
            run_data["output_root"] = Path(run_data["output_root"])
        try:
            run = LiveE2ERun(**run_data)
        except TypeError as e:
            raise ValueError(f"Invalid run definition in manifest: {e}") from e

        if run.suite_kind not in ("paper", "diagnostic"):
            raise ValueError(f"Invalid suite_kind: {run.suite_kind}, must be 'paper' or 'diagnostic'")

        runs.append(run)

    return LiveE2EManifest(
        run_id=data["run_id"],
        runs=runs,
        description=data.get("description"),
        benchmark_script=data.get("benchmark_script", "test/lora/benchmark_lora.sh"),
    )
```

- [ ] **Step 3: Write failing test for manifest parsing**

```python
import pytest
from pathlib import Path
import tempfile
from tools.evaluation.live_e2e.manifest import load_manifest, LiveE2EManifest, LiveE2ERun


def _write_temp_manifest(tmp_path: Path, content: str) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(content, encoding="utf-8")
    return manifest_path


def test_load_valid_paper_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = """
run_id: test_paper_suite
description: "Test paper suite"
runs:
  - run_label: baseline_lora_gpu
    suite_kind: paper
    mode_label: baseline
    compute_device: "gpu:gpu,gpu:gpu"
    warmup_requests: 2
  - run_label: colora_execution_first
    suite_kind: paper
    mode_label: execution_first
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: execution_first
    overlap_policy: calibrated
    cpu_workers: 2
    cpu_queue_depth: 8
    nsys_enabled: false
"""
        manifest_path = _write_temp_manifest(tmp_path, content)
        manifest = load_manifest(manifest_path)
        assert isinstance(manifest, LiveE2EManifest)
        assert manifest.run_id == "test_paper_suite"
        assert len(manifest.runs) == 2
        assert manifest.runs[0].run_label == "baseline_lora_gpu"
        assert manifest.runs[0].suite_kind == "paper"
        assert manifest.runs[0].nsys_enabled is False
        assert manifest.runs[1].miss_handling_mode == "execution_first"
        assert manifest.runs[1].cpu_workers == 2


def test_reject_invalid_suite_kind():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = """
run_id: bad_test
runs:
  - run_label: test
    suite_kind: invalid
    mode_label: test
    compute_device: "gpu:gpu"
"""
        manifest_path = _write_temp_manifest(tmp_path, content)
        with pytest.raises(ValueError, match="Invalid suite_kind"):
            load_manifest(manifest_path)


def test_canonical_paper_suite_has_three_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = """
run_id: paper_canonical_1
description: "Canonical three-run comparison for paper"
runs:
  - run_label: baseline_lora_gpu
    suite_kind: paper
    mode_label: baseline
    compute_device: "gpu:gpu,gpu:gpu"
  - run_label: colora_execution_first
    suite_kind: paper
    mode_label: execution_first
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: execution_first
    overlap_policy: calibrated
    nsys_enabled: false
  - run_label: colora_load_then_run
    suite_kind: paper
    mode_label: load_then_run
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: load_then_run
    overlap_policy: calibrated
    nsys_enabled: false
"""
        manifest_path = _write_temp_manifest(tmp_path, content)
        manifest = load_manifest(manifest_path)
        assert len(manifest.runs) == 3
        expected_labels = {"baseline_lora_gpu", "colora_execution_first", "colora_load_then_run"}
        assert {r.run_label for r in manifest.runs} == expected_labels
```

- [ ] **Step 4: Run test to confirm it fails (module not found)**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.evaluation.live_e2e'`.

- [ ] **Step 5: Run test again after files created and confirm it passes**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py -v
```
Expected: All three tests PASS.

- [ ] **Step 6: Write the example paper suite manifest**

```yaml
# Example canonical paper suite manifest for live E2E evaluation
# Three configurations: baseline + two COLoRA policies

run_id: canonical_paper_suite_01
description: >
  Canonical three-way comparison for the paper:
  - Baseline: LoRA all-on-GPU
  - COLoRA: execution_first with calibrated overlap
  - COLoRA: load_then_run with calibrated overlap

benchmark_script: test/lora/benchmark_lora.sh

runs:
  - run_label: baseline_lora_gpu
    suite_kind: paper
    mode_label: baseline
    compute_device: "all:gpu"
    warmup_requests: 4
    measurement_requests: 32
    nsys_enabled: false
    output_root: artifacts/evaluation/live_e2e

  - run_label: colora_execution_first
    suite_kind: paper
    mode_label: execution_first
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: execution_first
    overlap_policy: calibrated
    async_fallback: true
    cpu_workers: 2
    cpu_queue_depth: 8
    cpu_batch_timeout_us: 500
    speculative_dispatch: true
    warmup_requests: 4
    measurement_requests: 32
    nsys_enabled: false
    output_root: artifacts/evaluation/live_e2e

  - run_label: colora_load_then_run
    suite_kind: paper
    mode_label: load_then_run
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: load_then_run
    overlap_policy: calibrated
    async_fallback: true
    cpu_workers: 2
    cpu_queue_depth: 8
    cpu_batch_timeout_us: 500
    speculative_dispatch: false
    warmup_requests: 4
    measurement_requests: 32
    nsys_enabled: false
    output_root: artifacts/evaluation/live_e2e
```

- [ ] **Step 7: Commit the package, manifest module, tests, and example**

```bash
git add \
  tools/evaluation/live_e2e/__init__.py \
  tools/evaluation/live_e2e/manifest.py \
  test/lora/test_live_e2e_manifest_parsing.py \
  configs/live_e2e/paper_suite_example.yaml
git commit -m "feat: live-e2e manifest schema and parsing"
```

---

## Task 2: Implement the thin runner with output layout and nsys wrapping

**Files:**
- Create: `tools/evaluation/live_e2e/runner.py`
- Modify: `test/lora/test_live_e2e_manifest_parsing.py` (add command construction test)

- [ ] **Step 1: Add failing test for benchmark command construction**

Add this to `test/lora/test_live_e2e_manifest_parsing.py`:

```python
def test_build_benchmark_command_baseline():
    from tools.evaluation.live_e2e.runner import build_benchmark_command
    from tools.evaluation.live_e2e.manifest import LiveE2ERun

    run = LiveE2ERun(
        run_label="baseline_lora_gpu",
        suite_kind="paper",
        mode_label="baseline",
        compute_device="all:gpu",
    )

    cmd = build_benchmark_command(run, benchmark_script="/path/benchmark_lora.sh")
    assert "/path/benchmark_lora.sh" in cmd
    assert "--compute-device all:gpu" in cmd
    assert "--per-request-log" in cmd


def test_build_benchmark_command_colora():
    from tools.evaluation.live_e2e.runner import build_benchmark_command
    from tools.evaluation.live_e2e.manifest import LiveE2ERun

    run = LiveE2ERun(
        run_label="colora_execution_first",
        suite_kind="paper",
        mode_label="execution_first",
        compute_device="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid",
        miss_handling_mode="execution_first",
        overlap_policy="calibrated",
        cpu_workers=2,
        cpu_queue_depth=8,
        async_fallback=True,
    )

    cmd = build_benchmark_command(run, benchmark_script="/path/benchmark_lora.sh")
    assert "--colora-miss-policy execution_first" in cmd
    assert "--colora-overlap-policy calibrated" in cmd
    assert "--colora-cpu-workers 2" in cmd
    assert "--colora-cpu-queue-depth 8" in cmd
    assert "--colora-async-fallback true" in cmd
```

- [ ] **Step 2: Run tests to confirm new tests fail**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py::test_build_benchmark_command_baseline -v
```
Expected: FAIL with `No module named 'tools.evaluation.live_e2e.runner'`.

- [ ] **Step 3: Implement the runner module**

```python
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import json

from tools.case_study.common import ensure_dir, write_json
from tools.evaluation.live_e2e.manifest import LiveE2EManifest, LiveE2ERun


def build_benchmark_command(run: LiveE2ERun, benchmark_script: str) -> str:
    """Build the benchmark_lora.sh command string from a run definition."""
    parts = [benchmark_script]
    parts.append(f"--compute-device {run.compute_device}")

    if run.miss_handling_mode is not None:
        parts.append(f"--colora-miss-policy {run.miss_handling_mode}")
    if run.overlap_policy is not None:
        parts.append(f"--colora-overlap-policy {run.overlap_policy}")
    if run.async_fallback is not None:
        parts.append(f"--colora-async-fallback {str(run.async_fallback).lower()}")
    if run.cpu_workers is not None:
        parts.append(f"--colora-cpu-workers {run.cpu_workers}")
    if run.cpu_queue_depth is not None:
        parts.append(f"--colora-cpu-queue-depth {run.cpu_queue_depth}")
    if run.cpu_batch_timeout_us is not None:
        parts.append(f"--colora-cpu-batch-timeout-us {run.cpu_batch_timeout_us}")
    if run.speculative_dispatch is not None:
        parts.append(f"--colora-speculative-dispatch {str(run.speculative_dispatch).lower()}")
    if run.warmup_requests is not None:
        parts.append(f"--warmup-requests {run.warmup_requests}")
    if run.measurement_requests is not None:
        parts.append(f"--measurement-requests {run.measurement_requests}")

    # The per-request log path will be determined at runtime in the run directory
    return " ".join(parts)


def build_nsys_command(base_cmd: str, output_path: Path, run: LiveE2ERun) -> str:
    """Wrap the benchmark command with nsys profile."""
    prefix = run.nsys_output_prefix or f"{run.run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    nsys_cmd = (
        f"nsys profile --stats=true --output {output_path / prefix} {base_cmd}"
    )
    return nsys_cmd


def get_run_output_dir(manifest: LiveE2EManifest, run: LiveE2ERun) -> Path:
    """Get the output directory for a run, following the artifact layout."""
    if run.suite_kind == "paper":
        return run.output_root / manifest.run_id / "paper_runs" / run.run_label
    elif run.suite_kind == "diagnostic":
        return run.output_root / manifest.run_id / "diagnostic_runs" / run.run_label
    else:
        raise ValueError(f"Unknown suite_kind: {run.suite_kind}")


def get_summaries_dir(manifest: LiveE2EManifest) -> Path:
    """Get the summaries directory for an evaluation."""
    return manifest.runs[0].output_root / manifest.run_id / "summaries"


def run_single(manifest: LiveE2EManifest, run: LiveE2ERun, capture_stdout: bool = True) -> dict:
    """Execute a single run, capture outputs, return run result metadata."""
    output_dir = get_run_output_dir(manifest, run)
    ensure_dir(output_dir)

    # Write config snapshot before running
    config_snapshot_path = output_dir / "config_snapshot.json"
    write_json(config_snapshot_path, run.to_metadata())

    # Build the command
    base_cmd = build_benchmark_command(run, manifest.benchmark_script)
    # Append the per-request log path to our output directory
    per_request_log_path = output_dir / "per_request_metrics.jsonl"
    base_cmd += f" --per-request-log {per_request_log_path}"

    if run.nsys_enabled:
        cmd = build_nsys_command(base_cmd, output_dir, run)
    else:
        cmd = base_cmd

    print(f"\n=== Running {run.run_label} ===")
    print(f"Command: {cmd}")
    print(f"Output directory: {output_dir}")

    # Execute and capture output
    stdout_path = output_dir / "benchmark_stdout.log"
    stderr_path = output_dir / "benchmark_stderr.log"

    start_time = datetime.now()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_stdout,
            text=True,
        )
    except Exception as e:
        end_time = datetime.now()
        result = {
            "run_label": run.run_label,
            "suite_kind": run.suite_kind,
            "valid": False,
            "failure_stage": "execution",
            "error": str(e),
            "start_time": start_time.isoformat(),
            "end_time": (end_time - start_time).total_seconds(),
            "command": cmd,
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "run_result.json", result)
        return result

    end_time = datetime.now()

    if capture_stdout:
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

    # Validate result
    valid = True
    failure_reason = None

    if proc.returncode != 0:
        valid = False
        failure_reason = f"benchmark exited with code {proc.returncode}"

    if not per_request_log_path.exists():
        valid = False
        failure_reason = "per-request log file not created"

    if run.nsys_enabled:
        # Check that at least one nsys file was created
        nsys_files = list(output_dir.glob("*.qdrep")) + list(output_dir.glob("*.nsys-rep"))
        if not nsys_files:
            valid = False
            failure_reason = "nsys enabled but no profiling artifact created"

    result = {
        "run_label": run.run_label,
        "suite_kind": run.suite_kind,
        "mode_label": run.mode_label,
        "valid": valid,
        "failure_reason": failure_reason,
        "exit_code": proc.returncode,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "command": cmd,
        "output_dir": str(output_dir),
        "per_request_log_path": str(per_request_log_path),
        "start_time": start_time.isoformat(),
        "nsys_enabled": run.nsys_enabled,
    }

    write_json(output_dir / "run_result.json", result)
    print(f"=== Completed {run.run_label}: valid={valid}")
    if failure_reason:
        print(f"Failure: {failure_reason}")

    return result


def run_manifest(manifest_path: Path) -> List[dict]:
    """Run all runs defined in a manifest."""
    manifest = load_manifest(manifest_path)
    print(f"Loaded manifest run_id={manifest.run_id}, {len(manifest.runs)} runs")

    results = []
    for run in manifest.runs:
        result = run_single(manifest, run)
        results.append(result)

    # Write overall results summary
    summaries_dir = get_summaries_dir(manifest)
    ensure_dir(summaries_dir)
    write_json(summaries_dir / "manifest_results.json", {
        "manifest_path": str(manifest_path),
        "run_id": manifest.run_id,
        "description": manifest.description,
        "total_runs": len(results),
        "valid_runs": sum(1 for r in results if r["valid"]),
        "results": results,
    })

    print(f"\n=== All runs complete ===")
    print(f"Total: {len(results)}, Valid: {sum(1 for r in results if r['valid'])}")
    print(f"Results in: {summaries_dir}")

    return results
```

- [ ] **Step 4: Import `load_manifest` at top of `runner.py`**

Add at the top:
```python
from tools.evaluation.live_e2e.manifest import load_manifest
```

- [ ] **Step 5: Run tests and confirm they pass**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py -v
```
Expected: All five tests PASS.

- [ ] **Step 6: Commit the runner implementation**

```bash
git add tools/evaluation/live_e2e/runner.py test/lora/test_live_e2e_manifest_parsing.py
git commit -m "feat: live-e2e runner with output layout and nsys wrapping"
```

---

## Task 3: Implement the summary post-processor

**Files:**
- Create: `tools/evaluation/live_e2e/summarize.py`
- Create: `test/lora/test_live_e2e_summary_parsing.py`

- [ ] **Step 1: Write failing summary parsing tests**

```python
from pathlib import Path
import tempfile
import json
import pytest
from tools.evaluation.live_e2e.summarize import summarize_run, percentile


def test_percentile_calculation():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(data, 50) == 5.5
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 10.0


def test_summarize_valid_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        per_request_path = tmp_path / "per_request_metrics.jsonl"

        lines = []
        for i in range(10):
            # All successful, latency from 10ms to 100ms (0.01s to 0.1s)
            line = {
                "index": i,
                "adapter_id": f"lora_{i % 3}",
                "status": "ok",
                "latency_s": 0.01 * (i + 1),
                "completion_tokens": 32,
            }
            lines.append(json.dumps(line))

        per_request_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary = summarize_run(
            per_request_path=per_request_path,
            run_label="test_run",
            suite_kind="paper",
            mode_label="baseline",
        )

        assert summary["run_label"] == "test_run"
        assert summary["valid"] is True
        assert summary["request_count"] == 10
        assert summary["success_count"] == 10
        assert summary["success_rate"] == 1.0
        # p50 latency should be ~0.055s = 55ms
        assert abs(summary["latency_p50_ms"] - 55.0) < 1.0
        # p95 should be ~95.5ms
        assert abs(summary["latency_p95_ms"] - 95.5) < 2.0
        assert summary["total_completion_tokens"] == 10 * 32
        assert summary["total_latency_seconds"] == sum(0.01 * (i + 1) for i in range(10))


def test_summarize_partial_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        per_request_path = tmp_path / "per_request_metrics.jsonl"

        lines = []
        for i in range(10):
            status = "ok" if i < 8 else "error"
            line = {
                "index": i,
                "adapter_id": f"lora_{i}",
                "status": status,
                "latency_s": 0.01 * (i + 1),
                "completion_tokens": 32,
            }
            lines.append(json.dumps(line))

        per_request_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary = summarize_run(
            per_request_path=per_request_path,
            run_label="test_run",
            suite_kind="paper",
            mode_label="baseline",
        )

        assert summary["request_count"] == 10
        assert summary["success_count"] == 8
        assert abs(summary["success_rate"] - 0.8) < 0.001
        assert summary["valid"] is True  # Still valid if some succeeded
```

- [ ] **Step 2: Run tests to confirm it fails**

Run:
```bash
pytest test/lora/test_live_e2e_summary_parsing.py -v
```
Expected: FAIL with `No module named 'tools.evaluation.live_e2e.summarize'`.

- [ ] **Step 3: Implement the summarizer**

```python
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from statistics import mean

from tools.case_study.common import (
    ensure_parent_dir,
    write_json,
    percentile,
)


def load_per_request_metrics(per_request_path: Path) -> List[Dict[str, Any]]:
    """Load per-request metrics from JSONL log."""
    metrics = []
    with per_request_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics


def summarize_run(
    per_request_path: Path,
    run_label: str,
    suite_kind: str,
    mode_label: str,
    min_valid_requests: int = 1,
) -> Dict[str, Any]:
    """Summarize a single run from its per-request log."""
    if not per_request_path.exists():
        return {
            "run_label": run_label,
            "suite_kind": suite_kind,
            "mode_label": mode_label,
            "valid": False,
            "reason": f"per-request log not found at {per_request_path}",
            "request_count": 0,
            "success_count": 0,
        }

    metrics = load_per_request_metrics(per_request_path)
    request_count = len(metrics)
    successes = [m for m in metrics if m.get("status") == "ok"]
    success_count = len(successes)

    if request_count < min_valid_requests or success_count < min_valid_requests:
        return {
            "run_label": run_label,
            "suite_kind": suite_kind,
            "mode_label": mode_label,
            "valid": False,
            "reason": f"only {success_count} valid successes, need at least {min_valid_requests}",
            "request_count": request_count,
            "success_count": success_count,
        }

    # Compute latency statistics in milliseconds
    latencies_ms = [m["latency_s"] * 1000.0 for m in successes]
    latencies_ms.sort()

    summary = {
        "run_label": run_label,
        "suite_kind": suite_kind,
        "mode_label": mode_label,
        "valid": True,
        "request_count": request_count,
        "success_count": success_count,
        "success_rate": success_count / request_count if request_count > 0 else 0.0,
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_p99_ms": percentile(latencies_ms, 99),
        "latency_mean_ms": mean(latencies_ms) if latencies_ms else 0.0,
        "latency_min_ms": min(latencies_ms) if latencies_ms else 0.0,
        "latency_max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }

    # Compute throughput from total tokens and total time
    total_completion_tokens = sum(m.get("completion_tokens", 0) for m in successes)
    if "start_offset_s" in metrics[0] and "finish_offset_s" in metrics[-1]:
        # Throughput over the whole measurement window
        start_offset = min(m["start_offset_s"] for m in successes if "start_offset_s" in m)
        finish_offset = max(m["finish_offset_s"] for m in successes if "finish_offset_s" in m)
        total_window_seconds = finish_offset - start_offset
        if total_window_seconds > 0:
            throughput_tokens_per_second = total_completion_tokens / total_window_seconds
        else:
            throughput_tokens_per_second = 0.0
    else:
        # Fallback: sum of individual latencies
        total_latency_seconds = sum(m["latency_s"] for m in successes)
        throughput_tokens_per_second = total_completion_tokens / total_latency_seconds if total_latency_seconds > 0 else 0.0

    summary["total_completion_tokens"] = total_completion_tokens
    summary["throughput_tokens_per_second"] = throughput_tokens_per_second

    return summary


def collect_and_summarize_manifest(
    manifest_run_id: str,
    results_json_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    """Collect results from all runs in a manifest and produce merged summary."""
    with results_json_path.open("r", encoding="utf-8") as f:
        manifest_results = json.load(f)

    all_summaries = []
    for result in manifest_results["results"]:
        if not result["valid"]:
            # Invalid run gets a summary marked invalid
            summary = {
                "run_label": result["run_label"],
                "suite_kind": result["suite_kind"],
                "mode_label": result["mode_label"],
                "valid": False,
                "reason": result.get("failure_reason", "benchmark run failed"),
                "request_count": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_p99_ms": None,
                "throughput_tokens_per_second": None,
            }
            all_summaries.append(summary)
            continue

        per_request_path = Path(result["per_request_log_path"])
        summary = summarize_run(
            per_request_path=per_request_path,
            run_label=result["run_label"],
            suite_kind=result["suite_kind"],
            mode_label=result["mode_label"],
        )
        all_summaries.append(summary)

    # Write top-level summary
    output_summary = {
        "manifest_run_id": manifest_run_id,
        "description": manifest_results.get("description"),
        "total_runs": len(all_summaries),
        "valid_runs": sum(1 for s in all_summaries if s["valid"]),
        "summaries": all_summaries,
    }

    summary_json_path = output_root / "live_e2e_summary.json"
    write_json(summary_json_path, output_summary)

    # Write comparison CSV for paper (only valid paper runs)
    csv_path = output_root / "live_e2e_comparison.csv"
    paper_summaries = [s for s in all_summaries if s["valid"] and s["suite_kind"] == "paper"]

    if paper_summaries:
        fieldnames = [
            "run_label",
            "mode_label",
            "request_count",
            "success_count",
            "success_rate",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "latency_mean_ms",
            "throughput_tokens_per_second",
        ]

        ensure_parent_dir(csv_path)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in paper_summaries:
                row = {k: s.get(k) for k in fieldnames}
                writer.writerow(row)

    return output_summary
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:
```bash
pytest test/lora/test_live_e2e_summary_parsing.py -v
```
Expected: All three tests PASS.

- [ ] **Step 5: Commit the summarizer**

```bash
git add tools/evaluation/live_e2e/summarize.py test/lora/test_live_e2e_summary_parsing.py
git commit -m "feat: live-e2e summary post-processor"
```

---

## Task 4: Add CLI entrypoint

**Files:**
- Create: `tools/evaluation/live_e2e/__main__.py`

- [ ] **Step 1: Implement CLI entrypoint**

```python
import argparse
from pathlib import Path
import sys

from tools.evaluation.live_e2e.runner import run_manifest
from tools.evaluation.live_e2e.summarize import collect_and_summarize_manifest, get_summaries_dir
from tools.evaluation.live_e2e.manifest import load_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Live E2E evaluation runner for COLoRA paper evidence."
    )
    parser.add_argument(
        "--manifest", "-m",
        required=True,
        type=Path,
        help="Path to YAML manifest file describing the evaluation runs",
    )
    parser.add_argument(
        "--summarize-only", "-s",
        action="store_true",
        help="Only run summarization from existing results, do not re-execute runs",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.summarize_only:
        # Load manifest to get run_id and output location
        manifest = load_manifest(args.manifest)
        summaries_dir = get_summaries_dir(manifest)
        results_json_path = summaries_dir / "manifest_results.json"
        if not results_json_path.exists():
            print(f"Error: manifest_results.json not found at {results_json_path}. Run the full evaluation first.", file=sys.stderr)
            sys.exit(1)
        summary = collect_and_summarize_manifest(
            manifest_run_id=manifest.run_id,
            results_json_path=results_json_path,
            output_root=summaries_dir,
        )
        print(f"\nSummary complete:")
        print(f"  Total runs: {summary['total_runs']}")
        print(f"  Valid runs: {summary['valid_runs']}")
        print(f"  Output: {summaries_dir / 'live_e2e_summary.json'}")
        print(f"  Comparison CSV: {summaries_dir / 'live_e2e_comparison.csv'}")
        sys.exit(0)
    else:
        # Run all the evaluation
        results = run_manifest(args.manifest)
        manifest = load_manifest(args.manifest)
        summaries_dir = get_summaries_dir(manifest)
        # After running, automatically do the summarization
        results_json_path = summaries_dir / "manifest_results.json"
        summary = collect_and_summarize_manifest(
            manifest_run_id=manifest.run_id,
            results_json_path=results_json_path,
            output_root=summaries_dir,
        )
        print(f"\nEvaluation and summarization complete:")
        print(f"  Total runs executed: {len(results)}")
        print(f"  Valid runs executed: {sum(1 for r in results if r['valid'])}")
        print(f"  Summary: {summaries_dir / 'live_e2e_summary.json'}")
        print(f"  Paper comparison CSV: {summaries_dir / 'live_e2e_comparison.csv'}")
        # Exit with non-zero if no valid paper runs
        valid_paper = sum(
            1 for s in summary["summaries"]
            if s.get("valid") and s.get("suite_kind") == "paper"
        )
        expected_paper = sum(
            1 for r in manifest.runs
            if r.suite_kind == "paper"
        )
        if valid_paper < expected_paper:
            print(f"\nWarning: Only {valid_paper}/{expected_paper} paper runs completed successfully.", file=sys.stderr)
            if valid_paper == 0:
                sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the CLI prints help**

Run:
```bash
python -m tools.evaluation.live_e2e --help
```
Expected: Prints help message with `--manifest` and `--summarize-only` options.

- [ ] **Step 3: Commit the CLI entrypoint**

```bash
git add tools/evaluation/live_e2e/__main__.py
git commit -m "feat: live-e2e CLI entrypoint"
```

---

## Task 5: Add smoke/integration tests for dry-run command generation

**Files:**
- Modify: `test/lora/test_live_e2e_manifest_parsing.py`
- Add: dry-run test that constructs commands for the canonical paper suite

- [ ] **Step 1: Add a dry-run test for the example paper suite manifest**

Add this to `test/lora/test_live_e2e_manifest_parsing.py`:

```python
def test_example_paper_suite_manifest_parses_and_builds_commands():
    from tools.evaluation.live_e2e.manifest import load_manifest
    from tools.evaluation.live_e2e.runner import build_benchmark_command, get_run_output_dir

    manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/paper_suite_example.yaml"
    assert manifest_path.exists()

    manifest = load_manifest(manifest_path)
    assert manifest.run_id == "canonical_paper_suite_01"
    assert len(manifest.runs) == 3

    # Check output paths follow the spec layout
    for run in manifest.runs:
        output_dir = get_run_output_dir(manifest, run)
        assert str(output_dir).endswith(f"paper_runs/{run.run_label}")
        assert "artifacts/evaluation/live_e2e" in str(output_dir)

    # Build all three commands and check expected flags are present
    cmds = []
    for run in manifest.runs:
        cmd = build_benchmark_command(run, "test/lora/benchmark_lora.sh")
        cmds.append(cmd)

    # Baseline command has all-gpu
    assert "all:gpu" in cmds[0]

    # execution_first has correct policy
    assert "execution_first" in cmds[1]
    assert "calibrated" in cmds[1]
    assert "cpu-workers 2" in cmds[1]
    assert "speculative-dispatch true" in cmds[1]

    # load_then_run has correct policy
    assert "load_then_run" in cmds[2]
    assert "calibrated" in cmds[2]
    assert "speculative-dispatch false" in cmds[2]
```

- [ ] **Step 2: Run all manifest tests and confirm they pass**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py -v
```
Expected: All six tests PASS.

- [ ] **Step 3: Add diagnostic nsys command construction test**

Add this to `test/lora/test_live_e2e_manifest_parsing.py`:

```python
def test_nsys_wrapper_builds_correct_command():
    from tools.evaluation.live_e2e.manifest import LiveE2ERun
    from tools.evaluation.live_e2e.runner import build_benchmark_command, build_nsys_command

    run = LiveE2ERun(
        run_label="diagnostic_execution_first",
        suite_kind="diagnostic",
        mode_label="execution_first",
        compute_device="vl_storage:gpu,vl_compute:gpu",
        miss_handling_mode="execution_first",
        nsys_enabled=True,
        nsys_output_prefix="debug_execution_first",
    )

    base_cmd = build_benchmark_command(run, "test/lora/benchmark_lora.sh")
    nsys_cmd = build_nsys_command(base_cmd, Path("/tmp/output"), run)

    assert "nsys profile" in nsys_cmd
    assert "--output /tmp/output/debug_execution_first" in nsys_cmd
    assert "execution_first" in nsys_cmd
```

- [ ] **Step 4: Run all tests and confirm they pass**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py -v
pytest test/lora/test_live_e2e_summary_parsing.py -v
```
Expected: All seven manifest tests and three summary tests PASS (total 10/10).

- [ ] **Step 5: Commit the added tests**

```bash
git add test/lora/test_live_e2e_manifest_parsing.py
git commit -m "test: add dry-run smoke tests for canonical paper suite"
```

---

## Task 6: Manual validation documentation and smoke test instructions

**Files:**
- Create: `tools/evaluation/live_e2e/README.md`

- [ ] **Step 1: Write README with smoke test instructions**

```markdown
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
```

- [ ] **Step 2: Commit the README**

```bash
git add tools/evaluation/live_e2e/README.md
git commit -m "docs: add live-e2e README with smoke test instructions"
```

---

## Task 7: Final test run and verification

- [ ] **Step 1: Run all new tests**

Run:
```bash
pytest test/lora/test_live_e2e_manifest_parsing.py -v
pytest test/lora/test_live_e2e_summary_parsing.py -v
```
Expected: All tests PASS.

- [ ] **Step 2: Verify the example manifest parses correctly with dry run**

Run:
```bash
python -c "
from pathlib import Path
from tools.evaluation.live_e2e.manifest import load_manifest
from tools.evaluation.live_e2e.runner import build_benchmark_command
manifest = load_manifest(Path('configs/live_e2e/paper_suite_example.yaml'))
print(f'Loaded {len(manifest.runs)} runs for run_id={manifest.run_id}')
for i, run in enumerate(manifest.runs):
    cmd = build_benchmark_command(run, manifest.benchmark_script)
    print(f'  [{i+1}] {run.run_label} -> {len(cmd)} chars command')
"
```
Expected: Output shows 3 runs, each command built successfully.

- [ ] **Step 3: Verify CLI --help works**

Run:
```bash
python -m tools.evaluation.live_e2e --help
```
Expected: Help prints with options.

---

## Verification Check List

- [ ] All manifest schema tests pass
- [ ] All summary parsing tests pass
- [ ] Example paper manifest parses and builds all three commands correctly
- [ ] Output paths follow the required layout under `artifacts/evaluation/live_e2e/<run_id>/`
- [ ] nsys wrapping produces correct command for diagnostic runs
- [ ] Summary parser correctly handles mixed success/error requests
- [ ] Summaries produce the required latency percentiles and throughput
- [ ] Merged CSV contains only valid paper runs with the expected columns

