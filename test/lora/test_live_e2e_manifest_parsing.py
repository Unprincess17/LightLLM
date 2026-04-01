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
