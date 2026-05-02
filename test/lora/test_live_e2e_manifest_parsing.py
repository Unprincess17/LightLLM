import json
import pytest
import subprocess
import sys
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
    overlap_policy: request_skip
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
    overlap_policy: request_skip
    nsys_enabled: false
  - run_label: colora_load_then_run
    suite_kind: paper
    mode_label: load_then_run
    compute_device: "vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"
    miss_handling_mode: load_then_run
    overlap_policy: request_skip
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
    assert "--compute_device all:gpu" in cmd


def test_build_benchmark_command_colora():
    from tools.evaluation.live_e2e.runner import build_benchmark_command
    from tools.evaluation.live_e2e.manifest import LiveE2ERun

    run = LiveE2ERun(
        run_label="colora_execution_first",
        suite_kind="paper",
        mode_label="execution_first",
        compute_device="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid",
        miss_handling_mode="execution_first",
        overlap_policy="request_skip",
        overlap_mode="full",
        adapter_ids="lora_dummy_0,lora_dummy_1",
        lora_dirs="/tmp/lora_dummy_0,/tmp/lora_dummy_1",
        cpu_workers=2,
        cpu_queue_depth=8,
        async_fallback=True,
    )

    cmd = build_benchmark_command(run, benchmark_script="/path/benchmark_lora.sh")
    assert "--colora_miss_policy execution_first" in cmd
    assert "--colora_overlap_mode full" in cmd
    assert "--adapter_ids lora_dummy_0,lora_dummy_1" in cmd
    assert "--lora_dirs /tmp/lora_dummy_0,/tmp/lora_dummy_1" in cmd
    assert "--colora_request_skip 1" in cmd
    assert "--colora_cpu_workers 2" in cmd
    assert "--colora_cpu_queue_depth 8" in cmd
    assert "--colora_async_fallback 1" in cmd


def test_manifest_adapter_trace_backward_compat_alias():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = """
run_id: trace_alias_test
runs:
  - run_label: trace_replay
    suite_kind: paper
    mode_label: baseline
    compute_device: "all:gpu"
    adapter_trace_path: "/tmp/legacy_adapter_trace.jsonl"
"""
        manifest_path = _write_temp_manifest(tmp_path, content)
        manifest = load_manifest(manifest_path)
        run = manifest.runs[0]
        assert run.adapter_trace_path == "/tmp/legacy_adapter_trace.jsonl"
        assert run.measurement_adapter_trace_path == "/tmp/legacy_adapter_trace.jsonl"


def test_build_benchmark_command_includes_trace_and_server_endpoint_flags():
    from tools.evaluation.live_e2e.runner import build_benchmark_command
    from tools.evaluation.live_e2e.manifest import LiveE2ERun

    run = LiveE2ERun(
        run_label="trace_replay_non_default_port",
        suite_kind="diagnostic",
        mode_label="baseline",
        compute_device="all:gpu",
        warmup_adapter_trace_path="/tmp/warmup_trace.jsonl",
        measurement_adapter_trace_path="/tmp/measure_trace.jsonl",
        server_host="127.0.0.1",
        server_port=18040,
    )

    cmd = build_benchmark_command(run, benchmark_script="/path/benchmark_lora.sh")
    assert "--warmup_adapter_trace_path /tmp/warmup_trace.jsonl" in cmd
    assert "--measure_adapter_trace_path /tmp/measure_trace.jsonl" in cmd
    assert "--server_host 127.0.0.1" in cmd
    assert "--server_port 18040" in cmd


def test_build_benchmark_command_extended_colora_pass_through():
    from tools.evaluation.live_e2e.runner import build_benchmark_command
    from tools.evaluation.live_e2e.manifest import LiveE2ERun

    run = LiveE2ERun(
        run_label="colora_extended_pass_through",
        suite_kind="paper",
        mode_label="execution_first_extended",
        compute_device="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid",
        miss_handling_mode="cpu_first",
        overlap_policy="request_skip",
        overlap_mode="full",
        async_fallback=True,
        cpu_workers=2,
        cpu_queue_depth=8,
        cpu_batch_timeout_us=500,
        max_continuations=12,
        cache_budget_mb=1024,
        promote_min_hits=3,
        promote_window=64,
        max_promote_per_step=4,
        decay=0.8,
        deferred_promotion_delta_steps=5,
        promotion_ema_alpha=0.6,
        temporal_prefetch=True,
        temporal_prefetch_layer_whitelist="0,1,2",
        temporal_hot_cache_slots=32,
        speculative_dispatch=True,
        spec_layer_whitelist="0,1,2",
    )
    cmd = build_benchmark_command(run, benchmark_script="/path/benchmark_lora.sh")
    assert "--colora_overlap_mode full" in cmd
    assert "--colora_max_continuations 12" in cmd
    assert "--colora_cache_budget_mb 1024" in cmd
    assert "--colora_promote_min_hits 3" in cmd
    assert "--colora_promote_window 64" in cmd
    assert "--colora_max_promote_per_step 4" in cmd
    assert "--colora_decay 0.8" in cmd
    assert "--colora_deferred_promotion_delta_steps 5" in cmd
    assert "--colora_promotion_ema_alpha 0.6" in cmd
    assert "--colora_temporal_prefetch" in cmd
    assert "--colora_temporal_prefetch_layer_whitelist 0,1,2" in cmd
    assert "--colora_temporal_hot_cache_slots 32" in cmd
    assert "--colora_speculative_dispatch" in cmd
    assert "--colora_spec_layer_whitelist 0,1,2" in cmd

def test_build_benchmark_command_injects_kernel_and_packer_env():
    from tools.evaluation.live_e2e.runner import build_benchmark_command
    from tools.evaluation.live_e2e.manifest import LiveE2ERun

    run = LiveE2ERun(
        run_label="colora_kernel_packer",
        suite_kind="paper",
        mode_label="execution_first",
        compute_device="moe_storage:cpu,moe_compute:hybrid",
        cpu_kernel_mode="naive",
        coalescing_packer=False,
    )
    cmd = build_benchmark_command(run, benchmark_script="/path/benchmark_lora.sh")
    assert cmd.startswith("MOE_COALESCING_PACKER=0 COLORA_CPU_KERNEL_MODE=naive ")
    assert "/path/benchmark_lora.sh" in cmd


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

    # cpu_first (execution-first) has correct policy; speculative_dispatch omitted => server default false
    assert "--colora_miss_policy cpu_first" in cmds[1]
    assert "--colora_request_skip 1" in cmds[1]
    assert "cpu_workers 2" in cmds[1]
    assert "MOE_COALESCING_PACKER=1" in cmds[1]
    assert "COLORA_CPU_KERNEL_MODE=avx" in cmds[1]
    assert "--colora_speculative_dispatch" not in cmds[1]
    assert "--no_colora_speculative_dispatch" not in cmds[1]

    # load_then_run has correct policy
    assert "load_then_run" in cmds[2]
    assert "--colora_request_skip 1" in cmds[2]
    assert "MOE_COALESCING_PACKER=1" in cmds[2]
    assert "COLORA_CPU_KERNEL_MODE=avx" in cmds[2]
    assert "--colora_speculative_dispatch" not in cmds[2]
    assert "--no_colora_speculative_dispatch" not in cmds[2]

    assert "MOE_COALESCING_PACKER=1" in cmds[0]
    assert "COLORA_CPU_KERNEL_MODE=avx" in cmds[0]


def test_faithful_component_manifest_parses_and_builds_extended_flags():
    from tools.evaluation.live_e2e.manifest import load_manifest
    from tools.evaluation.live_e2e.runner import build_benchmark_command

    manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/paper_suite_faithful_components.yaml"
    assert manifest_path.exists()
    manifest = load_manifest(manifest_path)
    assert len(manifest.runs) == 4

    cmd0 = build_benchmark_command(manifest.runs[0], "test/lora/benchmark_lora.sh")
    assert "--colora_overlap_mode full" in cmd0
    assert "--colora_max_continuations 8" in cmd0
    assert "--colora_cache_budget_mb 2048" in cmd0
    assert "--no_colora_temporal_prefetch" in cmd0

    cmd2 = build_benchmark_command(manifest.runs[2], "test/lora/benchmark_lora.sh")
    assert "--colora_temporal_prefetch" in cmd2
    assert "--colora_temporal_prefetch_layer_whitelist 0,1,2,3,4,5,6,7" in cmd2
    assert "--colora_temporal_hot_cache_slots 64" in cmd2

    cmd3 = build_benchmark_command(manifest.runs[3], "test/lora/benchmark_lora.sh")
    assert "--colora_speculative_dispatch" in cmd3
    assert "--colora_spec_layer_whitelist 0,1,2,3,4,5,6,7" in cmd3


def test_real_trace_example_manifest_parses_and_builds_trace_flags():
    from tools.evaluation.live_e2e.manifest import load_manifest
    from tools.evaluation.live_e2e.runner import build_benchmark_command

    manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/real_trace_alibaba_example.yaml"
    assert manifest_path.exists()
    manifest = load_manifest(manifest_path)
    assert len(manifest.runs) == 1

    cmd = build_benchmark_command(manifest.runs[0], "test/lora/benchmark_lora.sh")
    assert "--server_host 127.0.0.1" in cmd
    assert "--server_port 18040" in cmd
    assert "--warmup_adapter_trace_path /path/to/adapter_trace_mapped_corr_warmup.jsonl" in cmd
    assert "--measure_adapter_trace_path /path/to/adapter_trace_mapped_corr.jsonl" in cmd


def test_cli_help_includes_manifest_and_summarize_only_flags():
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "tools.evaluation.live_e2e", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "--manifest" in proc.stdout
    assert "--summarize-only" in proc.stdout
    assert "--overwrite" in proc.stdout


def test_read_existing_run_result(tmp_path):
    from tools.evaluation.live_e2e.runner import read_existing_run_result

    assert read_existing_run_result(tmp_path) is None
    bad = tmp_path / "run_result.json"
    bad.write_text("not json", encoding="utf-8")
    assert read_existing_run_result(tmp_path) is None
    good = tmp_path / "run_result.json"
    good.write_text('{"valid": true, "run_label": "x"}', encoding="utf-8")
    d = read_existing_run_result(tmp_path)
    assert d["valid"] is True
    assert d["run_label"] == "x"


def test_run_manifest_skips_valid_runs_without_overwrite(tmp_path, monkeypatch):
    from tools.evaluation.live_e2e.runner import run_manifest

    out_root = tmp_path / "artifacts"
    run_dir = out_root / "resume_manifest_01" / "paper_runs" / "only_run"
    run_dir.mkdir(parents=True)
    per_path = run_dir / "per_request_metrics.jsonl"
    per_path.write_text(
        '{"status": "ok", "latency_s": 0.1, "completion_tokens": 1, '
        '"start_offset_s": 0, "finish_offset_s": 1.0}\n',
        encoding="utf-8",
    )
    run_dir.joinpath("run_result.json").write_text(
        json.dumps(
            {
                "run_label": "only_run",
                "suite_kind": "paper",
                "mode_label": "m",
                "valid": True,
                "per_request_log_path": str(per_path),
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
run_id: resume_manifest_01
runs:
  - run_label: only_run
    suite_kind: paper
    mode_label: m
    compute_device: "vl_storage:gpu,vl_compute:gpu"
    warmup_requests: 0
    measurement_requests: 1
    nsys_enabled: false
    output_root: {out_root}
""",
        encoding="utf-8",
    )

    def _should_not_run(*_a, **_k):
        raise AssertionError("run_single should not be called when prior result is valid")

    monkeypatch.setattr(
        "tools.evaluation.live_e2e.runner.run_single",
        _should_not_run,
    )

    results = run_manifest(manifest_path, overwrite=False)
    assert len(results) == 1
    assert results[0]["valid"] is True
    assert results[0].get("skipped") is True


def test_run_manifest_runs_when_prior_invalid(tmp_path, monkeypatch):
    from tools.evaluation.live_e2e.runner import run_manifest

    out_root = tmp_path / "artifacts"
    run_dir = out_root / "resume_manifest_02" / "paper_runs" / "only_run"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run_result.json").write_text(
        '{"valid": false, "run_label": "only_run", "suite_kind": "paper"}',
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest2.yaml"
    manifest_path.write_text(
        f"""
run_id: resume_manifest_02
runs:
  - run_label: only_run
    suite_kind: paper
    mode_label: m
    compute_device: "vl_storage:gpu,vl_compute:gpu"
    warmup_requests: 0
    measurement_requests: 1
    nsys_enabled: false
    output_root: {out_root}
""",
        encoding="utf-8",
    )

    called = {"n": 0}

    def fake_run_single(manifest, run, capture_stdout=True):
        called["n"] += 1
        return {
            "run_label": run.run_label,
            "suite_kind": run.suite_kind,
            "mode_label": run.mode_label,
            "valid": True,
            "per_request_log_path": str(run_dir / "per_request_metrics.jsonl"),
        }

    monkeypatch.setattr(
        "tools.evaluation.live_e2e.runner.run_single",
        fake_run_single,
    )

    results = run_manifest(manifest_path, overwrite=False)
    assert called["n"] == 1
    assert len(results) == 1
    assert results[0]["valid"] is True
    assert not results[0].get("skipped")


def test_run_manifest_overwrite_runs_despite_valid_prior(tmp_path, monkeypatch):
    from tools.evaluation.live_e2e.runner import run_manifest

    out_root = tmp_path / "artifacts"
    run_dir = out_root / "resume_manifest_03" / "paper_runs" / "only_run"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run_result.json").write_text(
        '{"valid": true, "run_label": "only_run", "suite_kind": "paper", '
        '"mode_label": "m", "per_request_log_path": "/tmp/x"}',
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest3.yaml"
    manifest_path.write_text(
        f"""
run_id: resume_manifest_03
runs:
  - run_label: only_run
    suite_kind: paper
    mode_label: m
    compute_device: "vl_storage:gpu,vl_compute:gpu"
    warmup_requests: 0
    measurement_requests: 1
    nsys_enabled: false
    output_root: {out_root}
""",
        encoding="utf-8",
    )

    called = {"n": 0}

    def fake_run_single(manifest, run, capture_stdout=True):
        called["n"] += 1
        return {
            "run_label": run.run_label,
            "suite_kind": run.suite_kind,
            "mode_label": run.mode_label,
            "valid": True,
            "per_request_log_path": str(run_dir / "per_request_metrics.jsonl"),
        }

    monkeypatch.setattr(
        "tools.evaluation.live_e2e.runner.run_single",
        fake_run_single,
    )

    results = run_manifest(manifest_path, overwrite=True)
    assert called["n"] == 1
    assert not results[0].get("skipped")


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
    assert "--trace=cuda,nvtx" in nsys_cmd
    assert "--force-overwrite=true" in nsys_cmd
    assert "--output=" in nsys_cmd and "/tmp/output/debug_execution_first" in nsys_cmd
    assert " bash -c " in nsys_cmd
    assert "execution_first" in nsys_cmd


def test_benchmark_startup_cleanup_uses_non_fatal_pkill_guards():
    script_path = Path(__file__).resolve().parents[2] / "test/lora/benchmark_lora.sh"
    content = script_path.read_text(encoding="utf-8")
    assert 'if pgrep -f "lightllm.server|lightllm::|gunicorn|multiprocessing.resource_tracker|multiprocessing.spawn" >/dev/null; then' in content
    assert 'pkill -9 -f "lightllm.server|lightllm::|gunicorn" 2>/dev/null || true' in content
    assert 'pkill -9 -f "multiprocessing.resource_tracker|multiprocessing.spawn" 2>/dev/null || true' in content


def test_manifest_lora_clone_count_parses_and_builds_command():
    from tools.evaluation.live_e2e.manifest import load_manifest
    from tools.evaluation.live_e2e.runner import build_benchmark_command

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = """
run_id: clone_test
runs:
  - run_label: colora_with_clones
    suite_kind: paper
    mode_label: execution_first
    compute_device: "all:gpu"
    lora_dirs: /tmp/lora_dummy_0
    lora_clone_count: 10
"""
        manifest_path = _write_temp_manifest(tmp_path, content)
        manifest = load_manifest(manifest_path)
        assert len(manifest.runs) == 1
        assert manifest.runs[0].lora_clone_count == 10

        cmd = build_benchmark_command(manifest.runs[0], "test/lora/benchmark_lora.sh")
        assert "--lora_clone_count 10" in cmd
        assert "--lora_dirs /tmp/lora_dummy_0" in cmd


def test_manifest_max_concurrent_requests_parses_and_builds_command():
    from tools.evaluation.live_e2e.manifest import load_manifest
    from tools.evaluation.live_e2e.runner import build_benchmark_command

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = """
run_id: concurrency_test
runs:
  - run_label: colora_with_concurrency_limit
    suite_kind: paper
    mode_label: colora_min
    compute_device: "all:gpu"
    max_concurrent_requests: 32
    warmup_requests: 512
"""
        manifest_path = _write_temp_manifest(tmp_path, content)
        manifest = load_manifest(manifest_path)
        assert len(manifest.runs) == 1
        assert manifest.runs[0].max_concurrent_requests == 32

        cmd = build_benchmark_command(manifest.runs[0], "test/lora/benchmark_lora.sh")
        assert "--max_concurrent_requests 32" in cmd
        assert "--warmup_num_requests 512" in cmd
