import json
import pytest
from pathlib import Path
import tempfile

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.lora.trace_expert_injection import TraceExpertInjection
from tools.evaluation.live_e2e.manifest import load_manifest, LiveE2EManifest, LiveE2ERun
from tools.evaluation.live_e2e.runner import build_benchmark_command


def _write_temp_manifest(tmp_path: Path, content: str) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(content, encoding="utf-8")
    return manifest_path


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestStartArgsRouterTracePath:
    def test_router_trace_path_defaults_to_none(self):
        args = StartArgs()
        assert args.router_trace_path is None

    def test_router_trace_path_can_be_set(self):
        args = StartArgs(router_trace_path="/tmp/trace.jsonl")
        assert args.router_trace_path == "/tmp/trace.jsonl"

    def test_router_trace_path_is_not_duplicated(self):
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(StartArgs)]
        count = field_names.count("router_trace_path")
        assert count == 1, f"router_trace_path appears {count} times in StartArgs fields"


class TestLiveE2ERunRouterTracePath:
    def test_router_trace_path_defaults_to_none(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="test",
            compute_device="all:gpu",
        )
        assert run.router_trace_path is None

    def test_router_trace_path_can_be_set(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="test",
            compute_device="all:gpu",
            router_trace_path="/tmp/router_trace.jsonl",
        )
        assert run.router_trace_path == "/tmp/router_trace.jsonl"

    def test_router_trace_path_in_to_metadata(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="test",
            compute_device="all:gpu",
            router_trace_path="/tmp/router_trace.jsonl",
        )
        meta = run.to_metadata()
        assert "router_trace_path" in meta
        assert meta["router_trace_path"] == "/tmp/router_trace.jsonl"

    def test_router_trace_path_none_in_to_metadata(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="test",
            compute_device="all:gpu",
        )
        meta = run.to_metadata()
        assert "router_trace_path" in meta
        assert meta["router_trace_path"] is None


class TestBuildBenchmarkCommandRouterTracePath:
    def test_router_trace_path_included_when_set(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="colora_min",
            compute_device="all:gpu",
            router_trace_path="artifacts/evaluation/p3_traces/pressure_high_async/high_async_trace.jsonl",
        )
        cmd = build_benchmark_command(run, "test/lora/benchmark_lora.sh")
        assert "--router_trace_path artifacts/evaluation/p3_traces/pressure_high_async/high_async_trace.jsonl" in cmd

    def test_router_trace_path_omitted_when_none(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="baseline",
            compute_device="all:gpu",
        )
        cmd = build_benchmark_command(run, "test/lora/benchmark_lora.sh")
        assert "--router_trace_path" not in cmd

    def test_router_trace_path_with_other_trace_flags(self):
        run = LiveE2ERun(
            run_label="test",
            suite_kind="paper",
            mode_label="colora_full",
            compute_device="all:gpu",
            warmup_adapter_trace_path="/tmp/warmup.jsonl",
            measurement_adapter_trace_path="/tmp/measure.jsonl",
            router_trace_path="/tmp/router_trace.jsonl",
        )
        cmd = build_benchmark_command(run, "test/lora/benchmark_lora.sh")
        assert "--warmup_adapter_trace_path /tmp/warmup.jsonl" in cmd
        assert "--measure_adapter_trace_path /tmp/measure.jsonl" in cmd
        assert "--router_trace_path /tmp/router_trace.jsonl" in cmd


class TestManifestParsingRouterTracePath:
    def test_load_manifest_with_router_trace_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            content = """
run_id: router_trace_test
runs:
  - run_label: colora_with_router_trace
    suite_kind: paper
    mode_label: colora_min
    compute_device: "all:gpu"
    router_trace_path: artifacts/evaluation/p3_traces/pressure_high_async/high_async_trace.jsonl
"""
            manifest_path = _write_temp_manifest(tmp_path, content)
            manifest = load_manifest(manifest_path)
            assert len(manifest.runs) == 1
            assert manifest.runs[0].router_trace_path == "artifacts/evaluation/p3_traces/pressure_high_async/high_async_trace.jsonl"

    def test_load_manifest_without_router_trace_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            content = """
run_id: no_router_trace_test
runs:
  - run_label: baseline
    suite_kind: paper
    mode_label: baseline
    compute_device: "all:gpu"
"""
            manifest_path = _write_temp_manifest(tmp_path, content)
            manifest = load_manifest(manifest_path)
            assert len(manifest.runs) == 1
            assert manifest.runs[0].router_trace_path is None

    def test_load_manifest_mixed_runs_with_and_without_router_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            content = """
run_id: mixed_router_trace_test
runs:
  - run_label: baseline_no_trace
    suite_kind: diagnostic
    mode_label: baseline
    compute_device: "all:gpu"
  - run_label: colora_with_trace
    suite_kind: paper
    mode_label: colora_min
    compute_device: "all:gpu"
    router_trace_path: /tmp/router_trace.jsonl
"""
            manifest_path = _write_temp_manifest(tmp_path, content)
            manifest = load_manifest(manifest_path)
            assert len(manifest.runs) == 2
            assert manifest.runs[0].router_trace_path is None
            assert manifest.runs[1].router_trace_path == "/tmp/router_trace.jsonl"


class TestTraceExpertInjectionIntegration:
    NUM_EXPERTS = 32
    TOP_K = 4

    def _build_trace_rows(self):
        rows = []
        for req_idx in range(4):
            for layer_id in range(4):
                for token_pos in range(3):
                    seed = req_idx * 1000 + layer_id * 50 + token_pos
                    expert_set = [(seed + rank * 13) % self.NUM_EXPERTS for rank in range(self.TOP_K)]
                    rows.append({
                        "event": "router_trace",
                        "arrival_idx": len(rows),
                        "req_idx": req_idx,
                        "phase": "decode",
                        "layer_id": layer_id,
                        "token_pos": token_pos,
                        "topk_experts": expert_set,
                        "topk_weights": [1.0 / self.TOP_K] * self.TOP_K,
                    })
        return rows

    def test_load_router_trace_from_file(self, tmp_path):
        trace_path = tmp_path / "router_trace.jsonl"
        _write_jsonl(trace_path, self._build_trace_rows())

        injection = TraceExpertInjection(self.NUM_EXPERTS)
        event_count = injection.load_router_trace(str(trace_path))
        assert event_count == 4 * 4 * 3

    def test_load_router_trace_nonexistent_file(self):
        injection = TraceExpertInjection(self.NUM_EXPERTS)
        with pytest.raises(Exception):
            injection.load_router_trace("/nonexistent/path/trace.jsonl")

    def test_load_router_trace_empty_file(self, tmp_path):
        trace_path = tmp_path / "empty_trace.jsonl"
        _write_jsonl(trace_path, [])

        injection = TraceExpertInjection(self.NUM_EXPERTS)
        event_count = injection.load_router_trace(str(trace_path))
        assert event_count == 0

    def test_get_experts_after_load(self, tmp_path):
        trace_path = tmp_path / "router_trace.jsonl"
        _write_jsonl(trace_path, self._build_trace_rows())

        injection = TraceExpertInjection(self.NUM_EXPERTS)
        injection.load_router_trace(str(trace_path))

        experts = injection.get_experts(layer_id=0, req_idx=0, token_pos=0)
        assert experts is not None
        assert len(experts) == self.TOP_K
        assert all(0 <= e < self.NUM_EXPERTS for e in experts)

    def test_get_experts_missing_key(self, tmp_path):
        trace_path = tmp_path / "router_trace.jsonl"
        _write_jsonl(trace_path, self._build_trace_rows())

        injection = TraceExpertInjection(self.NUM_EXPERTS)
        injection.load_router_trace(str(trace_path))

        assert injection.get_experts(layer_id=99, req_idx=99, token_pos=99) is None

    def test_verify_trace_coverage(self, tmp_path):
        trace_path = tmp_path / "router_trace.jsonl"
        _write_jsonl(trace_path, self._build_trace_rows())

        injection = TraceExpertInjection(self.NUM_EXPERTS)
        injection.load_router_trace(str(trace_path))

        coverage = injection.verify_trace_coverage(
            req_indices=[0, 1, 99],
            token_positions=[0, 0, 0],
            layers=[0, 1],
        )
        assert coverage["total"] == 6
        assert coverage["covered"] == 4
        assert coverage["missing"] == 2


class TestManagerKvargsRouterTracePath:
    def test_router_trace_path_in_kvargs_construction(self):
        manager_path = Path(__file__).resolve().parents[2] / "lightllm/server/router/manager.py"
        content = manager_path.read_text(encoding="utf-8")
        assert '"router_trace_path"' in content, (
            "router_trace_path key must be present in RouterManager kvargs dict"
        )


class TestBenchmarkLoraShellRouterTracePath:
    def test_benchmark_script_accepts_router_trace_path(self):
        script_path = Path(__file__).resolve().parents[2] / "test/lora/benchmark_lora.sh"
        content = script_path.read_text(encoding="utf-8")
        assert "--router_trace_path" in content, (
            "benchmark_lora.sh must accept --router_trace_path argument"
        )

    def test_benchmark_script_forwards_router_trace_path_to_server(self):
        script_path = Path(__file__).resolve().parents[2] / "test/lora/benchmark_lora.sh"
        content = script_path.read_text(encoding="utf-8")
        assert "ROUTER_TRACE_PATH" in content, (
            "benchmark_lora.sh must have ROUTER_TRACE_PATH variable"
        )
        assert '--router_trace_path "$ROUTER_TRACE_PATH"' in content or \
               "--router_trace_path $ROUTER_TRACE_PATH" in content or \
               '--router_trace_path "$ROUTER_TRACE_PATH"' in content, (
            "benchmark_lora.sh must forward ROUTER_TRACE_PATH to start_server.sh"
        )


class TestStartServerShellRouterTracePath:
    def test_start_server_passes_router_trace_path_as_cli_arg(self):
        script_path = Path(__file__).resolve().parents[2] / "test/lora/start_server.sh"
        content = script_path.read_text(encoding="utf-8")
        assert "--router_trace_path" in content, (
            "start_server.sh must accept --router_trace_path argument"
        )


class TestApiCliRouterTracePath:
    def test_api_cli_has_router_trace_path_argument(self):
        from lightllm.server.api_cli import make_argument_parser
        parser = make_argument_parser()
        actions = {a.dest: a for a in parser._actions}
        assert "router_trace_path" in actions, (
            "api_cli must have --router_trace_path argument"
        )
        assert actions["router_trace_path"].default is None


class TestP3PressureHighAsyncManifest:
    def test_manifest_file_exists(self):
        manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/p3_pressure_high_async.yaml"
        assert manifest_path.exists(), "p3_pressure_high_async.yaml must exist"

    def test_manifest_loads_successfully(self):
        manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/p3_pressure_high_async.yaml"
        manifest = load_manifest(manifest_path)
        assert manifest.run_id == "p3_pressure_high_async"
        assert len(manifest.runs) >= 2

    def test_manifest_has_router_trace_path_in_colora_runs(self):
        manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/p3_pressure_high_async.yaml"
        manifest = load_manifest(manifest_path)
        runs_with_trace = [r for r in manifest.runs if r.router_trace_path is not None]
        assert len(runs_with_trace) >= 1, (
            "At least one run must have router_trace_path set"
        )

    def test_manifest_load_then_run_has_no_router_trace(self):
        manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/p3_pressure_high_async.yaml"
        manifest = load_manifest(manifest_path)
        ltr_runs = [r for r in manifest.runs if r.mode_label == "load_then_run"]
        assert len(ltr_runs) >= 1
        for run in ltr_runs:
            assert run.router_trace_path is None, (
                "load_then_run baseline should not have router_trace_path"
            )

    def test_manifest_builds_valid_commands(self):
        manifest_path = Path(__file__).resolve().parents[2] / "configs/live_e2e/p3_pressure_high_async.yaml"
        manifest = load_manifest(manifest_path)
        for run in manifest.runs:
            cmd = build_benchmark_command(run, "test/lora/benchmark_lora.sh")
            assert "--compute_device" in cmd
            if run.router_trace_path is not None:
                assert f"--router_trace_path {run.router_trace_path}" in cmd
