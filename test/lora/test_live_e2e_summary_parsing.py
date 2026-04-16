from pathlib import Path
import tempfile
import json
import pytest
from tools.evaluation.live_e2e.summarize import summarize_run, percentile, collect_and_summarize_manifest


def test_percentile_calculation():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(data, 0.50) == 5.5
    assert percentile(data, 0.0) == 1.0
    assert percentile(data, 1.0) == 10.0


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


def test_summarize_missing_per_request_log_is_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        missing_path = tmp_path / "missing.jsonl"

        summary = summarize_run(
            per_request_path=missing_path,
            run_label="missing_run",
            suite_kind="paper",
            mode_label="baseline",
        )

        assert summary["valid"] is False
        assert summary["request_count"] == 0
        assert summary["success_count"] == 0
        assert "not found" in summary["reason"]


def test_collect_and_summarize_manifest_preserves_invalid_run_reason():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        results_json_path = tmp_path / "manifest_results.json"
        output_root = tmp_path / "summaries"

        manifest_results = {
            "run_id": "test_manifest",
            "description": "invalid-run coverage",
            "results": [
                {
                    "run_label": "broken_run",
                    "suite_kind": "paper",
                    "mode_label": "baseline",
                    "valid": False,
                    "failure_reason": "per-request log file not created",
                }
            ],
        }
        results_json_path.write_text(json.dumps(manifest_results), encoding="utf-8")

        summary = collect_and_summarize_manifest(
            manifest_run_id="test_manifest",
            results_json_path=results_json_path,
            output_root=output_root,
        )

        assert summary["total_runs"] == 1
        assert summary["valid_runs"] == 0
        assert summary["summaries"][0]["valid"] is False
        assert summary["summaries"][0]["reason"] == "per-request log file not created"
        assert (output_root / "live_e2e_summary.json").exists()


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
