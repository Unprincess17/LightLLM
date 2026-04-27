#!/usr/bin/env python3
"""Compare COLoRA load-then-run and cpu-first live E2E policy results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.case_study.common import ensure_parent_dir, percentile, write_csv, write_json
from tools.evaluation.live_e2e.summarize import load_per_request_metrics


DEFAULT_ARTIFACTS_ROOT = Path("artifacts/evaluation/live_e2e")


def _mode_key(summary: Mapping[str, Any]) -> str:
    label = str(summary.get("run_label", "")).lower()
    mode = str(summary.get("mode_label", "")).lower()
    if "miss-load_then_run" in label or "load_then_run" in mode:
        return "load_then_run"
    if "miss-cpu_first" in label or "cpu_first" in mode or "execution_first" in mode:
        return "cpu_first"
    return mode


def _metrics_path(
    artifacts_root: Path,
    run_id: str,
    summary: Mapping[str, Any],
    manifest_results_by_label: Mapping[str, Mapping[str, Any]],
) -> Path:
    label = str(summary["run_label"])
    result = manifest_results_by_label.get(label)
    if result and result.get("per_request_log_path"):
        return Path(str(result["per_request_log_path"]))

    suite_kind = str(summary.get("suite_kind", "paper"))
    suite_dir = "paper_runs" if suite_kind == "paper" else "diagnostic_runs"
    return artifacts_root / run_id / suite_dir / label / "per_request_metrics.jsonl"


def _window_seconds(successes: Sequence[Mapping[str, Any]]) -> float:
    if not successes:
        return 0.0
    if all("start_offset_s" in row and "finish_offset_s" in row for row in successes):
        start = min(float(row["start_offset_s"]) for row in successes)
        finish = max(float(row["finish_offset_s"]) for row in successes)
        return max(finish - start, 0.0)
    return sum(float(row.get("latency_s", 0.0)) for row in successes)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def summarize_per_request_metrics(
    per_request_path: Path,
    *,
    run_label: str,
    mode_label: str,
    suite_kind: str,
) -> dict[str, Any]:
    rows = load_per_request_metrics(per_request_path)
    request_count = len(rows)
    successes = [row for row in rows if row.get("status") == "ok"]
    success_count = len(successes)
    latencies_ms = sorted(float(row["latency_s"]) * 1000.0 for row in successes)
    window_s = _window_seconds(successes)
    completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in successes)

    return {
        "run_label": run_label,
        "suite_kind": suite_kind,
        "mode_label": mode_label,
        "request_count": request_count,
        "success_count": success_count,
        "error_count": request_count - success_count,
        "error_rate": _rounded((request_count - success_count) / request_count) if request_count else 0.0,
        "latency_p50_ms": _rounded(percentile(latencies_ms, 0.50)),
        "latency_p90_ms": _rounded(percentile(latencies_ms, 0.90)),
        "latency_p95_ms": _rounded(percentile(latencies_ms, 0.95)),
        "latency_p99_ms": _rounded(percentile(latencies_ms, 0.99)),
        "latency_max_ms": _rounded(max(latencies_ms)) if latencies_ms else 0.0,
        "rps": _rounded(success_count / window_s) if window_s > 0 else 0.0,
        "tokens_per_second": _rounded(completion_tokens / window_s) if window_s > 0 else 0.0,
        "per_request_metrics_path": str(per_request_path),
    }


def _load_manifest_results(summary_dir: Path) -> dict[str, Mapping[str, Any]]:
    path = summary_dir / "manifest_results.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["run_label"]): row for row in payload.get("results", [])}


def _invalid_mode_messages(summaries: Sequence[Mapping[str, Any]], missing_modes: Sequence[str]) -> list[str]:
    invalid_by_mode = {
        _mode_key(summary): summary
        for summary in summaries
        if summary.get("valid") is False
    }
    messages = []
    for mode in missing_modes:
        summary = invalid_by_mode.get(mode)
        if summary is None:
            continue
        reason = str(summary.get("reason", "invalid summary"))
        run_label = str(summary.get("run_label", ""))
        success_count = summary.get("success_count")
        request_count = summary.get("request_count")
        messages.append(
            f"{mode} invalid ({reason}; successes={success_count}/{request_count}; run_label={run_label})"
        )
    return messages


def compare_run_id(
    run_id: str,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    p99_ratio_gate: float = 4.0,
) -> dict[str, Any]:
    summary_dir = artifacts_root / run_id / "summaries"
    summary_path = summary_dir / "live_e2e_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_results_by_label = _load_manifest_results(summary_dir)

    runs = []
    for summary in payload.get("summaries", []):
        if summary.get("valid") is False:
            continue
        path = _metrics_path(artifacts_root, run_id, summary, manifest_results_by_label)
        runs.append(
            summarize_per_request_metrics(
                path,
                run_label=str(summary["run_label"]),
                mode_label=str(summary.get("mode_label", "")),
                suite_kind=str(summary.get("suite_kind", "")),
            )
        )

    by_mode = {_mode_key(row): row for row in runs}
    missing = [mode for mode in ("load_then_run", "cpu_first") if mode not in by_mode]
    if missing:
        invalid_messages = _invalid_mode_messages(payload.get("summaries", []), missing)
        if invalid_messages:
            raise ValueError(
                "required run mode(s) are invalid: " + "; ".join(invalid_messages)
            )
        raise ValueError(f"missing required run mode(s): {', '.join(missing)}")

    load_then_run = by_mode["load_then_run"]
    cpu_first = by_mode["cpu_first"]
    cpu_p99 = float(cpu_first["latency_p99_ms"])
    ratio = _rounded(float(load_then_run["latency_p99_ms"]) / cpu_p99) if cpu_p99 > 0 else 0.0
    comparison = {
        "load_then_run_label": load_then_run["run_label"],
        "cpu_first_label": cpu_first["run_label"],
        "p99_ratio_load_then_run_over_cpu_first": ratio,
        "p99_ratio_gate": p99_ratio_gate,
        "p99_ratio_gate_pass": ratio >= p99_ratio_gate,
    }
    return {
        "run_id": run_id,
        "artifacts_root": str(artifacts_root),
        "runs": runs,
        "comparison": comparison,
    }


def write_outputs(result: Mapping[str, Any], *, json_output: Path, csv_output: Path) -> None:
    write_json(json_output, result)
    comparison = result["comparison"]
    rows = []
    for run in result["runs"]:
        rows.append(
            {
                **run,
                "p99_ratio_load_then_run_over_cpu_first": comparison["p99_ratio_load_then_run_over_cpu_first"],
                "p99_ratio_gate": comparison["p99_ratio_gate"],
                "p99_ratio_gate_pass": comparison["p99_ratio_gate_pass"],
            }
        )
    fieldnames = [
        "run_label",
        "suite_kind",
        "mode_label",
        "request_count",
        "success_count",
        "error_count",
        "error_rate",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_max_ms",
        "rps",
        "tokens_per_second",
        "p99_ratio_load_then_run_over_cpu_first",
        "p99_ratio_gate",
        "p99_ratio_gate_pass",
        "per_request_metrics_path",
    ]
    write_csv(csv_output, fieldnames, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="live_e2e run_id under artifacts/evaluation/live_e2e")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--p99-ratio-gate", type=float, default=4.0)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_run_id(args.run_id, artifacts_root=args.artifacts_root, p99_ratio_gate=args.p99_ratio_gate)
    summary_dir = args.artifacts_root / args.run_id / "summaries"
    json_output = args.json_output or summary_dir / "colora_policy_compare.json"
    csv_output = args.csv_output or summary_dir / "colora_policy_compare.csv"
    ensure_parent_dir(json_output)
    ensure_parent_dir(csv_output)
    write_outputs(result, json_output=json_output, csv_output=csv_output)
    print(f"Wrote JSON: {json_output}")
    print(f"Wrote CSV: {csv_output}")
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
    return 0 if result["comparison"]["p99_ratio_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
