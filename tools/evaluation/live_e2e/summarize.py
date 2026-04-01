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
        "latency_p50_ms": percentile(latencies_ms, 0.50),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "latency_p99_ms": percentile(latencies_ms, 0.99),
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
