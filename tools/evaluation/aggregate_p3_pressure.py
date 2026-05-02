#!/usr/bin/env python3
"""Aggregate P3 pressure E2E run results into a summary CSV.

Reads per_request_metrics.jsonl and colora_stats.json from each run directory
under the input root, computes TPOT percentiles and throughput, and writes
pressure_e2e_summary.csv.

Usage:
    python tools/evaluation/aggregate_p3_pressure.py \
        --input results/p3_pressure \
        --output artifacts/evaluation/p3
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.case_study.common import ensure_parent_dir, percentile


_COLORA_LINE_RE = re.compile(r"\[COLoRA\]\s+layer=.*")
_KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _extract_colora_stats_from_log(stdout_log_path: Path) -> Dict[str, Any]:
    """Parse [COLoRA] lines from server stdout log and aggregate stats."""
    if not stdout_log_path.exists():
        return {}

    per_key: Dict[str, List[float]] = {}
    with stdout_log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not _COLORA_LINE_RE.search(line):
                continue
            for k, v in _KV_RE.findall(line):
                try:
                    fv = float(v.rstrip(","))
                except ValueError:
                    continue
                per_key.setdefault(k, []).append(fv)

    if not per_key:
        return {}

    counter_keys = {
        "colora_hit_tokens", "colora_miss_tokens",
        "hit_tokens", "miss_tokens",
        "cpu_compute_time", "gpu_compute_time",
        "cpu_queue_wait_time", "cpu_queue_wait",
        "cpu_join_stall_time",
        "d2h_bytes", "h2d_bytes",
        "weight_h2d_bytes", "weight_h2d_time",
        "d2h_activation_time", "h2d_residual_time",
        "pack_time", "merge_time", "admit_time",
        "fallback_degrade_count",
        "promotion_drop_total", "promotion_admitted",
        "promotion_reject_delta", "promotion_reject_no_ema",
        "tracker_queue_drop",
        "prefetch_submitted", "prefetch_ready_hits",
        "prefetch_not_ready", "prefetch_stale",
        "prefetch_false_positives", "prefetch_slot_overwrite",
        "moe_kernel_calls", "moe_kernel_tokens",
        "attempted_bind", "successful_bind",
        "cpu_async_submitted", "cpu_inline_executed",
        "blocking_promotion_count",
        "cache_evictions_total",
    }
    gauge_keys = {
        "cpu_queue_depth", "promotion_drop_queue_high_watermark",
        "cache_capacity_slots", "cache_resident_slots", "cache_free_slots",
        "overlap_ratio",
    }

    result: Dict[str, Any] = {}
    for k, vs in per_key.items():
        if k in counter_keys:
            result[k] = sum(vs)
        elif k in gauge_keys:
            result[k] = vs[-1] if vs else 0
        else:
            result[k] = vs[-1] if vs else 0

    # Map log key aliases to canonical names
    if "hit_tokens" in result and "colora_hit_tokens" not in result:
        result["colora_hit_tokens"] = result["hit_tokens"]
    if "miss_tokens" in result and "colora_miss_tokens" not in result:
        result["colora_miss_tokens"] = result["miss_tokens"]
    if "cpu_queue_wait" in result and "cpu_queue_wait_time" not in result:
        result["cpu_queue_wait_time"] = result["cpu_queue_wait"]

    hit = result.get("colora_hit_tokens", 0)
    miss = result.get("colora_miss_tokens", 0)
    total = hit + miss
    if total > 0:
        result["cache_hit_rate"] = hit / total
        result["cache_miss_rate"] = miss / total

    return result


def _parse_run_label(label: str) -> Dict[str, str]:
    """Extract structured fields from P3 run label convention.

    Expected format: p3__stage-X__trace-NAME__pressure-LEVEL__...
    """
    parts = {}
    for segment in label.split("__"):
        if "-" not in segment:
            continue
        key, _, value = segment.partition("-")
        parts[key] = value
    return parts


def _aggregate_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate a single run directory into a summary row."""
    per_req_path = run_dir / "per_request_metrics.jsonl"
    colora_path = run_dir / "colora_stats.json"
    result_path = run_dir / "run_result.json"

    if not per_req_path.exists():
        return None

    manifest = _load_json(run_dir / "config_snapshot.json") or {}
    run_result = _load_json(result_path) or {}

    run_label = run_result.get("run_label", run_dir.name)
    parsed = _parse_run_label(run_label)

    mode_label = manifest.get("mode_label", parsed.get("miss", ""))
    cache_budget_mb = manifest.get("cache_budget_mb", 0)
    trace_name = parsed.get("trace", "")
    pressure = parsed.get("pressure", "")

    # Load per-request metrics (measurement phase only)
    all_metrics = _load_jsonl(per_req_path)
    measurement = [m for m in all_metrics if m.get("phase") == "measurement"]
    if not measurement and all_metrics:
        measurement = all_metrics

    ok_metrics = [m for m in measurement if m.get("status") == "ok"]
    fail_metrics = [m for m in measurement if m.get("status") != "ok"]

    num_requests = len(measurement)
    num_failed = len(fail_metrics)

    row: Dict[str, Any] = {
        "run_id": run_label,
        "policy": mode_label,
        "trace_name": trace_name,
        "cache_pressure": pressure,
        "cache_budget_mb": cache_budget_mb,
        "num_requests": num_requests,
        "num_failed": num_failed,
    }

    # TPOT percentiles (in microseconds)
    if ok_metrics:
        tpot_values = [m["tpot_mean_us"] for m in ok_metrics if m.get("tpot_mean_us") is not None]
        if tpot_values:
            tpot_values.sort()
            row["p50_tpot_us"] = percentile(tpot_values, 0.50)
            row["p90_tpot_us"] = percentile(tpot_values, 0.90)
            row["p95_tpot_us"] = percentile(tpot_values, 0.95)
            row["p99_tpot_us"] = percentile(tpot_values, 0.99)
        else:
            row["p50_tpot_us"] = None
            row["p90_tpot_us"] = None
            row["p95_tpot_us"] = None
            row["p99_tpot_us"] = None

        # Throughput
        total_tokens = sum(m.get("completion_tokens", 0) for m in ok_metrics)
        offsets_start = [m.get("start_offset_s", 0) for m in ok_metrics if m.get("start_offset_s") is not None]
        offsets_end = [m.get("finish_offset_s", 0) for m in ok_metrics if m.get("finish_offset_s") is not None]
        if offsets_start and offsets_end:
            window = max(offsets_end) - min(offsets_start)
            if window > 0:
                row["throughput_req_s"] = len(ok_metrics) / window
                row["throughput_tok_s"] = total_tokens / window
            else:
                row["throughput_req_s"] = 0
                row["throughput_tok_s"] = 0
        else:
            total_latency = sum(m.get("latency_s", 0) for m in ok_metrics if m.get("latency_s") is not None)
            row["throughput_req_s"] = len(ok_metrics) / total_latency if total_latency > 0 else 0
            row["throughput_tok_s"] = total_tokens / total_latency if total_latency > 0 else 0
    else:
        row["p50_tpot_us"] = None
        row["p90_tpot_us"] = None
        row["p95_tpot_us"] = None
        row["p99_tpot_us"] = None
        row["throughput_req_s"] = 0
        row["throughput_tok_s"] = 0

    # CoLoRA stats: prefer JSON file, fall back to parsing stdout log
    colora = _load_json(colora_path)
    if not colora or colora_path.stat().st_size <= 2:
        log_stats = _extract_colora_stats_from_log(run_dir / "benchmark_stdout.log")
        if log_stats:
            colora = log_stats
    if colora is None:
        colora = {}

    row["observed_miss_rate"] = colora.get("cache_miss_rate", None)
    row["cpu_queue_wait_time_us"] = colora.get("cpu_queue_wait_time", 0)
    row["promotion_bytes"] = colora.get("weight_h2d_bytes", 0)
    row["activation_bytes"] = colora.get("d2h_bytes", 0)
    row["residual_bytes"] = colora.get("h2d_bytes", 0)

    # Blocking promotion metrics for load_then_run foreground recovery pressure
    row["blocking_promotion_count"] = colora.get("blocking_promotion_count", 0)
    row["blocking_promotion_bytes"] = colora.get("blocking_promotion_bytes", 0)
    row["blocking_promotion_time"] = colora.get("blocking_promotion_time", 0)

    # Pre-promotion key-level counters (demand miss BEFORE blocking promotion)
    row["demand_key_access_count"] = colora.get("demand_key_access_count", 0)
    row["demand_key_miss_count"] = colora.get("demand_key_miss_count", 0)

    return row


def _discover_run_dirs(input_root: Path) -> List[Path]:
    """Find all run directories containing per_request_metrics.jsonl."""
    run_dirs = []
    # First try paper_runs / diagnostic_runs subdirectories
    for subdir in input_root.rglob("paper_runs"):
        for run_dir in subdir.iterdir():
            if run_dir.is_dir() and (run_dir / "per_request_metrics.jsonl").exists():
                run_dirs.append(run_dir)
    for subdir in input_root.rglob("diagnostic_runs"):
        for run_dir in subdir.iterdir():
            if run_dir.is_dir() and (run_dir / "per_request_metrics.jsonl").exists():
                run_dirs.append(run_dir)
    # Fallback: look directly under input_root for run directories
    if not run_dirs and input_root.is_dir():
        for run_dir in input_root.iterdir():
            if run_dir.is_dir() and (run_dir / "per_request_metrics.jsonl").exists():
                run_dirs.append(run_dir)
        # Also walk one level deeper
        for subdir in input_root.iterdir():
            if subdir.is_dir():
                for run_dir in subdir.iterdir():
                    if run_dir.is_dir() and (run_dir / "per_request_metrics.jsonl").exists():
                        run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def main():
    parser = argparse.ArgumentParser(description="Aggregate P3 pressure E2E results")
    parser.add_argument("--input", type=Path, required=True, help="Root directory containing run results")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for summary CSV")
    args = parser.parse_args()

    input_root = args.input
    output_dir = args.output
    ensure_parent_dir(output_dir / "pressure_e2e_summary.csv")

    run_dirs = _discover_run_dirs(input_root)
    if not run_dirs:
        print(f"No run directories found under {input_root}")
        return

    print(f"Found {len(run_dirs)} run directories")

    rows = []
    for run_dir in run_dirs:
        row = _aggregate_run(run_dir)
        if row is not None:
            rows.append(row)
            print(f"  {row['run_id']}: {row['num_requests']} reqs, "
                  f"p50_tpot={row.get('p50_tpot_us', 'N/A')} us, "
                  f"miss_rate={row.get('observed_miss_rate', 'N/A')}")

    if not rows:
        print("No valid runs to aggregate")
        return

    # Write CSV with minimal required columns
    fieldnames = [
        "run_id", "policy", "trace_name", "cache_pressure", "cache_budget_mb",
        "num_requests", "num_failed",
        "observed_miss_rate",
        "p50_tpot_us", "p90_tpot_us", "p95_tpot_us", "p99_tpot_us",
        "throughput_req_s", "throughput_tok_s",
        "promotion_bytes", "activation_bytes", "residual_bytes",
        "cpu_queue_wait_time_us",
        "blocking_promotion_count", "blocking_promotion_bytes", "blocking_promotion_time",
        "demand_key_access_count", "demand_key_miss_count",
    ]

    csv_path = output_dir / "pressure_e2e_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSummary written to {csv_path}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
