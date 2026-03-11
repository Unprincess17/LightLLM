#!/usr/bin/env python3
"""Normalize the Azure Functions trace into an adapter-arrival trace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import load_global_config, numeric_summary, stage_output_dir, top_counter_rows, write_csv, write_json


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def as_dict(self) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }
        return {
            "count": self.count,
            "mean_ms": self.total / self.count,
            "min_ms": float(self.min_value or 0.0),
            "max_ms": float(self.max_value or 0.0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Azure Functions trace into JSONL")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override stage output directory")
    parser.add_argument("--trace_path", type=str, default=None, help="Azure trace text file path")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap for preprocessing")
    parser.add_argument("--session_gap_ms", type=int, default=None, help="Idle gap used to derive session ids")
    return parser.parse_args()


def preprocess_trace_file(
    trace_path: Path,
    raw_output_path: Path,
    session_gap_ms: int,
    max_rows: Optional[int] = None,
    top_tenants_to_report: int = 50,
) -> dict:
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    app_counts: Counter = Counter()
    func_counts: Counter = Counter()
    session_counter: Counter = Counter()
    per_app_interarrival: DefaultDict[str, RunningStats] = defaultdict(RunningStats)
    global_interarrival = RunningStats()
    duration_stats = RunningStats()

    last_start_ts_by_app: Dict[str, int] = {}
    last_end_ts_by_app: Dict[str, int] = {}
    prev_global_start_ts: Optional[int] = None
    first_start_ts: Optional[int] = None
    last_end_ts: Optional[int] = None

    parsed_rows = 0
    parse_errors = 0

    with trace_path.open("r", encoding="utf-8", newline="") as handle, raw_output_path.open("w", encoding="utf-8") as writer:
        reader = csv.DictReader(handle)
        for line_num, row in enumerate(reader, start=2):
            if max_rows is not None and parsed_rows >= max_rows:
                break

            try:
                app_id = str(row["app"]).strip()
                func_token = str(row.get("func", "")).strip()
                func_id = func_token or None
                end_ts_ms = int(round(float(row["end_timestamp"]) * 1000.0))
                duration_ms = int(round(float(row["duration"]) * 1000.0))
            except Exception:
                parse_errors += 1
                continue

            if duration_ms < 0:
                parse_errors += 1
                continue

            start_ts_ms = end_ts_ms - duration_ms
            if first_start_ts is None or start_ts_ms < first_start_ts:
                first_start_ts = start_ts_ms
            if last_end_ts is None or end_ts_ms > last_end_ts:
                last_end_ts = end_ts_ms

            if prev_global_start_ts is not None:
                gap_ms = start_ts_ms - prev_global_start_ts
                if gap_ms >= 0:
                    global_interarrival.add(float(gap_ms))
            prev_global_start_ts = start_ts_ms

            previous_start_for_app = last_start_ts_by_app.get(app_id)
            if previous_start_for_app is not None:
                gap_ms = start_ts_ms - previous_start_for_app
                if gap_ms >= 0:
                    per_app_interarrival[app_id].add(float(gap_ms))

            previous_end_for_app = last_end_ts_by_app.get(app_id)
            if previous_end_for_app is None or (start_ts_ms - previous_end_for_app) > session_gap_ms:
                session_counter[app_id] += 1
            session_id = f"{app_id}:{session_counter[app_id]}"

            last_start_ts_by_app[app_id] = start_ts_ms
            last_end_ts_by_app[app_id] = end_ts_ms
            app_counts[app_id] += 1
            if func_id is not None:
                func_counts[func_id] += 1
            duration_stats.add(float(duration_ms))

            record = {
                "arrival_idx": parsed_rows,
                "start_ts": start_ts_ms,
                "end_ts": end_ts_ms,
                "duration_ms": duration_ms,
                "app_id": app_id,
                "func_id": func_id,
                "session_id": session_id,
                "raw_trace_source": trace_path.name,
            }
            writer.write(json.dumps(record, ensure_ascii=True))
            writer.write("\n")
            parsed_rows += 1

            if parsed_rows % 200000 == 0:
                print(f"processed {parsed_rows} rows")

    tenant_rows = top_counter_rows(app_counts, "app_id", total=parsed_rows, limit=max(len(app_counts), top_tenants_to_report))
    interarrival_rows = [
        {
            "scope": "global",
            "key": "all",
            **global_interarrival.as_dict(),
            "request_count": parsed_rows,
        }
    ]
    for tenant_row in top_counter_rows(app_counts, "app_id", total=parsed_rows, limit=top_tenants_to_report):
        stats = per_app_interarrival.get(str(tenant_row["app_id"]), RunningStats()).as_dict()
        interarrival_rows.append(
            {
                "scope": "app",
                "key": tenant_row["app_id"],
                "request_count": tenant_row["count"],
                **stats,
            }
        )

    summary = {
        "trace_path": str(trace_path),
        "row_count": parsed_rows,
        "parse_errors": parse_errors,
        "duration_valid_ratio": float(parsed_rows) / max(parsed_rows + parse_errors, 1),
        "unique_apps": len(app_counts),
        "unique_funcs": len(func_counts),
        "first_start_ts": first_start_ts,
        "last_end_ts": last_end_ts,
        "duration_ms": duration_stats.as_dict(),
        "top_tenants": top_counter_rows(app_counts, "app_id", total=parsed_rows, limit=top_tenants_to_report),
    }
    return {
        "summary": summary,
        "tenant_rows": tenant_rows,
        "interarrival_rows": interarrival_rows,
    }


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    case_config = config.get("case_study", {})
    azure_config = case_config.get("azure_trace", {})
    paths_config = config.get("paths", {})

    trace_path = Path(args.trace_path or paths_config["azure_trace"])
    session_gap_ms = int(args.session_gap_ms or azure_config.get("session_gap_ms", 300000))
    top_tenants_to_report = int(azure_config.get("top_tenants_to_report", 50))
    output_dir = stage_output_dir("adapter_trace", config, args.run_id, args.output_dir)

    raw_output_path = output_dir / "adapter_trace_raw.jsonl"
    summary_path = output_dir / "adapter_trace_summary.json"
    tenant_popularity_path = output_dir / "tenant_popularity.csv"
    interarrival_stats_path = output_dir / "interarrival_stats.csv"

    outputs = preprocess_trace_file(
        trace_path=trace_path,
        raw_output_path=raw_output_path,
        session_gap_ms=session_gap_ms,
        max_rows=args.max_rows,
        top_tenants_to_report=top_tenants_to_report,
    )
    write_json(summary_path, outputs["summary"])
    write_csv(tenant_popularity_path, ["rank", "app_id", "count", "share"], outputs["tenant_rows"])
    write_csv(
        interarrival_stats_path,
        ["scope", "key", "request_count", "count", "mean_ms", "min_ms", "max_ms"],
        outputs["interarrival_rows"],
    )

    print(f"wrote raw adapter trace to {raw_output_path}")


if __name__ == "__main__":
    main()
