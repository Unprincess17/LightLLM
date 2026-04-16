import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

from manifest import load_manifest

COLORA_LINE_RE = re.compile(r"\[COLoRA\]\s+layer=.*")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


def _safe_float(v: str):
    try:
        return float(v)
    except ValueError:
        return None


def _safe_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def _extract_colora_metrics(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {}
    if not log_path.exists():
        return {}

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not COLORA_LINE_RE.search(line):
                continue
            for k, v in KV_RE.findall(line):
                fv = _safe_float(v.rstrip(","))
                if fv is None:
                    continue
                metrics.setdefault(k, []).append(fv)

    return {f"{k}_avg": mean(vs) for k, vs in metrics.items() if vs}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate fixed-matrix live_e2e summaries and COLoRA mechanism metrics."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    output_root = manifest.runs[0].output_root / manifest.run_id
    summaries_dir = output_root / "summaries"
    summary_json = summaries_dir / "live_e2e_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary not found: {summary_json}")

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    run_rows = []
    grouped: Dict[str, Dict[str, List[float]]] = {}

    for s in summary["summaries"]:
        if not s.get("valid"):
            continue
        run_label = s["run_label"]
        mode_label = s["mode_label"]
        run_dir = output_root / "paper_runs" / run_label
        log_metrics = _extract_colora_metrics(run_dir / "benchmark_stdout.log")
        row = {
            "run_label": run_label,
            "mode_label": mode_label,
            "throughput_tokens_per_second": float(s.get("throughput_tokens_per_second", 0.0)),
            "latency_p95_ms": float(s.get("latency_p95_ms", 0.0)),
            "latency_p99_ms": float(s.get("latency_p99_ms", 0.0)),
        }
        row.update(log_metrics)
        run_rows.append(row)

        bucket = grouped.setdefault(mode_label, {})
        for key in ("throughput_tokens_per_second", "latency_p95_ms", "latency_p99_ms"):
            bucket.setdefault(key, []).append(row[key])
        for key, value in log_metrics.items():
            bucket.setdefault(key, []).append(float(value))

    group_rows = []
    for mode_label, values in grouped.items():
        out = {"mode_label": mode_label}
        for metric_name, metric_values in values.items():
            m, s = _safe_stats(metric_values)
            out[f"{metric_name}_mean"] = m
            out[f"{metric_name}_std"] = s
        group_rows.append(out)

    run_csv = summaries_dir / "matrix_run_level_metrics.csv"
    if run_rows:
        fields = sorted(run_rows[0].keys())
        with run_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(run_rows)

    group_csv = summaries_dir / "matrix_grouped_metrics.csv"
    if group_rows:
        fields = sorted(group_rows[0].keys())
        with group_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(group_rows)

    out_json = summaries_dir / "matrix_and_mechanism_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(args.manifest),
                "run_rows": run_rows,
                "grouped_rows": group_rows,
                "run_csv": str(run_csv),
                "group_csv": str(group_csv),
            },
            f,
            indent=2,
        )

    print(f"Wrote run-level metrics to: {run_csv}")
    print(f"Wrote grouped metrics to: {group_csv}")
    print(f"Wrote JSON report to: {out_json}")


if __name__ == "__main__":
    main()
