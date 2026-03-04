#!/usr/bin/env python3
"""
Parse Study2 NVTX ranges from an nsys-exported SQLite database.

Expected NVTX labels from study2_latency_crossover.py:
    Study2/N=<N>/GPU_Stream
    Study2/N=<N>/CPU_Stream
    Study2/N=<N>/Gather
    Study2/N=<N>/Transfer_ToCPU
    Study2/N=<N>/CPU_AVX_Compute
    Study2/N=<N>/Transfer_ToGPU
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


NVTX_PATTERN = re.compile(r"^Study2/N=(\d+)/(GPU_Stream|CPU_Stream|Gather|Transfer_ToCPU|CPU_AVX_Compute|Transfer_ToGPU)$")
REAL_MODEL_PATTERN = re.compile(
    r"^Study2/Layer=(\d+)/Expert=(\d+)/Step=(\d+)/N=(\d+)/"
    r"(GPU_Stream|Gather|Transfer_ToCPU|CPU_AVX_Compute|Transfer_ToGPU)"
    r"(?:/([A-Za-z0-9_]+))?$"
)


@dataclass
class Row:
    n_tokens: int
    t_gpu_stream_us: float
    t_cpu_stream_us: float
    gather_us: float
    to_cpu_us: float
    cpu_compute_us: float
    to_gpu_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Study2 NVTX timing from nsys SQLite report")
    parser.add_argument("--sqlite", type=str, required=True, help="Path to nsys-exported SQLite file")
    parser.add_argument("--output-csv", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--agg", type=str, default="median", choices=["median", "mean"])
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer filter for real-model traces. If omitted, uses all layers.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "synthetic", "real"],
        help="auto detects label format; synthetic parses Study2/N=..., real parses Study2/Layer=...",
    )
    return parser.parse_args()


def _aggregate(values: Sequence[float], mode: str) -> float:
    if not values:
        return float("nan")
    if mode == "mean":
        return float(statistics.fmean(values))
    return float(statistics.median(values))


def _fetch_nvtx_rows(conn: sqlite3.Connection) -> List[tuple[str, int, int]]:
    query = """
    SELECT
      COALESCE(NULLIF(e.text, ''), s.value) AS name,
      e.start,
      e.end
    FROM NVTX_EVENTS e
    LEFT JOIN StringIds s ON e.textId = s.id
    WHERE e.end IS NOT NULL
    """
    return [(str(name), int(start), int(end)) for name, start, end in conn.execute(query).fetchall() if name is not None]


def _build_rows(records: Sequence[tuple[str, int, int]], agg: str) -> List[Row]:
    by_n: Dict[int, Dict[str, List[float]]] = {}
    for name, start, end in records:
        match = NVTX_PATTERN.match(name)
        if not match:
            continue
        n_tokens = int(match.group(1))
        metric = match.group(2)
        duration_us = (end - start) / 1000.0  # nsys timestamps are ns
        by_n.setdefault(n_tokens, {}).setdefault(metric, []).append(duration_us)

    rows: List[Row] = []
    for n_tokens in sorted(by_n.keys()):
        metrics = by_n[n_tokens]
        rows.append(
            Row(
                n_tokens=n_tokens,
                t_gpu_stream_us=_aggregate(metrics.get("GPU_Stream", []), agg),
                t_cpu_stream_us=_aggregate(metrics.get("CPU_Stream", []), agg),
                gather_us=_aggregate(metrics.get("Gather", []), agg),
                to_cpu_us=_aggregate(metrics.get("Transfer_ToCPU", []), agg),
                cpu_compute_us=_aggregate(metrics.get("CPU_AVX_Compute", []), agg),
                to_gpu_us=_aggregate(metrics.get("Transfer_ToGPU", []), agg),
            )
        )
    return rows


def _build_rows_real_model(records: Sequence[tuple[str, int, int]], agg: str, layer: int | None) -> List[Row]:
    # key: (layer, expert, step, n_tokens) -> metric accumulators
    sample_metrics: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}

    for name, start, end in records:
        match = REAL_MODEL_PATTERN.match(name)
        if not match:
            continue

        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        step_id = int(match.group(3))
        n_tokens = int(match.group(4))
        metric = match.group(5)
        dur_us = (end - start) / 1000.0  # nsys timestamps are ns

        if layer is not None and layer_id != layer:
            continue

        key = (layer_id, expert_id, step_id, n_tokens)
        if key not in sample_metrics:
            sample_metrics[key] = {
                "gpu_stream_us": 0.0,
                "gather_us": 0.0,
                "to_cpu_us": 0.0,
                "cpu_compute_us": 0.0,
                "to_gpu_us": 0.0,
            }

        if metric == "GPU_Stream":
            sample_metrics[key]["gpu_stream_us"] += dur_us
        elif metric == "Gather":
            sample_metrics[key]["gather_us"] += dur_us
        elif metric == "Transfer_ToCPU":
            sample_metrics[key]["to_cpu_us"] += dur_us
        elif metric == "CPU_AVX_Compute":
            sample_metrics[key]["cpu_compute_us"] += dur_us
        elif metric == "Transfer_ToGPU":
            sample_metrics[key]["to_gpu_us"] += dur_us

    # group samples by N
    by_n: Dict[int, Dict[str, List[float]]] = {}
    for (_layer_id, _expert_id, _step_id, n_tokens), metrics in sample_metrics.items():
        gpu = metrics["gpu_stream_us"]
        gather = metrics["gather_us"]
        to_cpu = metrics["to_cpu_us"]
        cpu_compute = metrics["cpu_compute_us"]
        to_gpu = metrics["to_gpu_us"]
        cpu_total = gather + to_cpu + cpu_compute + to_gpu

        by_n.setdefault(n_tokens, {}).setdefault("gpu", []).append(gpu)
        by_n.setdefault(n_tokens, {}).setdefault("cpu_total", []).append(cpu_total)
        by_n.setdefault(n_tokens, {}).setdefault("gather", []).append(gather)
        by_n.setdefault(n_tokens, {}).setdefault("to_cpu", []).append(to_cpu)
        by_n.setdefault(n_tokens, {}).setdefault("cpu_compute", []).append(cpu_compute)
        by_n.setdefault(n_tokens, {}).setdefault("to_gpu", []).append(to_gpu)

    rows: List[Row] = []
    for n_tokens in sorted(by_n.keys()):
        data = by_n[n_tokens]
        rows.append(
            Row(
                n_tokens=n_tokens,
                t_gpu_stream_us=_aggregate(data.get("gpu", []), agg),
                t_cpu_stream_us=_aggregate(data.get("cpu_total", []), agg),
                gather_us=_aggregate(data.get("gather", []), agg),
                to_cpu_us=_aggregate(data.get("to_cpu", []), agg),
                cpu_compute_us=_aggregate(data.get("cpu_compute", []), agg),
                to_gpu_us=_aggregate(data.get("to_gpu", []), agg),
            )
        )
    return rows


def _find_threshold(rows: Sequence[Row]) -> int | None:
    for row in rows:
        if row.t_cpu_stream_us > row.t_gpu_stream_us:
            return row.n_tokens
    return None


def _print_rows(rows: Sequence[Row]) -> None:
    print(
        f"{'N':>6} {'T_GPU_Stream(us)':>18} {'T_CPU_Stream(us)':>18} "
        f"{'Gather(us)':>12} {'ToCPU(us)':>12} {'CPU_AVX(us)':>12} {'ToGPU(us)':>12}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r.n_tokens:>6d} "
            f"{r.t_gpu_stream_us:>18.3f} "
            f"{r.t_cpu_stream_us:>18.3f} "
            f"{r.gather_us:>12.3f} "
            f"{r.to_cpu_us:>12.3f} "
            f"{r.cpu_compute_us:>12.3f} "
            f"{r.to_gpu_us:>12.3f}"
        )


def _write_csv(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_tokens",
                "t_gpu_stream_us",
                "t_cpu_stream_us",
                "gather_us",
                "transfer_to_cpu_us",
                "cpu_avx_compute_us",
                "transfer_to_gpu_us",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "n_tokens": r.n_tokens,
                    "t_gpu_stream_us": f"{r.t_gpu_stream_us:.3f}",
                    "t_cpu_stream_us": f"{r.t_cpu_stream_us:.3f}",
                    "gather_us": f"{r.gather_us:.3f}",
                    "transfer_to_cpu_us": f"{r.to_cpu_us:.3f}",
                    "cpu_avx_compute_us": f"{r.cpu_compute_us:.3f}",
                    "transfer_to_gpu_us": f"{r.to_gpu_us:.3f}",
                }
            )


def main() -> None:
    args = parse_args()
    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        records = _fetch_nvtx_rows(conn)
    finally:
        conn.close()

    has_synthetic = any(NVTX_PATTERN.match(name) for name, _start, _end in records)
    has_real = any(REAL_MODEL_PATTERN.match(name) for name, _start, _end in records)

    mode = args.mode
    if mode == "auto":
        if has_real:
            mode = "real"
        elif has_synthetic:
            mode = "synthetic"
        else:
            mode = "synthetic"

    if mode == "real":
        rows = _build_rows_real_model(records, args.agg, args.layer)
    else:
        rows = _build_rows(records, args.agg)

    if not rows:
        raise RuntimeError(
            "No Study2 NVTX ranges found in SQLite report. "
            "Make sure Study2 labels were enabled and captured under nsys."
        )

    print(f"Detected parse mode: {mode}")
    if mode == "real":
        if args.layer is None:
            print("Layer filter: all")
        else:
            print(f"Layer filter: {args.layer}")

    _print_rows(rows)
    threshold = _find_threshold(rows)
    if threshold is None:
        print("\nN_threshold: not found in tested range.")
    else:
        print(f"\nN_threshold: {threshold} (first N where T_CPU_Stream > T_GPU_Stream)")

    if args.output_csv:
        out_path = Path(args.output_csv)
        _write_csv(rows, out_path)
        print(f"CSV written: {out_path}")


if __name__ == "__main__":
    main()
