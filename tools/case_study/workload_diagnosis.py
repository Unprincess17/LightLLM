#!/usr/bin/env python3
"""Lightweight workload diagnosis wrapper for P1 analysis outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import ensure_dir, write_csv

CONDITION_EXPERT_ONLY = "expert_only"
CONDITION_JOINT_CORR = "joint_corr"

SUMMARY_FIELDS = [
    "workload",
    "run_root",
    "summary_budget",
    "expert_only_miss_rate",
    "joint_corr_miss_rate",
    "joint_over_expert_miss_ratio",
    "expert_only_reuse_p95",
    "joint_corr_reuse_p95",
    "expert_only_hotset_coverage",
    "joint_corr_hotset_coverage",
]


@dataclass(frozen=True)
class WorkloadSummaryInput:
    workload: str
    run_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 workload diagnosis utility wrapper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="build summary.csv and required figures")
    summarize.add_argument(
        "--workloads_json",
        type=str,
        required=True,
        help="JSON file with [{\"workload\": <name>, \"run_root\": <path>}] entries",
    )
    summarize.add_argument(
        "--output_dir",
        type=str,
        default="results/workload_diagnosis",
        help="Output directory for summary/figures",
    )
    summarize.add_argument(
        "--summary_budget",
        type=int,
        default=2048,
        help="Cache budget used for summary.csv headline row values",
    )
    summarize.add_argument(
        "--topk_key",
        type=str,
        default="1000",
        help="Top-k key used from locality topk_coverage maps",
    )

    sample_alibaba = subparsers.add_parser(
        "sample-alibaba",
        help="build deterministic stratified sample from Alibaba filtered_lora_args.csv",
    )
    sample_alibaba.add_argument("--input_csv", type=str, required=True, help="input filtered_lora_args.csv")
    sample_alibaba.add_argument("--output_csv", type=str, required=True, help="output sampled csv path")
    sample_alibaba.add_argument("--sample_size", type=int, default=256, help="target sampled request count")
    sample_alibaba.add_argument("--seed", type=int, default=7, help="deterministic sampling seed")

    pressure = subparsers.add_parser(
        "generate-pressure",
        help="build pressure trace JSONL from one or more seed trace JSONL files",
    )
    pressure.add_argument(
        "--input_jsonl",
        type=str,
        nargs="+",
        required=True,
        help="one or more seed JSONL traces with adapter_id field",
    )
    pressure.add_argument("--output_jsonl", type=str, required=True, help="output pressure trace path")
    pressure.add_argument("--target_count", type=int, default=256, help="target trace length")
    pressure.add_argument("--burst_length", type=int, default=4, help="same-adapter burst length")
    pressure.add_argument("--seed", type=int, default=7, help="deterministic generation seed")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True))
            handle.write("\n")


def _condition_budget_miss_rate(cache_curve_rows: Sequence[Mapping[str, str]], condition: str, budget: int) -> float:
    for row in cache_curve_rows:
        if str(row.get("condition")) == condition and int(row.get("cache_budget", -1)) == int(budget):
            return float(row["miss_rate"])
    raise KeyError(f"missing miss rate row: condition={condition}, budget={budget}")


def stratified_sample_by_key(
    rows: Sequence[Mapping[str, object]],
    key_field: str,
    sample_size: int,
    seed: int,
) -> List[dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(rows):
        raise ValueError("sample_size cannot exceed population size")

    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        key = str(row.get(key_field))
        grouped.setdefault(key, []).append(row)

    total = len(rows)
    randomizer = random.Random(seed)
    allocations: Dict[str, int] = {}
    remainders: List[tuple[float, str]] = []

    used = 0
    for key, group in grouped.items():
        exact = sample_size * (len(group) / total)
        floor_value = int(exact)
        allocations[key] = min(floor_value, len(group))
        used += allocations[key]
        remainders.append((exact - floor_value, key))

    remaining = sample_size - used
    # Tie-break toward smaller groups to avoid collapsing minority combinations.
    remainders.sort(key=lambda item: (-item[0], len(grouped[item[1]]), item[1]))
    for _fraction, key in remainders:
        if remaining <= 0:
            break
        capacity = len(grouped[key]) - allocations[key]
        if capacity <= 0:
            continue
        allocations[key] += 1
        remaining -= 1

    if remaining > 0:
        for key in sorted(grouped):
            if remaining <= 0:
                break
            capacity = len(grouped[key]) - allocations[key]
            if capacity <= 0:
                continue
            take = min(capacity, remaining)
            allocations[key] += take
            remaining -= take

    sampled: List[dict] = []
    for key in sorted(grouped):
        group = [dict(row) for row in grouped[key]]
        randomizer.shuffle(group)
        sampled.extend(group[: allocations[key]])
    randomizer.shuffle(sampled)
    return sampled


def _build_composite_key_from_lora_args(raw_value: object) -> str:
    payload = ast.literal_eval(str(raw_value))
    if not isinstance(payload, list):
        return "none"
    components = set()
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("modelVersionId", "")).strip()
        if model_id:
            components.add(model_id)
    if not components:
        return "none"
    ordered = sorted(components)
    if len(ordered) == 1:
        return ordered[0]
    return "mix:" + "+".join(ordered)


def build_alibaba_stratified_sample(
    input_csv: Path,
    output_csv: Path,
    sample_size: int,
    seed: int,
) -> None:
    all_rows = _read_csv_rows(input_csv)
    enriched = []
    for row in all_rows:
        payload = dict(row)
        try:
            payload["composite_key"] = _build_composite_key_from_lora_args(payload.get("lora_args", "[]"))
        except Exception:
            payload["composite_key"] = "parse_error"
        enriched.append(payload)

    valid_rows = [row for row in enriched if row["composite_key"] not in ("none", "parse_error")]
    sampled = stratified_sample_by_key(
        rows=valid_rows,
        key_field="composite_key",
        sample_size=sample_size,
        seed=seed,
    )
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled:
            writer.writerow({key: row.get(key) for key in fieldnames})


def generate_pressure_trace(
    seed_rows: Sequence[Mapping[str, object]],
    target_count: int,
    burst_length: int,
    seed: int,
) -> List[dict]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if burst_length <= 0:
        raise ValueError("burst_length must be positive")
    if not seed_rows:
        raise ValueError("seed_rows must not be empty")

    randomizer = random.Random(seed)
    pools: Dict[str, List[Mapping[str, object]]] = {}
    for row in seed_rows:
        adapter_id = str(row.get("adapter_id", "unknown"))
        pools.setdefault(adapter_id, []).append(row)
    adapter_ids = sorted(pools)

    results: List[dict] = []
    cursor = 0
    while len(results) < target_count:
        adapter_id = adapter_ids[cursor % len(adapter_ids)]
        cursor += 1
        burst = min(burst_length, target_count - len(results))
        for _ in range(burst):
            source_row = randomizer.choice(pools[adapter_id])
            payload = dict(source_row)
            payload["arrival_idx"] = len(results)
            payload["pressure_source_idx"] = int(source_row.get("arrival_idx", -1))
            payload["pressure_seed_adapter"] = adapter_id
            results.append(payload)
            if len(results) >= target_count:
                break
    return results


def build_pressure_trace_from_jsonl(
    input_jsonl_paths: Sequence[Path],
    output_jsonl_path: Path,
    target_count: int,
    burst_length: int,
    seed: int,
) -> None:
    seed_rows: List[dict] = []
    for path in input_jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                record = json.loads(payload)
                if isinstance(record, dict):
                    seed_rows.append(record)
    pressure_rows = generate_pressure_trace(
        seed_rows=seed_rows,
        target_count=target_count,
        burst_length=burst_length,
        seed=seed,
    )
    _write_jsonl(output_jsonl_path, pressure_rows)


def build_summary_rows(
    workloads: Sequence[WorkloadSummaryInput],
    summary_budget: int,
    topk_key: str,
) -> List[dict]:
    rows: List[dict] = []
    for workload in workloads:
        run_root = Path(workload.run_root)
        locality_dir = run_root / "replay" / "locality"
        cache_dir = run_root / "replay" / "cache"

        cache_curve = _read_csv_rows(cache_dir / "cache_curve.csv")
        joint_corr_locality = _load_json(locality_dir / "joint_corr_locality.json")
        expert_only_locality = _load_json(locality_dir / "expert_only_locality.json")

        expert_only_miss = _condition_budget_miss_rate(cache_curve, CONDITION_EXPERT_ONLY, summary_budget)
        joint_corr_miss = _condition_budget_miss_rate(cache_curve, CONDITION_JOINT_CORR, summary_budget)
        miss_ratio = joint_corr_miss / max(expert_only_miss, 1e-12)

        row = {
            "workload": workload.workload,
            "run_root": str(run_root),
            "summary_budget": int(summary_budget),
            "expert_only_miss_rate": expert_only_miss,
            "joint_corr_miss_rate": joint_corr_miss,
            "joint_over_expert_miss_ratio": miss_ratio,
            "expert_only_reuse_p95": float(expert_only_locality["reuse_distance"]["finite_p95"]),
            "joint_corr_reuse_p95": float(joint_corr_locality["reuse_distance"]["finite_p95"]),
            "expert_only_hotset_coverage": float(expert_only_locality["topk_coverage"][str(topk_key)]),
            "joint_corr_hotset_coverage": float(joint_corr_locality["topk_coverage"][str(topk_key)]),
        }
        rows.append(row)
    return rows


def _plot_miss_vs_cache(workloads: Sequence[WorkloadSummaryInput], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(workloads), figsize=(5.2 * max(1, len(workloads)), 3.6), squeeze=False)
    for index, workload in enumerate(workloads):
        axis = axes[0][index]
        rows = _read_csv_rows(Path(workload.run_root) / "replay" / "cache" / "cache_curve.csv")
        for condition, color in ((CONDITION_EXPERT_ONLY, "#355070"), (CONDITION_JOINT_CORR, "#C8553D")):
            subset = [row for row in rows if str(row["condition"]) == condition]
            x_values = [int(row["cache_budget"]) for row in subset]
            y_values = [float(row["miss_rate"]) for row in subset]
            axis.plot(x_values, y_values, label=condition, color=color)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(workload.workload)
        axis.set_xlabel("cache objects")
        axis.set_ylabel("miss rate")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_locality_cdf(workloads: Sequence[WorkloadSummaryInput], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(workloads), figsize=(5.2 * max(1, len(workloads)), 3.6), squeeze=False)
    for index, workload in enumerate(workloads):
        axis = axes[0][index]
        rows = _read_csv_rows(Path(workload.run_root) / "replay" / "locality" / "popularity_rank.csv")
        for condition, color in ((CONDITION_EXPERT_ONLY, "#355070"), (CONDITION_JOINT_CORR, "#C8553D")):
            subset = [row for row in rows if str(row["condition"]) == condition]
            x_values = [int(row["rank"]) for row in subset]
            y_values = [float(row["cumulative_fraction"]) for row in subset]
            axis.plot(x_values, y_values, label=condition, color=color)
        axis.set_xscale("log", base=10)
        axis.set_title(workload.workload)
        axis.set_xlabel("sorted object rank")
        axis.set_ylabel("cumulative access fraction")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_reuse_distance(workloads: Sequence[WorkloadSummaryInput], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(workloads), figsize=(5.2 * max(1, len(workloads)), 3.6), squeeze=False)
    for index, workload in enumerate(workloads):
        axis = axes[0][index]
        rows = _read_csv_rows(Path(workload.run_root) / "replay" / "locality" / "reuse_distance_cdf.csv")
        for condition, color in ((CONDITION_EXPERT_ONLY, "#355070"), (CONDITION_JOINT_CORR, "#C8553D")):
            subset = [
                row
                for row in rows
                if str(row["condition"]) == condition and int(row["reuse_distance"]) >= 0
            ]
            x_values = [int(row["reuse_distance"]) for row in subset]
            y_values = [float(row["cdf"]) for row in subset]
            axis.plot(x_values, y_values, label=condition, color=color)
        axis.set_xscale("log", base=10)
        axis.set_title(workload.workload)
        axis.set_xlabel("reuse distance (decode steps)")
        axis.set_ylabel("CDF")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_workloads_manifest(path: Path) -> List[WorkloadSummaryInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--workloads_json must be a JSON list")
    workloads: List[WorkloadSummaryInput] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("workloads_json entry must be object")
        workloads.append(
            WorkloadSummaryInput(
                workload=str(entry["workload"]),
                run_root=Path(str(entry["run_root"])),
            )
        )
    if not workloads:
        raise ValueError("workloads_json must not be empty")
    return workloads


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        output_dir = ensure_dir(Path(args.output_dir))
        workloads = _load_workloads_manifest(Path(args.workloads_json))
        summary_rows = build_summary_rows(
            workloads=workloads,
            summary_budget=int(args.summary_budget),
            topk_key=str(args.topk_key),
        )

        write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
        _plot_locality_cdf(workloads, output_dir / "fig_locality_cdf.pdf")
        _plot_miss_vs_cache(workloads, output_dir / "fig_miss_vs_cache.pdf")
        _plot_reuse_distance(workloads, output_dir / "fig_reuse_distance.pdf")
        print(f"wrote workload diagnosis artifacts under {output_dir}")
        return

    if args.command == "sample-alibaba":
        build_alibaba_stratified_sample(
            input_csv=Path(args.input_csv),
            output_csv=Path(args.output_csv),
            sample_size=int(args.sample_size),
            seed=int(args.seed),
        )
        print(f"wrote sampled Alibaba trace: {args.output_csv}")
        return

    if args.command == "generate-pressure":
        build_pressure_trace_from_jsonl(
            input_jsonl_paths=[Path(path) for path in args.input_jsonl],
            output_jsonl_path=Path(args.output_jsonl),
            target_count=int(args.target_count),
            burst_length=int(args.burst_length),
            seed=int(args.seed),
        )
        print(f"wrote pressure trace: {args.output_jsonl}")
        return

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
