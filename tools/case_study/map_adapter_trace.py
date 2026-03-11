#!/usr/bin/env python3
"""Map normalized tenant arrivals to deterministic LoRA adapter identities."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import iter_jsonl, load_global_config, load_seed_config, parse_cardinalities, stable_hash_int, stage_output_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map tenant arrivals to adapter identities")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--seeds", type=str, default=None, help="Path to configs/seeds.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override stage output directory")
    parser.add_argument("--raw_trace_path", type=str, default=None, help="Normalized adapter trace JSONL")
    parser.add_argument("--cardinalities", type=str, default=None, help="Comma-separated adapter cardinalities")
    return parser.parse_args()


def build_primary_app_mapping(app_counts: Counter, cardinality: int, seed: int) -> Dict[str, int]:
    if cardinality <= 0:
        raise ValueError("cardinality must be positive")

    ordered_apps = [app_id for app_id, _count in sorted(app_counts.items(), key=lambda item: (-item[1], item[0]))]
    mapping: Dict[str, int] = {}

    for slot, app_id in enumerate(ordered_apps[:cardinality]):
        mapping[app_id] = slot
    for app_id in ordered_apps[cardinality:]:
        mapping[app_id] = stable_hash_int(app_id, seed) % cardinality
    return mapping


def compute_correlated_adapter_slot(
    indep_slot: int,
    cardinality: int,
    func_id: Optional[str],
    session_id: Optional[str],
    seed: int,
) -> int:
    spread = min(max(1, cardinality // 4), 8)
    correlation_token = func_id or session_id or "none"
    offset = stable_hash_int(correlation_token, seed + cardinality) % spread
    return int((indep_slot + offset) % cardinality)


def primary_output_name(mode: str, cardinality: int, primary_cardinality: int) -> str:
    if cardinality == primary_cardinality:
        return f"adapter_trace_mapped_{mode}.jsonl"
    return f"adapter_trace_mapped_{mode}_c{cardinality:03d}.jsonl"


def summarize_counter(counter: Counter, total: int, limit: int = 20) -> List[dict]:
    rows = []
    for rank, (key, count) in enumerate(counter.most_common(limit), start=1):
        rows.append(
            {
                "rank": rank,
                "adapter_id": key,
                "count": int(count),
                "share": float(count) / total if total else 0.0,
            }
        )
    return rows


def emit_mapped_traces(
    raw_trace_path: Path,
    output_dir: Path,
    cardinalities: List[int],
    primary_cardinality: int,
    seed: int,
) -> dict:
    if not cardinalities:
        raise ValueError("cardinalities must not be empty")

    app_counts: Counter = Counter()
    for record in iter_jsonl(raw_trace_path):
        app_counts[str(record["app_id"])] += 1

    indep_mappings = {cardinality: build_primary_app_mapping(app_counts, cardinality, seed) for cardinality in cardinalities}

    indep_counts: Dict[int, Counter] = {cardinality: Counter() for cardinality in cardinalities}
    corr_counts: Dict[int, Counter] = {cardinality: Counter() for cardinality in cardinalities}
    writer_handles = []
    file_handles = []

    try:
        for cardinality in cardinalities:
            indep_path = output_dir / primary_output_name("indep", cardinality, primary_cardinality)
            corr_path = output_dir / primary_output_name("corr", cardinality, primary_cardinality)
            indep_file = indep_path.open("w", encoding="utf-8")
            corr_file = corr_path.open("w", encoding="utf-8")
            file_handles.extend([indep_file, corr_file])
            writer_handles.append((cardinality, indep_path, indep_file, corr_path, corr_file))

        total_rows = 0
        for record in iter_jsonl(raw_trace_path):
            total_rows += 1
            app_id = str(record["app_id"])
            func_id = record.get("func_id")
            session_id = record.get("session_id")

            for cardinality, _indep_path, indep_file, _corr_path, corr_file in writer_handles:
                indep_slot = indep_mappings[cardinality][app_id]
                indep_adapter_id = f"lora_{indep_slot}"
                corr_slot = compute_correlated_adapter_slot(indep_slot, cardinality, func_id, session_id, seed)
                corr_adapter_id = f"lora_{corr_slot}"

                indep_payload = {
                    "arrival_idx": int(record["arrival_idx"]),
                    "start_ts": int(record["start_ts"]),
                    "end_ts": int(record["end_ts"]),
                    "duration_ms": int(record["duration_ms"]),
                    "app_id": app_id,
                    "func_id": func_id,
                    "session_id": session_id,
                    "adapter_id": indep_adapter_id,
                    "mapping_mode": "indep",
                    "cardinality": cardinality,
                }
                corr_payload = {
                    "arrival_idx": int(record["arrival_idx"]),
                    "start_ts": int(record["start_ts"]),
                    "end_ts": int(record["end_ts"]),
                    "duration_ms": int(record["duration_ms"]),
                    "app_id": app_id,
                    "func_id": func_id,
                    "session_id": session_id,
                    "adapter_id": corr_adapter_id,
                    "mapping_mode": "corr",
                    "cardinality": cardinality,
                }

                indep_file.write(json.dumps(indep_payload, ensure_ascii=True))
                indep_file.write("\n")
                corr_file.write(json.dumps(corr_payload, ensure_ascii=True))
                corr_file.write("\n")

                indep_counts[cardinality][indep_adapter_id] += 1
                corr_counts[cardinality][corr_adapter_id] += 1

                if total_rows % 200000 == 0 and cardinality == cardinalities[0]:
                    print(f"mapped {total_rows} rows")
    finally:
        for handle in file_handles:
            handle.close()

    output_files = {
        "indep": {cardinality: str(output_dir / primary_output_name("indep", cardinality, primary_cardinality)) for cardinality in cardinalities},
        "corr": {cardinality: str(output_dir / primary_output_name("corr", cardinality, primary_cardinality)) for cardinality in cardinalities},
    }
    return {
        "row_count": total_rows,
        "unique_apps": len(app_counts),
        "cardinalities": cardinalities,
        "primary_cardinality": primary_cardinality,
        "output_files": output_files,
        "adapter_popularity": {
            "indep": {str(cardinality): summarize_counter(indep_counts[cardinality], total_rows) for cardinality in cardinalities},
            "corr": {str(cardinality): summarize_counter(corr_counts[cardinality], total_rows) for cardinality in cardinalities},
        },
    }


def write_mapping_policy(path: Path, cardinalities: List[int], primary_cardinality: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spread_examples = {cardinality: min(max(1, cardinality // 4), 8) for cardinality in cardinalities}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Tenant-to-Adapter Mapping Policy\n\n")
        handle.write("This file documents the deterministic mapping used to transform Azure tenant arrivals into adapter identities.\n\n")
        handle.write("## Independent Mode\n\n")
        handle.write("- Count requests per `app_id`.\n")
        handle.write("- Sort apps by descending count, then lexical app id.\n")
        handle.write(f"- Map the hottest `cardinality` apps to unique slots `0..cardinality-1`.\n")
        handle.write(f"- Hash the tail apps into existing slots with seed `{seed}`.\n\n")
        handle.write("## Correlated Mode\n\n")
        handle.write("- Start from the independent app slot.\n")
        handle.write("- Derive a stable offset from `func_id`, or `session_id` if `func_id` is absent.\n")
        handle.write("- Add the offset modulo the target cardinality.\n")
        handle.write("- This keeps app-level popularity while introducing function/session-conditioned hot clusters.\n\n")
        handle.write(f"Primary cardinality: `{primary_cardinality}`\n\n")
        handle.write("Per-cardinality spread:\n\n")
        for cardinality in cardinalities:
            handle.write(f"- `{cardinality}` -> func/session spread `{spread_examples[cardinality]}`\n")


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config(args.seeds)
    case_config = config.get("case_study", {})

    output_dir = stage_output_dir("adapter_trace", config, args.run_id, args.output_dir)
    raw_trace_path = Path(args.raw_trace_path or (output_dir / "adapter_trace_raw.jsonl"))
    cardinalities = parse_cardinalities(args.cardinalities or case_config.get("adapter_cardinalities", [8, 32, 128]))
    primary_cardinality = int(case_config.get("primary_adapter_cardinality", 32))
    seed = int(seeds.get("adapter_mapping_seed", seeds.get("global_seed", 7)))

    mapping_summary = emit_mapped_traces(
        raw_trace_path=raw_trace_path,
        output_dir=output_dir,
        cardinalities=cardinalities,
        primary_cardinality=primary_cardinality,
        seed=seed,
    )
    write_mapping_policy(output_dir / "mapping_policy.md", cardinalities, primary_cardinality, seed)
    write_json(output_dir / "mapping_summary.json", mapping_summary)

    print(f"wrote mapped adapter traces under {output_dir}")


if __name__ == "__main__":
    main()
