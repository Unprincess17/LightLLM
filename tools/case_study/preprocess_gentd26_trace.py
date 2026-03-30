#!/usr/bin/env python3
"""Build request-level composite LoRA adapter traces from the GenTD26 dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    load_global_config,
    load_seed_config,
    numeric_summary,
    stable_hash_int,
    stage_output_dir,
    top_counter_rows,
    write_csv,
    write_json,
)


DEFAULT_GENTD26_TRACE_PATH = Path(
    "/home/shufan/alibaba-clusterdata/cluster-trace-v2026-GenAI/filtered_lora_args.csv"
)
RAW_TRACE_SOURCE = "gentd26_filtered_lora_args"
TOP_REPORT_LIMIT = 50
DEFAULT_CORR_JITTER_WINDOW = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GenTD26 LoRA requests into mapped adapter traces")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--seeds", type=str, default=None, help="Path to configs/seeds.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override stage output directory")
    parser.add_argument("--trace_path", type=str, default=None, help="GenTD26 filtered_lora_args.csv path")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap for preprocessing")
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Deterministic seed for the request-level indep permutation",
    )
    parser.add_argument(
        "--corr_jitter_window",
        type=int,
        default=DEFAULT_CORR_JITTER_WINDOW,
        help=(
            "Maximum deterministic request displacement applied to the correlated schedule. "
            "Set to 0 to preserve exact file order."
        ),
    )
    return parser.parse_args()


def normalize_optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_optional_float(value: object) -> Optional[float]:
    text = normalize_optional_text(value)
    if text is None:
        return None
    return float(text)


def parse_lora_args(raw_value: object) -> List[str]:
    text = normalize_optional_text(raw_value)
    if text is None:
        return []
    payload = ast.literal_eval(text)
    if not isinstance(payload, list):
        raise ValueError("lora_args must parse into a list")

    component_ids = set()
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        model_version_id = normalize_optional_text(item.get("modelVersionId"))
        if model_version_id is not None:
            component_ids.add(model_version_id)
    return sorted(component_ids)


def build_composite_key(component_lora_ids: Sequence[str]) -> str:
    if not component_lora_ids:
        raise ValueError("component_lora_ids must not be empty")
    if len(component_lora_ids) == 1:
        return str(component_lora_ids[0])
    return "mix:" + "+".join(str(component_id) for component_id in component_lora_ids)


def output_name(mode: str, cardinality: int, primary: bool) -> str:
    if primary:
        return f"adapter_trace_mapped_{mode}.jsonl"
    return f"adapter_trace_mapped_{mode}_c{cardinality:03d}.jsonl"


def summarize_adapter_counter(counter: Counter, total: int, limit: int = 20) -> List[dict]:
    rows = []
    for rank, (adapter_id, count) in enumerate(counter.most_common(limit), start=1):
        rows.append(
            {
                "rank": rank,
                "adapter_id": adapter_id,
                "count": int(count),
                "share": float(count) / float(total) if total else 0.0,
            }
        )
    return rows


def stable_permutation(length: int, seed: int) -> List[int]:
    return sorted(range(length), key=lambda index: (stable_hash_int(f"perm:{index}", seed), index))


def stable_bounded_jitter_permutation(length: int, seed: int, jitter_window: int) -> List[int]:
    if length <= 1 or jitter_window <= 0:
        return list(range(length))
    window = int(jitter_window)
    span = (2 * window) + 1
    return sorted(
        range(length),
        key=lambda index: (
            index + int(stable_hash_int(f"corr_jitter:{index}", seed) % span) - window,
            index,
        ),
    )


def write_jsonl_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True))
            handle.write("\n")


def write_mapping_policy(path: Path, cardinality: int, shuffle_seed: int, corr_jitter_window: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# GenTD26 Composite-Adapter Mapping Policy\n\n")
        handle.write(
            "This file documents the deterministic request-level LoRA mapping used for the motivation trace.\n\n"
        )
        handle.write("## Raw Request Normalization\n\n")
        handle.write("- Treat each `filtered_lora_args.csv` row as one LoRA-bearing request in file order.\n")
        handle.write("- Parse `lora_args` as a Python literal list.\n")
        handle.write("- Ignore `scale` values.\n")
        handle.write("- Extract `modelVersionId`, drop duplicates within the request, and sort them lexically.\n")
        handle.write("- Map each unique sorted component set to one composite adapter identity.\n\n")
        handle.write("## Adapter Slot Assignment\n\n")
        handle.write("- Count requests per composite adapter identity.\n")
        handle.write("- Sort composites by descending request count, then lexical composite key.\n")
        handle.write("- Assign one unique adapter slot per composite identity.\n")
        handle.write(f"- Primary cardinality is the full observed composite count: `{cardinality}`.\n\n")
        handle.write("## Corr / Indep Semantics\n\n")
        if corr_jitter_window > 0:
            handle.write(
                "- `corr`: preserve local file-order correlation, but apply a deterministic bounded jitter "
                f"of `+/- {corr_jitter_window}` requests to weaken extreme burstiness.\n"
            )
            handle.write("- Use `--corr_jitter_window 0` to recover exact file order.\n")
        else:
            handle.write("- `corr`: preserve the exact real file-order composite-adapter request sequence.\n")
        handle.write("- `indep`: apply a deterministic request-level permutation of the same adapter multiset.\n")
        handle.write(f"- Permutation seed: `{shuffle_seed}`.\n\n")
        handle.write("## Time Semantics\n\n")
        handle.write("- GenTD26 `filtered_lora_args.csv` does not provide request timestamps.\n")
        handle.write("- Downstream B9 therefore replays this trace using `arrival_idx` order only.\n")


def build_raw_summary(
    trace_path: Path,
    row_count: int,
    parse_errors: int,
    empty_lora_rows: int,
    base_lora_counts: Counter,
    composite_counts: Counter,
    component_count_hist: Counter,
    duration_ms_values: Sequence[float],
) -> dict:
    max_component_count = max(component_count_hist) if component_count_hist else 0
    return {
        "trace_path": str(trace_path),
        "trace_source": RAW_TRACE_SOURCE,
        "row_count": row_count,
        "parse_errors": parse_errors,
        "empty_lora_rows": empty_lora_rows,
        "unique_base_loras": len(base_lora_counts),
        "unique_composite_adapters": len(composite_counts),
        "single_lora_request_count": int(component_count_hist.get(1, 0)),
        "multi_lora_request_count": int(sum(count for width, count in component_count_hist.items() if width > 1)),
        "max_component_count": max_component_count,
        "duration_ms": numeric_summary(duration_ms_values),
        "component_count_hist": [
            {
                "component_count": int(component_count),
                "request_count": int(request_count),
                "share": float(request_count) / float(row_count) if row_count else 0.0,
            }
            for component_count, request_count in sorted(component_count_hist.items())
        ],
        "top_base_loras": top_counter_rows(
            base_lora_counts,
            "lora_id",
            total=sum(base_lora_counts.values()),
            limit=min(TOP_REPORT_LIMIT, len(base_lora_counts)),
        ),
        "top_composite_adapters": top_counter_rows(
            composite_counts,
            "app_id",
            total=row_count,
            limit=min(TOP_REPORT_LIMIT, len(composite_counts)),
        ),
        "arrival_order_note": "arrival_idx preserves file order because filtered_lora_args.csv does not expose request timestamps",
    }


def build_manifest_rows(
    ordered_composites: Sequence[str],
    composite_counts: Counter,
    components_by_key: Mapping[str, Sequence[str]],
    total_rows: int,
) -> List[dict]:
    rows: List[dict] = []
    for slot, composite_key in enumerate(ordered_composites):
        component_lora_ids = list(components_by_key[composite_key])
        request_count = int(composite_counts[composite_key])
        rows.append(
            {
                "adapter_slot": slot,
                "adapter_id": f"lora_{slot}",
                "composite_key": composite_key,
                "component_lora_ids": component_lora_ids,
                "component_count": len(component_lora_ids),
                "is_composite": len(component_lora_ids) > 1,
                "request_count": request_count,
                "request_share": float(request_count) / float(total_rows) if total_rows else 0.0,
            }
        )
    return rows


def build_mapped_payload(
    record: Mapping[str, object],
    arrival_idx: int,
    adapter_slot: int,
    mapping_mode: str,
    cardinality: int,
    source_arrival_idx: int,
    shuffle_seed: Optional[int],
) -> dict:
    payload = {
        "arrival_idx": int(arrival_idx),
        "adapter_id": f"lora_{adapter_slot}",
        "mapping_mode": mapping_mode,
        "cardinality": int(cardinality),
        "app_id": str(record["app_id"]),
        "component_lora_ids": list(record["component_lora_ids"]),
        "component_count": int(record["component_count"]),
        "duration_ms": int(record["duration_ms"]) if record.get("duration_ms") is not None else None,
        "source_arrival_idx": int(source_arrival_idx),
        "trace_source": RAW_TRACE_SOURCE,
    }
    if record.get("predict_type") is not None:
        payload["predict_type"] = str(record["predict_type"])
    if record.get("checkpoint_model_version_id") is not None:
        payload["checkpoint_model_version_id"] = str(record["checkpoint_model_version_id"])
    if shuffle_seed is not None:
        payload["shuffle_seed"] = int(shuffle_seed)
    return payload


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config(args.seeds)
    output_dir = stage_output_dir("adapter_trace", config, args.run_id, args.output_dir)

    paths_config = config.get("paths", {})
    trace_path = Path(args.trace_path or paths_config.get("gentd26_lora_trace", DEFAULT_GENTD26_TRACE_PATH))
    shuffle_seed = int(args.shuffle_seed or seeds.get("adapter_mapping_seed", seeds.get("global_seed", 7)))
    corr_jitter_window = max(int(args.corr_jitter_window), 0)

    raw_records: List[dict] = []
    base_lora_counts: Counter = Counter()
    composite_counts: Counter = Counter()
    component_count_hist: Counter = Counter()
    components_by_key: Dict[str, List[str]] = {}
    duration_ms_values: List[float] = []

    parse_errors = 0
    empty_lora_rows = 0

    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.max_rows is not None and len(raw_records) >= args.max_rows:
                break

            try:
                component_lora_ids = parse_lora_args(row.get("lora_args"))
            except Exception:
                parse_errors += 1
                continue

            if not component_lora_ids:
                empty_lora_rows += 1
                continue

            composite_key = build_composite_key(component_lora_ids)
            duration_seconds = parse_optional_float(row.get("exec_time_seconds"))
            duration_ms = None if duration_seconds is None else int(round(duration_seconds * 1000.0))

            record = {
                "arrival_idx": len(raw_records),
                "app_id": composite_key,
                "component_lora_ids": component_lora_ids,
                "component_count": len(component_lora_ids),
                "duration_ms": duration_ms,
                "predict_type": normalize_optional_text(row.get("predict_type")),
                "checkpoint_model_version_id": normalize_optional_text(row.get("checkpoint_model_version_id")),
                "raw_trace_source": trace_path.name,
            }
            raw_records.append(record)

            base_lora_counts.update(component_lora_ids)
            composite_counts[composite_key] += 1
            component_count_hist[len(component_lora_ids)] += 1
            components_by_key.setdefault(composite_key, list(component_lora_ids))
            if duration_ms is not None:
                duration_ms_values.append(float(duration_ms))

            if len(raw_records) % 5000 == 0:
                print(f"processed {len(raw_records)} LoRA-bearing requests")

    if not raw_records:
        raise ValueError(f"no non-empty LoRA requests parsed from {trace_path}")

    ordered_composites = [
        composite_key
        for composite_key, _count in sorted(composite_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    slot_by_composite = {composite_key: slot for slot, composite_key in enumerate(ordered_composites)}
    cardinality = len(slot_by_composite)

    raw_output_path = output_dir / "adapter_trace_raw.jsonl"
    summary_path = output_dir / "adapter_trace_summary.json"
    tenant_popularity_path = output_dir / "tenant_popularity.csv"
    manifest_path = output_dir / "composite_adapter_manifest.json"
    mapping_policy_path = output_dir / "mapping_policy.md"
    mapping_summary_path = output_dir / "mapping_summary.json"
    corr_primary_path = output_dir / output_name("corr", cardinality, primary=True)
    indep_primary_path = output_dir / output_name("indep", cardinality, primary=True)
    corr_cardinality_path = output_dir / output_name("corr", cardinality, primary=False)
    indep_cardinality_path = output_dir / output_name("indep", cardinality, primary=False)

    write_jsonl_rows(raw_output_path, raw_records)
    write_json(
        summary_path,
        build_raw_summary(
            trace_path=trace_path,
            row_count=len(raw_records),
            parse_errors=parse_errors,
            empty_lora_rows=empty_lora_rows,
            base_lora_counts=base_lora_counts,
            composite_counts=composite_counts,
            component_count_hist=component_count_hist,
            duration_ms_values=duration_ms_values,
        ),
    )
    write_csv(tenant_popularity_path, ["rank", "app_id", "count", "share"], top_counter_rows(composite_counts, "app_id", total=len(raw_records), limit=len(composite_counts)))
    write_json(
        manifest_path,
        {
            "trace_path": str(trace_path),
            "trace_source": RAW_TRACE_SOURCE,
            "primary_cardinality": cardinality,
            "manifest_rows": build_manifest_rows(
                ordered_composites=ordered_composites,
                composite_counts=composite_counts,
                components_by_key=components_by_key,
                total_rows=len(raw_records),
            ),
        },
    )

    adapter_counts = Counter()
    for composite_key, request_count in composite_counts.items():
        adapter_counts[f"lora_{slot_by_composite[composite_key]}"] = int(request_count)

    corr_permutation = stable_bounded_jitter_permutation(len(raw_records), shuffle_seed, corr_jitter_window)
    corr_rows = (
        build_mapped_payload(
            record=raw_records[source_arrival_idx],
            arrival_idx=arrival_idx,
            adapter_slot=slot_by_composite[str(raw_records[source_arrival_idx]["app_id"])],
            mapping_mode="corr",
            cardinality=cardinality,
            source_arrival_idx=source_arrival_idx,
            shuffle_seed=None,
        )
        for arrival_idx, source_arrival_idx in enumerate(corr_permutation)
    )
    permutation = stable_permutation(len(raw_records), shuffle_seed)
    indep_rows = (
        build_mapped_payload(
            record=raw_records[source_arrival_idx],
            arrival_idx=arrival_idx,
            adapter_slot=slot_by_composite[str(raw_records[source_arrival_idx]["app_id"])],
            mapping_mode="indep",
            cardinality=cardinality,
            source_arrival_idx=source_arrival_idx,
            shuffle_seed=shuffle_seed,
        )
        for arrival_idx, source_arrival_idx in enumerate(permutation)
    )

    corr_rows_materialized = list(corr_rows)
    indep_rows_materialized = list(indep_rows)
    write_jsonl_rows(corr_primary_path, corr_rows_materialized)
    write_jsonl_rows(corr_cardinality_path, corr_rows_materialized)
    write_jsonl_rows(indep_primary_path, indep_rows_materialized)
    write_jsonl_rows(indep_cardinality_path, indep_rows_materialized)

    write_mapping_policy(
        mapping_policy_path,
        cardinality=cardinality,
        shuffle_seed=shuffle_seed,
        corr_jitter_window=corr_jitter_window,
    )
    write_json(
        mapping_summary_path,
        {
            "trace_path": str(trace_path),
            "trace_source": RAW_TRACE_SOURCE,
            "row_count": len(raw_records),
            "unique_base_loras": len(base_lora_counts),
            "unique_composite_adapters": cardinality,
            "primary_cardinality": cardinality,
            "cardinalities": [cardinality],
            "multi_lora_request_count": int(sum(count for width, count in component_count_hist.items() if width > 1)),
            "sequence_semantics": {
                "corr": (
                    "bounded-jitter local-order composite-adapter sequence derived from file order"
                    if corr_jitter_window > 0
                    else "real request-level composite-adapter sequence in exact file order"
                ),
                "indep": "deterministic permutation of the same request-level composite-adapter multiset",
                "time": "order-only arrival_idx semantics; original request timestamps are unavailable",
            },
            "corr_order_policy": {
                "algorithm": "stable_bounded_jitter" if corr_jitter_window > 0 else "identity",
                "jitter_window": corr_jitter_window,
                "seed": shuffle_seed,
            },
            "permutation": {
                "algorithm": "stable_hash_sort",
                "seed": shuffle_seed,
            },
            "output_files": {
                "corr": {
                    "primary": str(corr_primary_path),
                    str(cardinality): str(corr_cardinality_path),
                },
                "indep": {
                    "primary": str(indep_primary_path),
                    str(cardinality): str(indep_cardinality_path),
                },
            },
            "adapter_popularity": {
                "corr": {str(cardinality): summarize_adapter_counter(adapter_counts, len(raw_records))},
                "indep": {str(cardinality): summarize_adapter_counter(adapter_counts, len(raw_records))},
            },
            "composite_manifest_path": str(manifest_path),
        },
    )

    print(
        "wrote GenTD26 composite adapter traces under "
        f"{output_dir} (rows={len(raw_records)}, unique_composites={cardinality})"
    )


if __name__ == "__main__":
    main()
