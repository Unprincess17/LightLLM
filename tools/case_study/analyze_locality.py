#!/usr/bin/env python3
"""Compute B7 locality metrics for expert-only and expert x LoRA replay traces."""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import orjson
from numba import njit

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    ensure_dir,
    load_global_config,
    numeric_summary,
    stage_output_dir,
    write_csv,
    write_json,
)


CONDITION_EXPERT_ONLY = "expert_only"
CONDITION_JOINT_INDEP = "joint_indep"
CONDITION_JOINT_CORR = "joint_corr"
CONDITION_ORDER = (
    CONDITION_EXPERT_ONLY,
    CONDITION_JOINT_INDEP,
    CONDITION_JOINT_CORR,
)

EXPERT_ID_STRIDE = 256
ADAPTER_SLOT_STRIDE = 256
COLD_REUSE_DISTANCE = -1
PROGRESS_EVERY_ROWS = 1_000_000
TOPK_SUMMARY_POINTS = (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)

POPULARITY_CSV_FIELDS = [
    "condition",
    "rank",
    "object_key",
    "count",
    "fraction",
    "cumulative_fraction",
]
REUSE_CDF_CSV_FIELDS = ["condition", "reuse_distance", "count", "fraction", "cdf"]
TOPK_COVERAGE_CSV_FIELDS = ["condition", "k", "coverage"]

OBJECT_KEY_DEFINITIONS = {
    CONDITION_EXPERT_ONLY: "(layer_id, expert_id)",
    CONDITION_JOINT_INDEP: "(layer_id, expert_id, adapter_id)",
    CONDITION_JOINT_CORR: "(layer_id, expert_id, adapter_id)",
}

OBJECT_KEY_NOTES = {
    CONDITION_EXPERT_ONLY: "Derived from the canonical B6 joined stream by dropping adapter_id, preserving the exact B6 event order.",
    CONDITION_JOINT_INDEP: "One access object per exploded joined-trace row using the independent adapter assignment.",
    CONDITION_JOINT_CORR: "One access object per exploded joined-trace row using the correlated adapter assignment.",
}


@dataclass
class ConditionAccessBuffer:
    condition: str
    access_ids: np.ndarray
    total_events: int


@dataclass
class StreamBuildResult:
    condition_buffers: Dict[str, ConditionAccessBuffer]
    per_request_unique_counts: Dict[str, List[int]]
    request_count: int
    total_events: int
    position_field_counts: Dict[str, int]
    invariant_checked_fields: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute locality metrics for B7 offline replay study")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for replay/locality artifacts",
    )
    parser.add_argument(
        "--joined_indep_path",
        type=str,
        default=None,
        help="Override joined_trace_indep.jsonl input path",
    )
    parser.add_argument(
        "--joined_corr_path",
        type=str,
        default=None,
        help="Override joined_trace_corr.jsonl input path",
    )
    parser.add_argument(
        "--qc_report_path",
        type=str,
        default=None,
        help="Override join_qc_report.json input path",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Reserved scratch directory argument; current implementation keeps access streams in memory",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=PROGRESS_EVERY_ROWS,
        help="Print progress after every N paired joined-trace rows",
    )
    parser.add_argument(
        "--keep_work_files",
        action="store_true",
        help="Retained for CLI compatibility; current implementation does not emit temporary work files",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def iter_jsonl_bytes(path: Path) -> Iterator[dict]:
    with path.open("rb") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = orjson.loads(raw_line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num} invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_num} is not a JSON object")
            yield record


def extract_position(record: Mapping[str, object]) -> Tuple[str, int]:
    if record.get("token_pos") is not None:
        return "token_pos", int(record["token_pos"])
    if record.get("chunk_idx") is not None:
        return "chunk_idx", int(record["chunk_idx"])
    raise ValueError("joined trace row must contain token_pos or chunk_idx")


def parse_adapter_slot(adapter_id: object) -> int:
    if adapter_id is None:
        raise ValueError("adapter_id is required for joint locality analysis")
    adapter_token = str(adapter_id)
    if not adapter_token.startswith("lora_"):
        raise ValueError(f"unsupported adapter_id format: {adapter_token}")
    slot = int(adapter_token.split("_", 1)[1])
    if slot < 0 or slot >= ADAPTER_SLOT_STRIDE:
        raise ValueError(
            f"adapter slot {slot} exceeds configured stride {ADAPTER_SLOT_STRIDE}; "
            "increase ADAPTER_SLOT_STRIDE in analyze_locality.py"
        )
    return slot


def pack_expert_object(layer_id: int, expert_id: int) -> int:
    if layer_id < 0:
        raise ValueError(f"layer_id must be non-negative, got {layer_id}")
    if expert_id < 0 or expert_id >= EXPERT_ID_STRIDE:
        raise ValueError(
            f"expert_id {expert_id} exceeds configured stride {EXPERT_ID_STRIDE}; "
            "increase EXPERT_ID_STRIDE in analyze_locality.py"
        )
    return layer_id * EXPERT_ID_STRIDE + expert_id + 1


def pack_joint_object(layer_id: int, expert_id: int, adapter_slot: int) -> int:
    return pack_expert_object(layer_id, expert_id) * ADAPTER_SLOT_STRIDE + adapter_slot


def decode_expert_object_key(object_id: int) -> str:
    base = int(object_id) - 1
    layer_id = base // EXPERT_ID_STRIDE
    expert_id = base % EXPERT_ID_STRIDE
    return f"layer={layer_id}|expert={expert_id}"


def decode_joint_object_key(object_id: int) -> str:
    base = int(object_id)
    expert_object = base // ADAPTER_SLOT_STRIDE
    adapter_slot = base % ADAPTER_SLOT_STRIDE
    expert_base = expert_object - 1
    layer_id = expert_base // EXPERT_ID_STRIDE
    expert_id = expert_base % EXPERT_ID_STRIDE
    return f"layer={layer_id}|expert={expert_id}|adapter=lora_{adapter_slot}"


def decode_object_key(condition: str, object_id: int) -> str:
    if condition == CONDITION_EXPERT_ONLY:
        return decode_expert_object_key(object_id)
    return decode_joint_object_key(object_id)


def resolve_total_events(qc_report: Mapping[str, object]) -> int:
    checks = qc_report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("join_qc_report.json is missing checks")
    per_mode_row_counts = checks.get("per_mode_row_counts")
    if not isinstance(per_mode_row_counts, Mapping):
        raise ValueError("join_qc_report.json is missing checks.per_mode_row_counts")
    indep_count = int(per_mode_row_counts.get("indep", 0))
    corr_count = int(per_mode_row_counts.get("corr", 0))
    if indep_count <= 0 or corr_count <= 0:
        raise ValueError("join_qc_report.json reported non-positive joined row counts")
    if indep_count != corr_count:
        raise ValueError(
            "joined indep/corr traces must have the same row count for aligned B7 analysis: "
            f"indep={indep_count}, corr={corr_count}"
        )
    return indep_count


def paired_row_alignment_fields(
    indep_row: Mapping[str, object],
    corr_row: Mapping[str, object],
) -> Tuple[List[str], str]:
    position_field_indep, position_value_indep = extract_position(indep_row)
    position_field_corr, position_value_corr = extract_position(corr_row)
    if position_field_indep != position_field_corr or position_value_indep != position_value_corr:
        raise ValueError(
            "joined indep/corr traces diverged on token/chunk position: "
            f"indep=({position_field_indep}={position_value_indep}) "
            f"corr=({position_field_corr}={position_value_corr})"
        )

    shared_fields = [
        "arrival_idx",
        "req_idx",
        "event_idx",
        "layer_id",
        "phase",
        "expert_id",
        position_field_indep,
    ]
    for field in shared_fields:
        if indep_row.get(field) != corr_row.get(field):
            raise ValueError(
                "joined indep/corr traces are not event-aligned: "
                f"field={field}, indep={indep_row.get(field)!r}, corr={corr_row.get(field)!r}"
            )
    return shared_fields, position_field_indep


def build_access_streams(
    joined_indep_path: Path,
    joined_corr_path: Path,
    total_events: int,
    work_dir: Path,
    progress_every: int,
) -> StreamBuildResult:
    _ = work_dir
    access_buffers = {
        CONDITION_EXPERT_ONLY: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_INDEP: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_CORR: np.empty(total_events, dtype=np.uint32),
    }

    per_request_unique_counts = {condition: [] for condition in CONDITION_ORDER}
    request_sets = {condition: set() for condition in CONDITION_ORDER}
    request_count = 0
    current_req_idx: Optional[int] = None
    current_position_field: Optional[str] = None
    position_field_counts: Dict[str, int] = {}
    invariant_checked_fields: List[str] = []

    indep_iter = iter_jsonl_bytes(joined_indep_path)
    corr_iter = iter_jsonl_bytes(joined_corr_path)

    rows_seen = 0
    for rows_seen, pair in enumerate(zip_longest(indep_iter, corr_iter), start=1):
        indep_row, corr_row = pair
        if indep_row is None or corr_row is None:
            raise ValueError("joined indep/corr traces have different row counts")

        shared_fields, position_field = paired_row_alignment_fields(indep_row, corr_row)
        if not invariant_checked_fields:
            invariant_checked_fields = list(shared_fields)
        if current_position_field is None:
            current_position_field = position_field
        position_field_counts[position_field] = position_field_counts.get(position_field, 0) + 1

        req_idx = int(indep_row["req_idx"])
        if current_req_idx is None:
            current_req_idx = req_idx
        elif req_idx < current_req_idx:
            raise ValueError(
                "joined trace request order regressed; B7 requires canonical non-decreasing req_idx order: "
                f"previous={current_req_idx}, current={req_idx}"
            )
        elif req_idx > current_req_idx:
            if req_idx != current_req_idx + 1:
                raise ValueError(
                    "joined trace req_idx order is not contiguous; B7 requires the same canonical request stream "
                    f"across conditions: previous={current_req_idx}, current={req_idx}"
                )
            for condition in CONDITION_ORDER:
                per_request_unique_counts[condition].append(len(request_sets[condition]))
                request_sets[condition].clear()
            request_count += 1
            current_req_idx = req_idx

        layer_id = int(indep_row["layer_id"])
        expert_id = int(indep_row["expert_id"])
        expert_object_id = pack_expert_object(layer_id, expert_id)
        indep_object_id = pack_joint_object(layer_id, expert_id, parse_adapter_slot(indep_row["adapter_id"]))
        corr_object_id = pack_joint_object(layer_id, expert_id, parse_adapter_slot(corr_row["adapter_id"]))

        buffer_index = rows_seen - 1
        access_buffers[CONDITION_EXPERT_ONLY][buffer_index] = np.uint32(expert_object_id)
        access_buffers[CONDITION_JOINT_INDEP][buffer_index] = np.uint32(indep_object_id)
        access_buffers[CONDITION_JOINT_CORR][buffer_index] = np.uint32(corr_object_id)

        request_sets[CONDITION_EXPERT_ONLY].add(expert_object_id)
        request_sets[CONDITION_JOINT_INDEP].add(indep_object_id)
        request_sets[CONDITION_JOINT_CORR].add(corr_object_id)

        if progress_every > 0 and rows_seen % progress_every == 0:
            print(f"materialized {rows_seen}/{total_events} aligned access objects")

    if rows_seen != total_events:
        raise ValueError(
            f"joined trace row count mismatch against join_qc_report.json: expected {total_events}, observed {rows_seen}"
        )

    if current_req_idx is not None:
        for condition in CONDITION_ORDER:
            per_request_unique_counts[condition].append(len(request_sets[condition]))
            request_sets[condition].clear()
        request_count += 1

    condition_buffers = {}
    for condition in CONDITION_ORDER:
        condition_buffers[condition] = ConditionAccessBuffer(
            condition=condition,
            access_ids=access_buffers[condition],
            total_events=total_events,
        )
    gc.collect()

    return StreamBuildResult(
        condition_buffers=condition_buffers,
        per_request_unique_counts=per_request_unique_counts,
        request_count=request_count,
        total_events=total_events,
        position_field_counts=position_field_counts,
        invariant_checked_fields=invariant_checked_fields,
    )


@njit(cache=False)
def fenwick_add(tree: np.ndarray, index: int, delta: int) -> None:
    size = tree.shape[0]
    while index < size:
        tree[index] += delta
        index += index & -index


@njit(cache=False)
def fenwick_prefix_sum(tree: np.ndarray, index: int) -> int:
    result = 0
    while index > 0:
        result += tree[index]
        index -= index & -index
    return result


@njit(cache=False)
def compute_reuse_histogram(access_ids: np.ndarray, id_space_size: int, max_distance: int) -> Tuple[int, np.ndarray]:
    total_events = access_ids.shape[0]
    tree = np.zeros(total_events + 1, dtype=np.int32)
    last_seen = np.zeros(id_space_size + 1, dtype=np.int64)
    histogram = np.zeros(max_distance + 1, dtype=np.int64)
    cold_count = 0

    for position in range(1, total_events + 1):
        object_id = int(access_ids[position - 1])
        previous_position = last_seen[object_id]
        if previous_position == 0:
            cold_count += 1
        else:
            distance = fenwick_prefix_sum(tree, position - 1) - fenwick_prefix_sum(tree, previous_position)
            histogram[distance] += 1
            fenwick_add(tree, previous_position, -1)
        fenwick_add(tree, position, 1)
        last_seen[object_id] = position

    return cold_count, histogram


def compute_entropy_bits(counts: np.ndarray, total_events: int) -> float:
    if total_events <= 0:
        return 0.0
    positive = counts[counts > 0].astype(np.float64)
    probabilities = positive / float(total_events)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_gini(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    if positive.size == 0:
        return 0.0
    ordered = np.sort(positive)
    n = ordered.size
    cumulative = np.cumsum(ordered)
    total = cumulative[-1]
    if total <= 0.0:
        return 0.0
    gini = (n + 1.0 - (2.0 * np.sum(cumulative) / total)) / n
    return float(max(0.0, min(1.0, gini)))


def histogram_quantile(counts: np.ndarray, quantile: float) -> float:
    total = int(np.sum(counts))
    if total <= 0:
        return 0.0
    if quantile <= 0.0:
        return float(np.flatnonzero(counts > 0)[0])
    if quantile >= 1.0:
        return float(np.flatnonzero(counts > 0)[-1])
    target = int(math.ceil(quantile * total))
    running = 0
    for value, count in enumerate(counts):
        running += int(count)
        if running >= target:
            return float(value)
    return float(np.flatnonzero(counts > 0)[-1])


def summarize_reuse_histogram(
    total_events: int,
    cold_count: int,
    reuse_histogram: np.ndarray,
) -> dict:
    finite_reuse_count = int(np.sum(reuse_histogram))
    if cold_count + finite_reuse_count != total_events:
        raise ValueError(
            "reuse-distance accounting mismatch: "
            f"cold={cold_count}, finite={finite_reuse_count}, total_events={total_events}"
        )

    if finite_reuse_count > 0:
        distances = np.arange(reuse_histogram.shape[0], dtype=np.float64)
        finite_mean = float(np.dot(distances, reuse_histogram.astype(np.float64)) / finite_reuse_count)
        finite_max = int(np.max(np.flatnonzero(reuse_histogram > 0)))
    else:
        finite_mean = 0.0
        finite_max = 0

    return {
        "cold_count": int(cold_count),
        "cold_fraction": float(cold_count / total_events) if total_events else 0.0,
        "finite_reuse_count": finite_reuse_count,
        "finite_reuse_fraction": float(finite_reuse_count / total_events) if total_events else 0.0,
        "finite_mean": finite_mean,
        "finite_median": histogram_quantile(reuse_histogram, 0.50),
        "finite_p95": histogram_quantile(reuse_histogram, 0.95),
        "finite_max": finite_max,
    }


def build_popularity_rows(condition: str, counts: np.ndarray, total_events: int) -> List[dict]:
    object_ids = [int(object_id) for object_id in np.flatnonzero(counts)]
    rows = []
    sortable = [
        (
            int(counts[object_id]),
            decode_object_key(condition, object_id),
        )
        for object_id in object_ids
    ]
    sortable.sort(key=lambda item: (-item[0], item[1]))

    cumulative_count = 0
    for rank, (count, object_key) in enumerate(sortable, start=1):
        cumulative_count += count
        fraction = float(count / total_events) if total_events else 0.0
        cumulative_fraction = float(cumulative_count / total_events) if total_events else 0.0
        rows.append(
            {
                "condition": condition,
                "rank": rank,
                "object_key": object_key,
                "count": count,
                "fraction": fraction,
                "cumulative_fraction": cumulative_fraction,
            }
        )
    return rows


def build_topk_rows(popularity_rows: Sequence[Mapping[str, object]]) -> List[dict]:
    return [
        {
            "condition": str(row["condition"]),
            "k": int(row["rank"]),
            "coverage": float(row["cumulative_fraction"]),
        }
        for row in popularity_rows
    ]


def build_reuse_cdf_rows(condition: str, total_events: int, cold_count: int, reuse_histogram: np.ndarray) -> List[dict]:
    rows = []
    cumulative_count = 0

    rows.append(
        {
            "condition": condition,
            "reuse_distance": COLD_REUSE_DISTANCE,
            "count": int(cold_count),
            "fraction": float(cold_count / total_events) if total_events else 0.0,
            "cdf": float(cold_count / total_events) if total_events else 0.0,
        }
    )
    cumulative_count += int(cold_count)

    for reuse_distance, count in enumerate(reuse_histogram):
        if count <= 0:
            continue
        cumulative_count += int(count)
        rows.append(
            {
                "condition": condition,
                "reuse_distance": int(reuse_distance),
                "count": int(count),
                "fraction": float(count / total_events) if total_events else 0.0,
                "cdf": float(cumulative_count / total_events) if total_events else 0.0,
            }
        )
    return rows


def selected_topk_coverage(popularity_rows: Sequence[Mapping[str, object]]) -> dict:
    if not popularity_rows:
        return {}
    coverage = {}
    for k in TOPK_SUMMARY_POINTS:
        if k <= len(popularity_rows):
            coverage[str(k)] = float(popularity_rows[k - 1]["cumulative_fraction"])
    coverage[str(len(popularity_rows))] = float(popularity_rows[-1]["cumulative_fraction"])
    return coverage


def analyze_condition(
    condition: str,
    buffer: ConditionAccessBuffer,
    per_request_unique_counts: Sequence[int],
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    access_ids = buffer.access_ids
    counts = np.bincount(access_ids)
    if counts.shape[0] <= 1:
        counts = np.pad(counts, (0, 2 - counts.shape[0]))

    nonzero_counts = counts[counts > 0]
    total_distinct_objects = int(nonzero_counts.size)
    entropy_bits = compute_entropy_bits(counts, buffer.total_events)
    normalized_entropy = (
        float(entropy_bits / math.log2(total_distinct_objects)) if total_distinct_objects > 1 else 0.0
    )
    effective_working_set_size = float(2.0**entropy_bits) if total_distinct_objects > 0 else 0.0
    gini_coefficient = compute_gini(counts)

    popularity_rows = build_popularity_rows(condition, counts, buffer.total_events)
    topk_rows = build_topk_rows(popularity_rows)

    cold_count, reuse_histogram = compute_reuse_histogram(
        access_ids=access_ids,
        id_space_size=counts.shape[0] - 1,
        max_distance=total_distinct_objects,
    )
    reuse_summary = summarize_reuse_histogram(
        total_events=buffer.total_events,
        cold_count=int(cold_count),
        reuse_histogram=reuse_histogram,
    )
    reuse_cdf_rows = build_reuse_cdf_rows(
        condition=condition,
        total_events=buffer.total_events,
        cold_count=int(cold_count),
        reuse_histogram=reuse_histogram,
    )

    per_request_summary = numeric_summary(per_request_unique_counts)
    per_request_summary["median"] = per_request_summary["p50"]

    summary = {
        "condition": condition,
        "object_key_definition": OBJECT_KEY_DEFINITIONS[condition],
        "object_key_note": OBJECT_KEY_NOTES[condition],
        "total_events": int(buffer.total_events),
        "total_distinct_objects": total_distinct_objects,
        "entropy_bits": entropy_bits,
        "normalized_entropy": normalized_entropy,
        "gini_coefficient": gini_coefficient,
        "effective_working_set_size": effective_working_set_size,
        "topk_coverage": selected_topk_coverage(popularity_rows),
        "reuse_distance": reuse_summary,
        "per_request_unique_object_count": per_request_summary,
        "top_objects": popularity_rows[:20],
    }

    del access_ids
    del counts
    del reuse_histogram
    gc.collect()
    return summary, popularity_rows, topk_rows, reuse_cdf_rows


def write_locality_outputs(
    output_dir: Path,
    summaries: Mapping[str, dict],
    popularity_rows: Sequence[Mapping[str, object]],
    reuse_cdf_rows: Sequence[Mapping[str, object]],
    topk_rows: Sequence[Mapping[str, object]],
) -> None:
    write_json(output_dir / "expert_only_locality.json", summaries[CONDITION_EXPERT_ONLY])
    write_json(output_dir / "joint_indep_locality.json", summaries[CONDITION_JOINT_INDEP])
    write_json(output_dir / "joint_corr_locality.json", summaries[CONDITION_JOINT_CORR])
    write_csv(output_dir / "popularity_rank.csv", POPULARITY_CSV_FIELDS, popularity_rows)
    write_csv(output_dir / "reuse_distance_cdf.csv", REUSE_CDF_CSV_FIELDS, reuse_cdf_rows)
    write_csv(output_dir / "topk_coverage.csv", TOPK_COVERAGE_CSV_FIELDS, topk_rows)


def run_locality_analysis(
    joined_indep_path: Path,
    joined_corr_path: Path,
    qc_report_path: Path,
    output_dir: Path,
    work_dir: Path,
    progress_every: int = PROGRESS_EVERY_ROWS,
) -> dict:
    qc_report = load_json(qc_report_path)
    total_events = resolve_total_events(qc_report)
    ensure_dir(output_dir)

    stream_result = build_access_streams(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        total_events=total_events,
        work_dir=work_dir,
        progress_every=progress_every,
    )

    summaries: Dict[str, dict] = {}
    popularity_rows: List[dict] = []
    reuse_cdf_rows: List[dict] = []
    topk_rows: List[dict] = []

    for condition in CONDITION_ORDER:
        print(f"computing locality metrics for {condition}")
        condition_summary, condition_popularity_rows, condition_topk_rows, condition_reuse_cdf_rows = analyze_condition(
            condition=condition,
            buffer=stream_result.condition_buffers[condition],
            per_request_unique_counts=stream_result.per_request_unique_counts[condition],
        )
        condition_summary["request_count"] = int(stream_result.request_count)
        condition_summary["request_stream_alignment"] = {
            "aligned_against": "joined_trace_indep.jsonl vs joined_trace_corr.jsonl",
            "checked_fields": stream_result.invariant_checked_fields,
            "position_field_counts": stream_result.position_field_counts,
        }
        summaries[condition] = condition_summary
        popularity_rows.extend(condition_popularity_rows)
        reuse_cdf_rows.extend(condition_reuse_cdf_rows)
        topk_rows.extend(condition_topk_rows)
        stream_result.condition_buffers[condition].access_ids = np.empty(0, dtype=np.uint32)
        gc.collect()

    write_locality_outputs(
        output_dir=output_dir,
        summaries=summaries,
        popularity_rows=popularity_rows,
        reuse_cdf_rows=reuse_cdf_rows,
        topk_rows=topk_rows,
    )

    return {
        "request_count": stream_result.request_count,
        "total_events": total_events,
        "summaries": summaries,
        "outputs": {
            "expert_only_locality_json": str(output_dir / "expert_only_locality.json"),
            "joint_indep_locality_json": str(output_dir / "joint_indep_locality.json"),
            "joint_corr_locality_json": str(output_dir / "joint_corr_locality.json"),
            "popularity_rank_csv": str(output_dir / "popularity_rank.csv"),
            "reuse_distance_cdf_csv": str(output_dir / "reuse_distance_cdf.csv"),
            "topk_coverage_csv": str(output_dir / "topk_coverage.csv"),
        },
    }


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)

    output_dir = stage_output_dir("replay/locality", config, args.run_id, args.output_dir)
    run_id = str(config.get("case_study", {}).get("default_run_id", "router_lora_case_v1")) if args.run_id is None else args.run_id
    joined_dir = stage_output_dir("joined_trace", config, run_id)

    joined_indep_path = Path(args.joined_indep_path) if args.joined_indep_path else joined_dir / "joined_trace_indep.jsonl"
    joined_corr_path = Path(args.joined_corr_path) if args.joined_corr_path else joined_dir / "joined_trace_corr.jsonl"
    qc_report_path = Path(args.qc_report_path) if args.qc_report_path else joined_dir / "join_qc_report.json"
    work_dir = Path(args.work_dir) if args.work_dir else output_dir

    result = run_locality_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=output_dir,
        work_dir=work_dir,
        progress_every=args.progress_every,
    )

    print(
        "wrote locality outputs: "
        f"{result['outputs']['expert_only_locality_json']}, "
        f"{result['outputs']['joint_indep_locality_json']}, "
        f"{result['outputs']['joint_corr_locality_json']}"
    )


if __name__ == "__main__":
    main()
