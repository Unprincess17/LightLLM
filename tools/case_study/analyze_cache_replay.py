#!/usr/bin/env python3
"""Compute B8 trace-driven cache replay metrics for aligned B0/B1/B2 streams."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from analyze_locality import (
    CONDITION_EXPERT_ONLY,
    CONDITION_JOINT_CORR,
    CONDITION_JOINT_INDEP,
    CONDITION_ORDER,
    OBJECT_KEY_DEFINITIONS,
    OBJECT_KEY_NOTES,
    iter_jsonl_bytes,
    pack_expert_object,
    pack_joint_object,
    paired_row_alignment_fields,
    parse_adapter_slot,
    resolve_total_events,
)
from common import (
    ensure_dir,
    load_global_config,
    load_seed_config,
    parse_cardinalities,
    stage_output_dir,
    write_csv,
    write_json,
)
from replay_core import BudgetReplaySummary, ConditionAccessBuffer, POLICY_LRU, simulate_cache_policy


POLICY_LRU = "lru"
DEFAULT_CACHE_POLICY = POLICY_LRU
DEFAULT_CACHE_BUDGETS = (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
PROGRESS_EVERY_ROWS = 1_000_000

CONDITION_LABELS = {
    CONDITION_EXPERT_ONLY: "B0",
    CONDITION_JOINT_INDEP: "B1",
    CONDITION_JOINT_CORR: "B2",
}

CACHE_CURVE_FIELDS = [
    "condition",
    "cache_budget",
    "total_events",
    "hits",
    "misses",
    "miss_rate",
    "hit_rate",
    "cold_miss_rate",
    "capacity_miss_rate",
]
PER_REQUEST_FIELDS = [
    "condition",
    "cache_budget",
    "req_idx",
    "miss_count",
    "hit_count",
    "total_events",
]
EVICTION_STATS_FIELDS = [
    "condition",
    "cache_budget",
    "total_evictions",
    "unique_evicted_objects",
    "mean_residency_if_available",
]


@dataclass
class StreamBuildResult:
    condition_buffers: Dict[str, ConditionAccessBuffer]
    request_offsets: np.ndarray
    request_ids: List[int]
    total_events: int
    position_field_counts: Dict[str, int]
    invariant_checked_fields: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute B8 offline cache replay metrics")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for replay/cache artifacts",
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
        "--policy",
        type=str,
        default=None,
        help="Cache replacement policy to replay (default: config value or lru)",
    )
    parser.add_argument(
        "--cache_budgets",
        type=str,
        default=None,
        help="Comma-separated cache budget grid in object slots (default: config value or built-in grid)",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=PROGRESS_EVERY_ROWS,
        help="Print progress after every N aligned joined-trace rows while building the replay stream",
    )
    return parser.parse_args()


def resolve_cache_section(config: Mapping[str, object]) -> Mapping[str, object]:
    case_section = config.get("case_study", {})
    if not isinstance(case_section, Mapping):
        return {}

    replay_cache = case_section.get("replay_cache", {})
    if isinstance(replay_cache, Mapping) and replay_cache:
        return replay_cache

    replay_section = case_section.get("replay", {})
    if not isinstance(replay_section, Mapping):
        return {}
    cache_section = replay_section.get("cache", {})
    if not isinstance(cache_section, Mapping):
        return {}
    return cache_section


def resolve_cache_policy(config: Mapping[str, object], override: Optional[str]) -> str:
    if override is not None:
        policy = override
    else:
        cache_section = resolve_cache_section(config)
        policy = cache_section.get("policy", DEFAULT_CACHE_POLICY)
    normalized = str(policy).strip().lower()
    if normalized != POLICY_LRU:
        raise ValueError(f"unsupported cache policy: {policy}")
    return normalized


def resolve_cache_budget_grid(config: Mapping[str, object], override: Optional[str]) -> List[int]:
    raw_value = override
    if raw_value is None:
        cache_section = resolve_cache_section(config)
        raw_value = cache_section.get("cache_budgets", cache_section.get("budget_grid", list(DEFAULT_CACHE_BUDGETS)))
    budgets = parse_cardinalities(raw_value)
    if not budgets:
        raise ValueError("cache budget grid must not be empty")

    normalized: List[int] = []
    previous: Optional[int] = None
    seen = set()
    for budget in budgets:
        if budget < 0:
            raise ValueError(f"cache budget must be non-negative, got {budget}")
        if budget in seen:
            raise ValueError(f"cache budget grid contains a duplicate entry: {budget}")
        if previous is not None and budget <= previous:
            raise ValueError(
                "cache budget grid must be strictly increasing to make the miss-rate curve auditable: "
                f"previous={previous}, current={budget}"
            )
        normalized.append(int(budget))
        previous = int(budget)
        seen.add(int(budget))
    return normalized


def build_access_streams(
    joined_indep_path: Path,
    joined_corr_path: Path,
    total_events: int,
    progress_every: int,
) -> StreamBuildResult:
    access_buffers = {
        CONDITION_EXPERT_ONLY: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_INDEP: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_CORR: np.empty(total_events, dtype=np.uint32),
    }
    max_object_ids = {condition: 0 for condition in CONDITION_ORDER}

    request_offsets = [0]
    request_ids: List[int] = []
    current_req_idx: Optional[int] = None
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
        position_field_counts[position_field] = position_field_counts.get(position_field, 0) + 1

        req_idx = int(indep_row["req_idx"])
        if current_req_idx is None:
            current_req_idx = req_idx
            request_ids.append(req_idx)
        elif req_idx < current_req_idx:
            raise ValueError(
                "joined trace request order regressed; B8 requires canonical non-decreasing req_idx order: "
                f"previous={current_req_idx}, current={req_idx}"
            )
        elif req_idx > current_req_idx:
            if req_idx != current_req_idx + 1:
                raise ValueError(
                    "joined trace req_idx order is not contiguous; B8 requires the same canonical request stream "
                    f"across conditions: previous={current_req_idx}, current={req_idx}"
                )
            request_offsets.append(rows_seen - 1)
            current_req_idx = req_idx
            request_ids.append(req_idx)

        layer_id = int(indep_row["layer_id"])
        expert_id = int(indep_row["expert_id"])
        expert_object_id = pack_expert_object(layer_id, expert_id)
        indep_object_id = pack_joint_object(layer_id, expert_id, parse_adapter_slot(indep_row["adapter_id"]))
        corr_object_id = pack_joint_object(layer_id, expert_id, parse_adapter_slot(corr_row["adapter_id"]))

        buffer_index = rows_seen - 1
        access_buffers[CONDITION_EXPERT_ONLY][buffer_index] = np.uint32(expert_object_id)
        access_buffers[CONDITION_JOINT_INDEP][buffer_index] = np.uint32(indep_object_id)
        access_buffers[CONDITION_JOINT_CORR][buffer_index] = np.uint32(corr_object_id)

        if expert_object_id > max_object_ids[CONDITION_EXPERT_ONLY]:
            max_object_ids[CONDITION_EXPERT_ONLY] = expert_object_id
        if indep_object_id > max_object_ids[CONDITION_JOINT_INDEP]:
            max_object_ids[CONDITION_JOINT_INDEP] = indep_object_id
        if corr_object_id > max_object_ids[CONDITION_JOINT_CORR]:
            max_object_ids[CONDITION_JOINT_CORR] = corr_object_id

        if progress_every > 0 and rows_seen % progress_every == 0:
            print(f"materialized {rows_seen}/{total_events} aligned cache-access objects")

    if rows_seen != total_events:
        raise ValueError(
            f"joined trace row count mismatch against join_qc_report.json: expected {total_events}, observed {rows_seen}"
        )

    if request_ids:
        request_offsets.append(total_events)

    condition_buffers = {
        condition: ConditionAccessBuffer(
            condition=condition,
            access_ids=access_buffers[condition],
            total_events=total_events,
            max_object_id=max_object_ids[condition],
        )
        for condition in CONDITION_ORDER
    }
    gc.collect()

    return StreamBuildResult(
        condition_buffers=condition_buffers,
        request_offsets=np.asarray(request_offsets, dtype=np.int64),
        request_ids=request_ids,
        total_events=total_events,
        position_field_counts=position_field_counts,
        invariant_checked_fields=invariant_checked_fields,
    )

def summary_to_cache_curve_row(summary: BudgetReplaySummary) -> dict:
    return {
        "condition": summary.condition,
        "cache_budget": summary.cache_budget,
        "total_events": summary.total_events,
        "hits": summary.hits,
        "misses": summary.misses,
        "miss_rate": summary.miss_rate,
        "hit_rate": summary.hit_rate,
        "cold_miss_rate": summary.cold_miss_rate,
        "capacity_miss_rate": summary.capacity_miss_rate,
    }


def summary_to_eviction_row(summary: BudgetReplaySummary) -> dict:
    return {
        "condition": summary.condition,
        "cache_budget": summary.cache_budget,
        "total_evictions": summary.total_evictions,
        "unique_evicted_objects": summary.unique_evicted_objects,
        "mean_residency_if_available": summary.mean_residency_if_available,
    }


def build_per_request_rows(
    summary: BudgetReplaySummary,
    request_ids: Sequence[int],
    request_offsets: np.ndarray,
) -> List[dict]:
    rows = []
    request_count = len(request_ids)
    if request_count != summary.per_request_hits.shape[0] or request_count != summary.per_request_misses.shape[0]:
        raise ValueError(
            "per-request replay output size mismatch: "
            f"condition={summary.condition}, budget={summary.cache_budget}, requests={request_count}, "
            f"hits_len={summary.per_request_hits.shape[0]}, misses_len={summary.per_request_misses.shape[0]}"
        )

    for request_ordinal, req_idx in enumerate(request_ids):
        start_offset = int(request_offsets[request_ordinal])
        end_offset = int(request_offsets[request_ordinal + 1])
        total_request_events = end_offset - start_offset
        rows.append(
            {
                "condition": summary.condition,
                "cache_budget": summary.cache_budget,
                "req_idx": int(req_idx),
                "miss_count": int(summary.per_request_misses[request_ordinal]),
                "hit_count": int(summary.per_request_hits[request_ordinal]),
                "total_events": int(total_request_events),
            }
        )
    return rows


def write_cache_outputs(
    output_dir: Path,
    cache_metrics: Mapping[str, object],
    cache_curve_rows: Sequence[Mapping[str, object]],
    per_request_rows: Sequence[Mapping[str, object]],
    eviction_rows: Sequence[Mapping[str, object]],
) -> None:
    write_json(output_dir / "cache_metrics.json", cache_metrics)
    write_csv(output_dir / "cache_curve.csv", CACHE_CURVE_FIELDS, cache_curve_rows)
    write_csv(output_dir / "per_request_miss_count.csv", PER_REQUEST_FIELDS, per_request_rows)
    write_csv(output_dir / "eviction_stats.csv", EVICTION_STATS_FIELDS, eviction_rows)


def run_cache_replay_analysis(
    joined_indep_path: Path,
    joined_corr_path: Path,
    qc_report_path: Path,
    output_dir: Path,
    policy: str,
    cache_budgets: Sequence[int],
    seed_config: Optional[Mapping[str, object]] = None,
    progress_every: int = PROGRESS_EVERY_ROWS,
) -> dict:
    import orjson

    qc_payload = orjson.loads(qc_report_path.read_bytes())
    if not isinstance(qc_payload, dict):
        raise ValueError(f"expected JSON object in {qc_report_path}")
    total_events = resolve_total_events(qc_payload)
    ensure_dir(output_dir)

    stream_result = build_access_streams(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        total_events=total_events,
        progress_every=progress_every,
    )

    cache_curve_rows: List[dict] = []
    per_request_rows: List[dict] = []
    eviction_rows: List[dict] = []
    per_condition_metrics: Dict[str, dict] = {}

    print(f"cache budget grid (object slots): {list(cache_budgets)}")
    for condition in CONDITION_ORDER:
        print(f"replaying cache metrics for {condition} with policy={policy}")
        access_buffer = stream_result.condition_buffers[condition]
        budget_metrics: Dict[str, dict] = {}
        for cache_budget in cache_budgets:
            print(f"  replay budget={cache_budget}")
            summary = simulate_cache_policy(
                policy=policy,
                access_buffer=access_buffer,
                request_offsets=stream_result.request_offsets,
                cache_budget=int(cache_budget),
            )
            cache_curve_rows.append(summary_to_cache_curve_row(summary))
            eviction_rows.append(summary_to_eviction_row(summary))
            per_request_rows.extend(
                build_per_request_rows(
                    summary=summary,
                    request_ids=stream_result.request_ids,
                    request_offsets=stream_result.request_offsets,
                )
            )
            budget_metrics[str(cache_budget)] = {
                "cache_budget": summary.cache_budget,
                "total_events": summary.total_events,
                "hits": summary.hits,
                "misses": summary.misses,
                "hit_rate": summary.hit_rate,
                "miss_rate": summary.miss_rate,
                "cold_misses": summary.cold_misses,
                "cold_miss_rate": summary.cold_miss_rate,
                "capacity_misses": summary.capacity_misses,
                "capacity_miss_rate": summary.capacity_miss_rate,
                "total_distinct_objects": summary.total_distinct_objects,
                "total_evictions": summary.total_evictions,
                "eviction_frequency": summary.eviction_frequency,
                "unique_evicted_objects": summary.unique_evicted_objects,
                "mean_residency_if_available": summary.mean_residency_if_available,
                "per_request_hit_count_summary": summary.per_request_hit_count_summary,
                "per_request_miss_count_summary": summary.per_request_miss_count_summary,
            }

        per_condition_metrics[condition] = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "object_key_definition": OBJECT_KEY_DEFINITIONS[condition],
            "object_key_note": OBJECT_KEY_NOTES[condition],
            "total_events": access_buffer.total_events,
            "max_object_id": access_buffer.max_object_id,
            "budgets": budget_metrics,
        }
        stream_result.condition_buffers[condition].access_ids = np.empty(0, dtype=np.uint32)
        gc.collect()

    cache_metrics = {
        "cache_policy": policy,
        "budget_unit": "objects",
        "cache_budget_grid": [int(value) for value in cache_budgets],
        "comparison_contract": {
            "identical_budget_grid_across_conditions": True,
            "identical_event_ordering_across_conditions": True,
            "identical_policy_across_conditions": True,
        },
        "condition_labels": CONDITION_LABELS,
        "request_count": len(stream_result.request_ids),
        "request_ids": [int(req_idx) for req_idx in stream_result.request_ids],
        "total_events": total_events,
        "request_stream_alignment": {
            "aligned_against": "joined_trace_indep.jsonl vs joined_trace_corr.jsonl",
            "checked_fields": stream_result.invariant_checked_fields,
            "position_field_counts": stream_result.position_field_counts,
        },
        "seeds": dict(seed_config or {}),
        "inputs": {
            "joined_indep_path": str(joined_indep_path),
            "joined_corr_path": str(joined_corr_path),
            "qc_report_path": str(qc_report_path),
        },
        "conditions": per_condition_metrics,
        "outputs": {
            "cache_metrics_json": str(output_dir / "cache_metrics.json"),
            "cache_curve_csv": str(output_dir / "cache_curve.csv"),
            "per_request_miss_count_csv": str(output_dir / "per_request_miss_count.csv"),
            "eviction_stats_csv": str(output_dir / "eviction_stats.csv"),
        },
    }

    write_cache_outputs(
        output_dir=output_dir,
        cache_metrics=cache_metrics,
        cache_curve_rows=cache_curve_rows,
        per_request_rows=per_request_rows,
        eviction_rows=eviction_rows,
    )
    return cache_metrics


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config()

    output_dir = stage_output_dir("replay/cache", config, args.run_id, args.output_dir)
    joined_dir = stage_output_dir("joined_trace", config, args.run_id)

    joined_indep_path = Path(args.joined_indep_path) if args.joined_indep_path else joined_dir / "joined_trace_indep.jsonl"
    joined_corr_path = Path(args.joined_corr_path) if args.joined_corr_path else joined_dir / "joined_trace_corr.jsonl"
    qc_report_path = Path(args.qc_report_path) if args.qc_report_path else joined_dir / "join_qc_report.json"

    policy = resolve_cache_policy(config, args.policy)
    cache_budgets = resolve_cache_budget_grid(config, args.cache_budgets)

    result = run_cache_replay_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=output_dir,
        policy=policy,
        cache_budgets=cache_budgets,
        seed_config=seeds,
        progress_every=args.progress_every,
    )

    print(
        "wrote cache replay outputs: "
        f"{result['outputs']['cache_metrics_json']}, "
        f"{result['outputs']['cache_curve_csv']}, "
        f"{result['outputs']['per_request_miss_count_csv']}, "
        f"{result['outputs']['eviction_stats_csv']}"
    )


if __name__ == "__main__":
    main()
