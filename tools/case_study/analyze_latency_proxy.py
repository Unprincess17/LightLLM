#!/usr/bin/env python3
"""Compute B9 latency proxy summaries on top of B8 cache replay outputs."""

from __future__ import annotations

import argparse
import csv
import gc
import math
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
from common import ensure_dir, load_global_config, load_seed_config, percentile, stage_output_dir, write_csv, write_json
from replay_core import (
    ConditionAccessBuffer,
    PHASE_DECODE,
    PHASE_PREFILL,
    PHASE_UNKNOWN,
    POLICY_LRU,
    PhaseReplaySummary,
    simulate_phase_replay,
)

CONDITION_LABELS = {
    CONDITION_EXPERT_ONLY: "B0",
    CONDITION_JOINT_INDEP: "B1",
    CONDITION_JOINT_CORR: "B2",
}
LATENCY_QUANTILE_FIELDS = ["condition", "cache_budget", "mean", "p50", "p95", "p99", "max"]
TAIL_BREAKDOWN_FIELDS = [
    "condition",
    "cache_budget",
    "req_idx",
    "latency_proxy",
    "miss_count",
    "cold_miss_count",
    "prefill_miss_count",
    "decode_miss_count",
]
PHASE_BREAKDOWN_FIELDS = ["condition", "cache_budget", "metric_name", "metric_value"]
DEFAULT_PROGRESS_EVERY_ROWS = 1_000_000
DEFAULT_TAIL_QUANTILE = 0.95
DEFAULT_PREFILL_MISS_RATIO = 0.25


@dataclass
class PhaseAwareStreamBuildResult:
    condition_buffers: Dict[str, ConditionAccessBuffer]
    phase_ids: np.ndarray
    request_offsets: np.ndarray
    request_ids: List[int]
    per_request_prefill_events: np.ndarray
    per_request_decode_events: np.ndarray
    total_events: int
    phase_available: bool
    position_field_counts: Dict[str, int]
    invariant_checked_fields: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute B9 latency proxy summaries for cache replay conditions")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for replay/latency artifacts",
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
        "--cache_curve_path",
        type=str,
        default=None,
        help="Override replay/cache/cache_curve.csv input path",
    )
    parser.add_argument(
        "--per_request_miss_path",
        type=str,
        default=None,
        help="Override replay/cache/per_request_miss_count.csv input path for consistency checks",
    )
    parser.add_argument(
        "--router_request_log_path",
        type=str,
        default=None,
        help="Override router_trace/router_request_log.jsonl input path used to fit the base compute term",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_ROWS,
        help="Print progress after every N aligned joined-trace rows while materializing the replay stream",
    )
    parser.add_argument(
        "--tail_quantile",
        type=float,
        default=DEFAULT_TAIL_QUANTILE,
        help="Tail threshold quantile used for tail_request_breakdown.csv",
    )
    parser.add_argument(
        "--base_request_ms",
        type=float,
        default=None,
        help="Optional override for the fitted request intercept term",
    )
    parser.add_argument(
        "--prefill_event_cost_ms",
        type=float,
        default=None,
        help="Optional override for the base compute term per prefill expert event",
    )
    parser.add_argument(
        "--decode_event_cost_ms",
        type=float,
        default=None,
        help="Optional override for the base compute term per decode expert event",
    )
    parser.add_argument(
        "--hit_cost_ms",
        type=float,
        default=None,
        help="Optional extra per-hit request cost added on top of the base compute term",
    )
    parser.add_argument(
        "--miss_penalty_ms",
        type=float,
        default=None,
        help="Optional aggregate miss penalty used when phase labels are unavailable",
    )
    parser.add_argument(
        "--miss_penalty_prefill_ms",
        type=float,
        default=None,
        help="Optional phase-aware miss penalty applied to prefill misses",
    )
    parser.add_argument(
        "--miss_penalty_decode_ms",
        type=float,
        default=None,
        help="Optional phase-aware miss penalty applied to decode misses",
    )
    parser.add_argument(
        "--prefill_miss_ratio",
        type=float,
        default=DEFAULT_PREFILL_MISS_RATIO,
        help="If prefill miss penalty is not overridden, use prefill_miss_ratio * decode_event_cost_ms",
    )
    return parser.parse_args()


def resolve_latency_section(config: Mapping[str, object]) -> Mapping[str, object]:
    case_section = config.get("case_study", {})
    if not isinstance(case_section, Mapping):
        return {}

    latency_section = case_section.get("replay_latency_proxy", {})
    if isinstance(latency_section, Mapping) and latency_section:
        return latency_section

    replay_section = case_section.get("replay", {})
    if not isinstance(replay_section, Mapping):
        return {}
    latency_section = replay_section.get("latency_proxy", {})
    if not isinstance(latency_section, Mapping):
        return {}
    return latency_section


def phase_name_to_id(phase: object) -> int:
    phase_name = str(phase or "").strip().lower()
    if phase_name == "prefill":
        return PHASE_PREFILL
    if phase_name == "decode":
        return PHASE_DECODE
    return PHASE_UNKNOWN


def load_cache_budget_grid_from_curve(path: Path) -> List[int]:
    budgets = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            budget = int(row["cache_budget"])
            if budget in seen:
                continue
            budgets.append(budget)
            seen.add(budget)
    if not budgets:
        raise ValueError(f"cache curve CSV is empty: {path}")
    return budgets


def load_expected_per_request_misses(path: Path) -> Dict[Tuple[str, int, int], int]:
    expected = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            expected[(row["condition"], int(row["cache_budget"]), int(row["req_idx"]))] = int(row["miss_count"])
    return expected


def load_request_latencies(path: Path) -> Dict[int, float]:
    import orjson

    latencies = {}
    with path.open("rb") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = orjson.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_num} is not a JSON object")
            if payload.get("success") is False:
                continue
            req_idx = int(payload["req_idx"])
            latencies[req_idx] = float(payload["latency_ms"])
    if not latencies:
        raise ValueError(f"request log is empty: {path}")
    return latencies


def build_phase_aware_streams(
    joined_indep_path: Path,
    joined_corr_path: Path,
    total_events: int,
    progress_every: int,
) -> PhaseAwareStreamBuildResult:
    access_buffers = {
        CONDITION_EXPERT_ONLY: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_INDEP: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_CORR: np.empty(total_events, dtype=np.uint32),
    }
    phase_ids = np.empty(total_events, dtype=np.uint8)
    max_object_ids = {condition: 0 for condition in CONDITION_ORDER}

    request_offsets = [0]
    request_ids: List[int] = []
    per_request_prefill_events: List[int] = []
    per_request_decode_events: List[int] = []
    current_prefill_events = 0
    current_decode_events = 0
    current_req_idx: Optional[int] = None
    phase_available = True
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
                "joined trace request order regressed; B9 requires canonical non-decreasing req_idx order: "
                f"previous={current_req_idx}, current={req_idx}"
            )
        elif req_idx > current_req_idx:
            if req_idx != current_req_idx + 1:
                raise ValueError(
                    "joined trace req_idx order is not contiguous; B9 requires the same canonical request stream "
                    f"across conditions: previous={current_req_idx}, current={req_idx}"
                )
            request_offsets.append(rows_seen - 1)
            per_request_prefill_events.append(current_prefill_events)
            per_request_decode_events.append(current_decode_events)
            current_prefill_events = 0
            current_decode_events = 0
            current_req_idx = req_idx
            request_ids.append(req_idx)

        layer_id = int(indep_row["layer_id"])
        expert_id = int(indep_row["expert_id"])
        expert_object_id = pack_expert_object(layer_id, expert_id)
        indep_object_id = pack_joint_object(layer_id, expert_id, parse_adapter_slot(indep_row["adapter_id"]))
        corr_object_id = pack_joint_object(layer_id, expert_id, parse_adapter_slot(corr_row["adapter_id"]))

        phase_id = phase_name_to_id(indep_row.get("phase"))
        if phase_id == PHASE_PREFILL:
            current_prefill_events += 1
        elif phase_id == PHASE_DECODE:
            current_decode_events += 1
        else:
            phase_available = False

        buffer_index = rows_seen - 1
        access_buffers[CONDITION_EXPERT_ONLY][buffer_index] = np.uint32(expert_object_id)
        access_buffers[CONDITION_JOINT_INDEP][buffer_index] = np.uint32(indep_object_id)
        access_buffers[CONDITION_JOINT_CORR][buffer_index] = np.uint32(corr_object_id)
        phase_ids[buffer_index] = np.uint8(phase_id)

        if expert_object_id > max_object_ids[CONDITION_EXPERT_ONLY]:
            max_object_ids[CONDITION_EXPERT_ONLY] = expert_object_id
        if indep_object_id > max_object_ids[CONDITION_JOINT_INDEP]:
            max_object_ids[CONDITION_JOINT_INDEP] = indep_object_id
        if corr_object_id > max_object_ids[CONDITION_JOINT_CORR]:
            max_object_ids[CONDITION_JOINT_CORR] = corr_object_id

        if progress_every > 0 and rows_seen % progress_every == 0:
            print(f"materialized {rows_seen}/{total_events} aligned cache-access objects for latency replay")

    if rows_seen != total_events:
        raise ValueError(
            f"joined trace row count mismatch against join_qc_report.json: expected {total_events}, observed {rows_seen}"
        )

    if request_ids:
        request_offsets.append(total_events)
        per_request_prefill_events.append(current_prefill_events)
        per_request_decode_events.append(current_decode_events)

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

    return PhaseAwareStreamBuildResult(
        condition_buffers=condition_buffers,
        phase_ids=phase_ids,
        request_offsets=np.asarray(request_offsets, dtype=np.int64),
        request_ids=request_ids,
        per_request_prefill_events=np.asarray(per_request_prefill_events, dtype=np.int64),
        per_request_decode_events=np.asarray(per_request_decode_events, dtype=np.int64),
        total_events=total_events,
        phase_available=phase_available,
        position_field_counts=position_field_counts,
        invariant_checked_fields=invariant_checked_fields,
    )

def summarize_latency_values(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    ordered = [float(value) for value in values]
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "p50": float(percentile(ordered, 0.50)),
        "p95": float(percentile(ordered, 0.95)),
        "p99": float(percentile(ordered, 0.99)),
        "max": float(max(ordered)),
    }


def compute_r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    if observed.size == 0:
        return 0.0
    residual_sum = float(np.sum((observed - predicted) ** 2))
    centered = observed - np.mean(observed)
    total_sum = float(np.sum(centered**2))
    if total_sum <= 0.0:
        return 1.0 if residual_sum <= 0.0 else 0.0
    return float(1.0 - (residual_sum / total_sum))


def fit_base_compute_model(
    request_ids: Sequence[int],
    per_request_prefill_events: np.ndarray,
    per_request_decode_events: np.ndarray,
    request_latencies_ms: Mapping[int, float],
    base_request_ms_override: Optional[float],
    prefill_event_cost_ms_override: Optional[float],
    decode_event_cost_ms_override: Optional[float],
) -> Dict[str, object]:
    if base_request_ms_override is not None and prefill_event_cost_ms_override is not None and decode_event_cost_ms_override is not None:
        return {
            "fit_method": "manual_override",
            "request_count": len(request_ids),
            "base_request_ms": float(base_request_ms_override),
            "prefill_event_cost_ms": float(prefill_event_cost_ms_override),
            "decode_event_cost_ms": float(decode_event_cost_ms_override),
            "mae_ms": None,
            "rmse_ms": None,
            "r2": None,
        }

    observed_req_ids = [req_idx for req_idx in request_ids if int(req_idx) in request_latencies_ms]
    if not observed_req_ids:
        raise ValueError("router request log does not overlap the replay request ids; cannot fit the base compute model")

    ordinal_by_req = {int(req_idx): ordinal for ordinal, req_idx in enumerate(request_ids)}
    observed = []
    design_rows = []
    for req_idx in observed_req_ids:
        ordinal = ordinal_by_req[int(req_idx)]
        observed.append(float(request_latencies_ms[int(req_idx)]))
        design_rows.append(
            [
                1.0,
                float(per_request_prefill_events[ordinal]),
                float(per_request_decode_events[ordinal]),
            ]
        )

    design = np.asarray(design_rows, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    coefficients = np.maximum(coefficients, 0.0)

    base_request_ms = float(base_request_ms_override) if base_request_ms_override is not None else float(coefficients[0])
    prefill_event_cost_ms = (
        float(prefill_event_cost_ms_override) if prefill_event_cost_ms_override is not None else float(coefficients[1])
    )
    decode_event_cost_ms = (
        float(decode_event_cost_ms_override) if decode_event_cost_ms_override is not None else float(coefficients[2])
    )

    predicted = (
        base_request_ms
        + design[:, 1] * prefill_event_cost_ms
        + design[:, 2] * decode_event_cost_ms
    )
    residual = y - predicted
    mae_ms = float(np.mean(np.abs(residual)))
    rmse_ms = float(math.sqrt(float(np.mean(residual**2))))

    return {
        "fit_method": "least_squares_nonnegative",
        "request_count": len(observed_req_ids),
        "base_request_ms": base_request_ms,
        "prefill_event_cost_ms": prefill_event_cost_ms,
        "decode_event_cost_ms": decode_event_cost_ms,
        "mae_ms": mae_ms,
        "rmse_ms": rmse_ms,
        "r2": compute_r2_score(y, predicted),
        "observed_req_idx_range": [int(min(observed_req_ids)), int(max(observed_req_ids))],
    }


def fit_total_event_cost_model(
    request_ids: Sequence[int],
    per_request_total_events: np.ndarray,
    request_latencies_ms: Mapping[int, float],
    base_request_ms_override: Optional[float],
    event_cost_ms_override: Optional[float],
) -> Dict[str, object]:
    if base_request_ms_override is not None and event_cost_ms_override is not None:
        return {
            "fit_method": "manual_override",
            "request_count": len(request_ids),
            "base_request_ms": float(base_request_ms_override),
            "event_cost_ms": float(event_cost_ms_override),
            "mae_ms": None,
            "rmse_ms": None,
            "r2": None,
        }

    observed_req_ids = [req_idx for req_idx in request_ids if int(req_idx) in request_latencies_ms]
    if not observed_req_ids:
        raise ValueError("router request log does not overlap the replay request ids; cannot fit the aggregate event-cost model")

    ordinal_by_req = {int(req_idx): ordinal for ordinal, req_idx in enumerate(request_ids)}
    observed = []
    design_rows = []
    for req_idx in observed_req_ids:
        ordinal = ordinal_by_req[int(req_idx)]
        observed.append(float(request_latencies_ms[int(req_idx)]))
        design_rows.append([1.0, float(per_request_total_events[ordinal])])

    design = np.asarray(design_rows, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    coefficients = np.maximum(coefficients, 0.0)

    base_request_ms = float(base_request_ms_override) if base_request_ms_override is not None else float(coefficients[0])
    event_cost_ms = float(event_cost_ms_override) if event_cost_ms_override is not None else float(coefficients[1])
    predicted = base_request_ms + design[:, 1] * event_cost_ms
    residual = y - predicted

    return {
        "fit_method": "least_squares_nonnegative",
        "request_count": len(observed_req_ids),
        "base_request_ms": base_request_ms,
        "event_cost_ms": event_cost_ms,
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(math.sqrt(float(np.mean(residual**2)))),
        "r2": compute_r2_score(y, predicted),
        "observed_req_idx_range": [int(min(observed_req_ids)), int(max(observed_req_ids))],
    }


def resolve_model_parameters(
    args: argparse.Namespace,
    config: Mapping[str, object],
    stream_result: PhaseAwareStreamBuildResult,
    request_log_path: Path,
) -> Dict[str, object]:
    latency_section = resolve_latency_section(config)
    request_latencies_ms = load_request_latencies(request_log_path)
    per_request_total_events = stream_result.request_offsets[1:] - stream_result.request_offsets[:-1]

    if stream_result.phase_available:
        fitted_model = fit_base_compute_model(
            request_ids=stream_result.request_ids,
            per_request_prefill_events=stream_result.per_request_prefill_events,
            per_request_decode_events=stream_result.per_request_decode_events,
            request_latencies_ms=request_latencies_ms,
            base_request_ms_override=args.base_request_ms
            if args.base_request_ms is not None
            else latency_section.get("base_request_ms"),
            prefill_event_cost_ms_override=args.prefill_event_cost_ms
            if args.prefill_event_cost_ms is not None
            else latency_section.get("prefill_event_cost_ms"),
            decode_event_cost_ms_override=args.decode_event_cost_ms
            if args.decode_event_cost_ms is not None
            else latency_section.get("decode_event_cost_ms"),
        )
        base_request_ms = float(fitted_model["base_request_ms"])
        prefill_event_cost_ms = float(fitted_model["prefill_event_cost_ms"])
        decode_event_cost_ms = float(fitted_model["decode_event_cost_ms"])
        event_cost_ms = float(
            (
                np.sum(stream_result.per_request_prefill_events) * prefill_event_cost_ms
                + np.sum(stream_result.per_request_decode_events) * decode_event_cost_ms
            )
            / max(stream_result.total_events, 1)
        )
    else:
        fitted_model = fit_total_event_cost_model(
            request_ids=stream_result.request_ids,
            per_request_total_events=per_request_total_events,
            request_latencies_ms=request_latencies_ms,
            base_request_ms_override=args.base_request_ms
            if args.base_request_ms is not None
            else latency_section.get("base_request_ms"),
            event_cost_ms_override=args.decode_event_cost_ms
            if args.decode_event_cost_ms is not None
            else latency_section.get("event_cost_ms"),
        )
        base_request_ms = float(fitted_model["base_request_ms"])
        event_cost_ms = float(fitted_model["event_cost_ms"])
        prefill_event_cost_ms = 0.0
        decode_event_cost_ms = event_cost_ms

    hit_cost_ms = (
        float(args.hit_cost_ms)
        if args.hit_cost_ms is not None
        else float(latency_section.get("hit_cost_ms", 0.0))
    )

    if stream_result.phase_available:
        prefill_ratio = float(args.prefill_miss_ratio)
        miss_penalty_prefill_ms = (
            float(args.miss_penalty_prefill_ms)
            if args.miss_penalty_prefill_ms is not None
            else float(
                latency_section.get(
                    "miss_penalty_prefill_ms",
                    max(prefill_event_cost_ms, prefill_ratio * decode_event_cost_ms),
                )
            )
        )
        miss_penalty_decode_ms = (
            float(args.miss_penalty_decode_ms)
            if args.miss_penalty_decode_ms is not None
            else float(
                latency_section.get(
                    "miss_penalty_decode_ms",
                    decode_event_cost_ms,
                )
            )
        )
        weighted_miss_penalty_ms = float(
            (
                np.sum(stream_result.per_request_prefill_events) * miss_penalty_prefill_ms
                + np.sum(stream_result.per_request_decode_events) * miss_penalty_decode_ms
            )
            / max(stream_result.total_events, 1)
        )
    else:
        miss_penalty_prefill_ms = None
        miss_penalty_decode_ms = None
        weighted_miss_penalty_ms = (
            float(args.miss_penalty_ms)
            if args.miss_penalty_ms is not None
            else float(latency_section.get("miss_penalty_ms", event_cost_ms))
        )

    return {
        "phase_aware": bool(stream_result.phase_available),
        "tail_quantile": float(args.tail_quantile),
        "base_request_ms": base_request_ms,
        "prefill_event_cost_ms": prefill_event_cost_ms,
        "decode_event_cost_ms": decode_event_cost_ms,
        "event_cost_ms": event_cost_ms,
        "hit_cost_ms": hit_cost_ms,
        "miss_penalty_ms": weighted_miss_penalty_ms,
        "miss_penalty_prefill_ms": miss_penalty_prefill_ms,
        "miss_penalty_decode_ms": miss_penalty_decode_ms,
        "prefill_miss_ratio": float(args.prefill_miss_ratio),
        "base_compute_fit": fitted_model,
    }


def compute_base_compute_latency(
    model_parameters: Mapping[str, object],
    total_events: int,
    prefill_events: int,
    decode_events: int,
    phase_available: bool,
) -> float:
    if phase_available:
        return float(model_parameters["base_request_ms"]) + (
            float(model_parameters["prefill_event_cost_ms"]) * float(prefill_events)
        ) + (
            float(model_parameters["decode_event_cost_ms"]) * float(decode_events)
        )
    return float(model_parameters["base_request_ms"]) + (
        float(model_parameters["event_cost_ms"]) * float(total_events)
    )


def compute_latency_proxy_rows(
    condition: str,
    cache_budget: int,
    request_ids: Sequence[int],
    summary: PhaseReplaySummary,
    per_request_prefill_events: np.ndarray,
    per_request_decode_events: np.ndarray,
    model_parameters: Mapping[str, object],
    phase_available: bool,
) -> Tuple[List[dict], Dict[str, float], List[dict], List[dict]]:
    per_request_rows: List[dict] = []
    latency_values: List[float] = []
    condition_label = CONDITION_LABELS[condition]

    if len(request_ids) != summary.per_request_hits.shape[0]:
        raise ValueError(
            "per-request replay output size mismatch: "
            f"condition={condition}, budget={cache_budget}, requests={len(request_ids)}, "
            f"hits_len={summary.per_request_hits.shape[0]}"
        )

    for request_ordinal, req_idx in enumerate(request_ids):
        prefill_events = int(per_request_prefill_events[request_ordinal])
        decode_events = int(per_request_decode_events[request_ordinal])
        hit_count = int(summary.per_request_hits[request_ordinal])
        miss_count = int(summary.per_request_misses[request_ordinal])
        total_event_count = hit_count + miss_count
        cold_miss_count = int(summary.per_request_cold_misses[request_ordinal])
        prefill_miss_count = int(summary.per_request_prefill_misses[request_ordinal]) if phase_available else 0
        decode_miss_count = int(summary.per_request_decode_misses[request_ordinal]) if phase_available else 0
        base_compute_cost = compute_base_compute_latency(
            model_parameters=model_parameters,
            total_events=total_event_count,
            prefill_events=prefill_events,
            decode_events=decode_events,
            phase_available=phase_available,
        )

        latency_proxy = base_compute_cost + (float(model_parameters["hit_cost_ms"]) * hit_count)
        if phase_available:
            latency_proxy += float(model_parameters["miss_penalty_prefill_ms"]) * prefill_miss_count
            latency_proxy += float(model_parameters["miss_penalty_decode_ms"]) * decode_miss_count
        else:
            latency_proxy += float(model_parameters["miss_penalty_ms"]) * miss_count

        latency_values.append(float(latency_proxy))
        per_request_rows.append(
            {
                "req_idx": int(req_idx),
                "latency_proxy": float(latency_proxy),
                "base_compute_cost": float(base_compute_cost),
                "hit_count": hit_count,
                "miss_count": miss_count,
                "cold_miss_count": cold_miss_count,
                "total_events": total_event_count,
                "prefill_events": prefill_events,
                "decode_events": decode_events,
                "prefill_miss_count": prefill_miss_count,
                "decode_miss_count": decode_miss_count,
            }
        )

    quantiles = summarize_latency_values(latency_values)
    tail_threshold = float(percentile(latency_values, float(model_parameters["tail_quantile"])))
    tail_rows = [
        {
            "condition": condition_label,
            "cache_budget": int(cache_budget),
            "req_idx": row["req_idx"],
            "latency_proxy": row["latency_proxy"],
            "miss_count": row["miss_count"],
            "cold_miss_count": row["cold_miss_count"],
            "prefill_miss_count": row["prefill_miss_count"],
            "decode_miss_count": row["decode_miss_count"],
        }
        for row in per_request_rows
        if float(row["latency_proxy"]) >= tail_threshold
    ]
    tail_rows.sort(key=lambda row: (-float(row["latency_proxy"]), int(row["req_idx"])))

    tail_source_rows = [row for row in per_request_rows if float(row["latency_proxy"]) >= tail_threshold]
    tail_count = len(tail_source_rows)
    total_tail_prefill_misses = float(sum(row["prefill_miss_count"] for row in tail_source_rows))
    total_tail_decode_misses = float(sum(row["decode_miss_count"] for row in tail_source_rows))
    total_tail_misses = total_tail_prefill_misses + total_tail_decode_misses
    total_prefill_misses = float(sum(row["prefill_miss_count"] for row in per_request_rows))
    total_decode_misses = float(sum(row["decode_miss_count"] for row in per_request_rows))
    total_misses = total_prefill_misses + total_decode_misses

    phase_breakdown_rows = []
    if phase_available:
        phase_breakdown_rows = [
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "mean_prefill_miss_count",
                "metric_value": total_prefill_misses / len(per_request_rows) if per_request_rows else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "mean_decode_miss_count",
                "metric_value": total_decode_misses / len(per_request_rows) if per_request_rows else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "mean_prefill_miss_share",
                "metric_value": (total_prefill_misses / total_misses) if total_misses > 0 else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "mean_decode_miss_share",
                "metric_value": (total_decode_misses / total_misses) if total_misses > 0 else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "tail_p95_request_count",
                "metric_value": float(tail_count),
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "tail_p95_prefill_miss_count_mean",
                "metric_value": total_tail_prefill_misses / tail_count if tail_count > 0 else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "tail_p95_decode_miss_count_mean",
                "metric_value": total_tail_decode_misses / tail_count if tail_count > 0 else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "tail_p95_prefill_miss_share",
                "metric_value": (total_tail_prefill_misses / total_tail_misses) if total_tail_misses > 0 else 0.0,
            },
            {
                "condition": condition_label,
                "cache_budget": int(cache_budget),
                "metric_name": "tail_p95_decode_miss_share",
                "metric_value": (total_tail_decode_misses / total_tail_misses) if total_tail_misses > 0 else 0.0,
            },
        ]

    return per_request_rows, quantiles, tail_rows, phase_breakdown_rows


def validate_against_b8_misses(
    condition: str,
    cache_budget: int,
    request_ids: Sequence[int],
    observed_per_request_misses: np.ndarray,
    expected_b8_misses: Optional[Mapping[Tuple[str, int, int], int]],
) -> None:
    if expected_b8_misses is None:
        return
    for request_ordinal, req_idx in enumerate(request_ids):
        key = (condition, int(cache_budget), int(req_idx))
        expected = expected_b8_misses.get(key)
        if expected is None:
            raise ValueError(f"B8 per-request miss CSV is missing key={key!r}")
        observed = int(observed_per_request_misses[request_ordinal])
        if observed != expected:
            raise ValueError(
                "B9 replay diverged from B8 per-request misses: "
                f"condition={condition}, budget={cache_budget}, req_idx={req_idx}, "
                f"expected={expected}, observed={observed}"
            )


def run_latency_proxy_analysis(
    joined_indep_path: Path,
    joined_corr_path: Path,
    qc_report_path: Path,
    cache_curve_path: Path,
    output_dir: Path,
    router_request_log_path: Path,
    seed_config: Optional[Mapping[str, object]] = None,
    per_request_miss_path: Optional[Path] = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY_ROWS,
    tail_quantile: float = DEFAULT_TAIL_QUANTILE,
    args: Optional[argparse.Namespace] = None,
    config: Optional[Mapping[str, object]] = None,
) -> dict:
    import orjson

    if args is None:
        raise ValueError("args must be provided so model parameters are logged exactly")
    if config is None:
        config = {}
    if tail_quantile <= 0.0 or tail_quantile >= 1.0:
        raise ValueError(f"tail_quantile must be between 0 and 1, got {tail_quantile}")

    qc_payload = orjson.loads(qc_report_path.read_bytes())
    if not isinstance(qc_payload, dict):
        raise ValueError(f"expected JSON object in {qc_report_path}")
    total_events = resolve_total_events(qc_payload)
    ensure_dir(output_dir)

    cache_budgets = load_cache_budget_grid_from_curve(cache_curve_path)
    expected_b8_misses = load_expected_per_request_misses(per_request_miss_path) if per_request_miss_path and per_request_miss_path.exists() else None

    stream_result = build_phase_aware_streams(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        total_events=total_events,
        progress_every=progress_every,
    )

    args.tail_quantile = tail_quantile
    model_parameters = resolve_model_parameters(
        args=args,
        config=config,
        stream_result=stream_result,
        request_log_path=router_request_log_path,
    )

    latency_quantile_rows: List[dict] = []
    tail_breakdown_rows: List[dict] = []
    phase_breakdown_rows: List[dict] = []
    per_condition_payloads: Dict[str, dict] = {}

    for condition in CONDITION_ORDER:
        print(f"replaying latency proxy for {condition} with policy={POLICY_LRU}")
        access_buffer = stream_result.condition_buffers[condition]
        budget_payloads: Dict[str, dict] = {}
        for cache_budget in cache_budgets:
            print(f"  replay budget={cache_budget}")
            summary = simulate_phase_replay(
                condition_buffer=access_buffer,
                phase_ids=stream_result.phase_ids,
                request_offsets=stream_result.request_offsets,
                cache_budget=int(cache_budget),
            )
            validate_against_b8_misses(
                condition=condition,
                cache_budget=int(cache_budget),
                request_ids=stream_result.request_ids,
                observed_per_request_misses=summary.per_request_misses,
                expected_b8_misses=expected_b8_misses,
            )
            per_request_rows, quantiles, tail_rows, phase_rows = compute_latency_proxy_rows(
                condition=condition,
                cache_budget=int(cache_budget),
                request_ids=stream_result.request_ids,
                summary=summary,
                per_request_prefill_events=stream_result.per_request_prefill_events,
                per_request_decode_events=stream_result.per_request_decode_events,
                model_parameters=model_parameters,
                phase_available=stream_result.phase_available,
            )
            latency_quantile_rows.append(
                {
                    "condition": CONDITION_LABELS[condition],
                    "cache_budget": int(cache_budget),
                    "mean": quantiles["mean"],
                    "p50": quantiles["p50"],
                    "p95": quantiles["p95"],
                    "p99": quantiles["p99"],
                    "max": quantiles["max"],
                }
            )
            tail_breakdown_rows.extend(tail_rows)
            phase_breakdown_rows.extend(phase_rows)
            budget_payloads[str(cache_budget)] = {
                "cache_budget": int(cache_budget),
                "hits": int(summary.hits),
                "misses": int(summary.misses),
                "cold_misses": int(summary.cold_misses),
                "capacity_misses": int(summary.capacity_misses),
                "total_evictions": int(summary.total_evictions),
                "unique_evicted_objects": int(summary.unique_evicted_objects),
                "mean_residency_if_available": summary.mean_residency_if_available,
                "latency_proxy_summary": quantiles,
                "tail_latency_threshold": float(percentile([row["latency_proxy"] for row in per_request_rows], tail_quantile)),
                "tail_request_count": len(tail_rows),
                "per_request_latency_proxy": per_request_rows,
            }

        per_condition_payloads[condition] = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "object_key_definition": OBJECT_KEY_DEFINITIONS[condition],
            "object_key_note": OBJECT_KEY_NOTES[condition],
            "total_events": access_buffer.total_events,
            "max_object_id": access_buffer.max_object_id,
            "budgets": budget_payloads,
        }
        stream_result.condition_buffers[condition].access_ids = np.empty(0, dtype=np.uint32)
        gc.collect()

    outputs = {
        "latency_proxy_json": str(output_dir / "latency_proxy.json"),
        "latency_quantiles_csv": str(output_dir / "latency_quantiles.csv"),
        "tail_request_breakdown_csv": str(output_dir / "tail_request_breakdown.csv"),
        "prefill_decode_breakdown_csv": str(output_dir / "prefill_decode_breakdown.csv"),
    }

    result = {
        "model_name": "case_study_latency_proxy_v1",
        "goal": "Relative comparison of B0/B1/B2 replay conditions; not an absolute production latency predictor.",
        "cache_policy": POLICY_LRU,
        "cache_budget_grid": [int(value) for value in cache_budgets],
        "condition_labels": CONDITION_LABELS,
        "phase_aware": bool(stream_result.phase_available),
        "tail_definition": {
            "quantile": float(tail_quantile),
            "rule": (
                "tail_request_breakdown.csv contains requests where latency_proxy >= "
                "the configured per-condition per-budget tail quantile threshold"
            ),
        },
        "formula": (
            "latency_proxy = base_request_ms + prefill_event_cost_ms * prefill_events + "
            "decode_event_cost_ms * decode_events + hit_cost_ms * hit_count + "
            "miss_penalty_prefill_ms * prefill_miss_count + miss_penalty_decode_ms * decode_miss_count"
            if stream_result.phase_available
            else
            "latency_proxy = base_request_ms + event_cost_ms * total_events + hit_cost_ms * hit_count + miss_penalty_ms * miss_count"
        ),
        "parameters": model_parameters,
        "request_count": len(stream_result.request_ids),
        "request_ids": [int(req_idx) for req_idx in stream_result.request_ids],
        "total_events": int(total_events),
        "request_phase_event_summary": {
            "prefill_events_total": int(np.sum(stream_result.per_request_prefill_events)),
            "decode_events_total": int(np.sum(stream_result.per_request_decode_events)),
            "prefill_events_mean": float(np.mean(stream_result.per_request_prefill_events)) if stream_result.request_ids else 0.0,
            "decode_events_mean": float(np.mean(stream_result.per_request_decode_events)) if stream_result.request_ids else 0.0,
        },
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
            "cache_curve_path": str(cache_curve_path),
            "per_request_miss_path": (str(per_request_miss_path) if per_request_miss_path else None),
            "router_request_log_path": str(router_request_log_path),
        },
        "conditions": per_condition_payloads,
        "outputs": outputs,
    }

    write_json(output_dir / "latency_proxy.json", result)
    write_csv(output_dir / "latency_quantiles.csv", LATENCY_QUANTILE_FIELDS, latency_quantile_rows)
    write_csv(output_dir / "tail_request_breakdown.csv", TAIL_BREAKDOWN_FIELDS, tail_breakdown_rows)
    if stream_result.phase_available:
        write_csv(output_dir / "prefill_decode_breakdown.csv", PHASE_BREAKDOWN_FIELDS, phase_breakdown_rows)

    return result


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config()

    joined_dir = stage_output_dir("joined_trace", config, args.run_id)
    cache_dir = stage_output_dir("replay/cache", config, args.run_id)
    router_trace_dir = stage_output_dir("router_trace", config, args.run_id)
    output_dir = stage_output_dir("replay/latency", config, args.run_id, args.output_dir)

    joined_indep_path = Path(args.joined_indep_path) if args.joined_indep_path else joined_dir / "joined_trace_indep.jsonl"
    joined_corr_path = Path(args.joined_corr_path) if args.joined_corr_path else joined_dir / "joined_trace_corr.jsonl"
    qc_report_path = Path(args.qc_report_path) if args.qc_report_path else joined_dir / "join_qc_report.json"
    cache_curve_path = Path(args.cache_curve_path) if args.cache_curve_path else cache_dir / "cache_curve.csv"
    per_request_miss_path = (
        Path(args.per_request_miss_path) if args.per_request_miss_path else cache_dir / "per_request_miss_count.csv"
    )
    router_request_log_path = (
        Path(args.router_request_log_path) if args.router_request_log_path else router_trace_dir / "router_request_log.jsonl"
    )

    result = run_latency_proxy_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        cache_curve_path=cache_curve_path,
        output_dir=output_dir,
        router_request_log_path=router_request_log_path,
        seed_config=seeds,
        per_request_miss_path=per_request_miss_path,
        progress_every=args.progress_every,
        tail_quantile=args.tail_quantile,
        args=args,
        config=config,
    )

    wrote_outputs = [
        result["outputs"]["latency_proxy_json"],
        result["outputs"]["latency_quantiles_csv"],
        result["outputs"]["tail_request_breakdown_csv"],
    ]
    if result["phase_aware"]:
        wrote_outputs.append(result["outputs"]["prefill_decode_breakdown_csv"])
    print("wrote latency proxy outputs: " + ", ".join(wrote_outputs))


if __name__ == "__main__":
    main()
