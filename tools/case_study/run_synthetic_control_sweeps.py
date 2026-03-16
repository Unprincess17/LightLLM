#!/usr/bin/env python3
"""Run B10 synthetic control sweeps on top of the case-study replay core."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from analyze_latency_proxy import (
    CONDITION_LABELS,
    DEFAULT_PREFILL_MISS_RATIO,
    DEFAULT_TAIL_QUANTILE,
    compute_latency_proxy_rows,
    fit_base_compute_model,
    load_request_latencies,
)
from analyze_locality import (
    ADAPTER_SLOT_STRIDE,
    CONDITION_EXPERT_ONLY,
    CONDITION_JOINT_CORR,
    CONDITION_JOINT_INDEP,
    iter_jsonl_bytes,
    pack_expert_object,
)
from common import (
    artifact_root,
    ensure_dir,
    load_global_config,
    load_seed_config,
    stable_hash_int,
    write_csv,
    write_json,
)
from replay_core import (
    ConditionAccessBuffer,
    PHASE_DECODE,
    PHASE_PREFILL,
    PHASE_UNKNOWN,
    simulate_phase_replay,
)


DEFAULT_TEMPLATE_REQUEST_COUNT = 128
DEFAULT_SYNTHETIC_REQUEST_COUNT = 192
DEFAULT_NUM_CLASSES = 8
DEFAULT_NUM_LORAS = (8, 32, 64, 128)
DEFAULT_SKEW_LEVELS = (0.0, 0.8, 1.4)
DEFAULT_BURSTINESS_LEVELS = (1.0, 2.5, 6.0)
DEFAULT_CORR_LEVELS = (0.0, 0.5, 0.95)
DEFAULT_CACHE_BUDGETS = (256, 1024, 4096)
DEFAULT_PRIMARY_NUM_LORAS = 32
DEFAULT_PRIMARY_SKEW = 0.8
DEFAULT_PRIMARY_BURSTINESS = 2.5
DEFAULT_PRIMARY_CORR = 0.5
RESULT_FIELDS = [
    "run_id",
    "condition",
    "num_loras",
    "skew",
    "burstiness",
    "corr_strength",
    "cache_budget",
    "miss_rate",
    "p95",
    "p99",
]


@dataclass(frozen=True)
class RequestTemplate:
    template_req_idx: int
    expert_access_ids: np.ndarray
    phase_ids: np.ndarray
    prefill_events: int
    decode_events: int
    request_class: int


@dataclass(frozen=True)
class SyntheticBaseStream:
    request_ids: List[int]
    request_template_ids: np.ndarray
    request_classes: np.ndarray
    request_offsets: np.ndarray
    request_lengths: np.ndarray
    expert_access_ids: np.ndarray
    expert_access_ids_shifted: np.ndarray
    phase_ids: np.ndarray
    per_request_prefill_events: np.ndarray
    per_request_decode_events: np.ndarray
    total_events: int
    max_expert_object_id: int


@dataclass(frozen=True)
class SweepPoint:
    family: str
    run_id: str
    num_loras: int
    skew: float
    burstiness: float
    corr_strength: float
    cache_budget: int
    indep_seed: int
    corr_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B10 synthetic-control replay sweeps")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--seeds", type=str, default=None, help="Path to configs/seeds.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Base case-study run id used for source artifacts")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for sweep artifacts (default: artifacts/case_study/<run_id>/sweeps)",
    )
    parser.add_argument(
        "--joined_indep_path",
        type=str,
        default=None,
        help="Override joined_trace_indep.jsonl used to build request templates",
    )
    parser.add_argument(
        "--router_request_log_path",
        type=str,
        default=None,
        help="Override router_request_log.jsonl used to fit latency-proxy coefficients",
    )
    parser.add_argument(
        "--template_request_count",
        type=int,
        default=DEFAULT_TEMPLATE_REQUEST_COUNT,
        help="How many canonical joined requests to load as replay templates",
    )
    parser.add_argument(
        "--synthetic_request_count",
        type=int,
        default=DEFAULT_SYNTHETIC_REQUEST_COUNT,
        help="How many synthetic requests to replay per sweep point",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help="Class buckets used for correlation-aware adapter assignment",
    )
    parser.add_argument(
        "--num_loras",
        type=str,
        default=",".join(str(value) for value in DEFAULT_NUM_LORAS),
        help="Comma-separated LoRA-count sweep",
    )
    parser.add_argument(
        "--skew_levels",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SKEW_LEVELS),
        help="Comma-separated adapter-skew levels (Zipf exponent; 0 means uniform)",
    )
    parser.add_argument(
        "--burstiness_levels",
        type=str,
        default=",".join(str(value) for value in DEFAULT_BURSTINESS_LEVELS),
        help="Comma-separated mean adapter run lengths",
    )
    parser.add_argument(
        "--corr_strength_levels",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CORR_LEVELS),
        help="Comma-separated correlation strengths in [0, 1]",
    )
    parser.add_argument(
        "--cache_budgets",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CACHE_BUDGETS),
        help="Comma-separated cache budgets in object slots",
    )
    parser.add_argument(
        "--tail_quantile",
        type=float,
        default=DEFAULT_TAIL_QUANTILE,
        help="Tail threshold quantile used to compute per-point p95/p99 summaries",
    )
    return parser.parse_args()


def parse_float_grid(raw: str) -> List[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def parse_int_grid(raw: str) -> List[int]:
    return [int(token.strip()) for token in str(raw).split(",") if token.strip()]


def phase_name_to_id(phase: object) -> int:
    phase_name = str(phase or "").strip().lower()
    if phase_name == "prefill":
        return PHASE_PREFILL
    if phase_name == "decode":
        return PHASE_DECODE
    return PHASE_UNKNOWN


def float_token(value: float) -> str:
    token = f"{float(value):.2f}"
    return token.replace("-", "m").replace(".", "p")


def bounded_seed(prefix: str, token: str, base_seed: int) -> int:
    return int(stable_hash_int(f"{prefix}|{token}", base_seed) % (2**31 - 1))


def load_request_templates(
    joined_indep_path: Path,
    request_limit: int,
    num_classes: int,
) -> List[RequestTemplate]:
    if request_limit <= 0:
        raise ValueError("template_request_count must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    templates: List[RequestTemplate] = []
    current_req_idx: Optional[int] = None
    current_experts: List[int] = []
    current_phases: List[int] = []
    current_histogram: Dict[int, int] = {}
    current_prefill_events = 0
    current_decode_events = 0

    def flush_current() -> None:
        nonlocal current_req_idx, current_experts, current_phases
        nonlocal current_histogram, current_prefill_events, current_decode_events
        if current_req_idx is None:
            return
        if not current_experts:
            raise ValueError(f"request template {current_req_idx} is empty")
        dominant_expert = max(current_histogram.items(), key=lambda item: (item[1], -item[0]))[0]
        templates.append(
            RequestTemplate(
                template_req_idx=int(current_req_idx),
                expert_access_ids=np.asarray(current_experts, dtype=np.uint32),
                phase_ids=np.asarray(current_phases, dtype=np.uint8),
                prefill_events=int(current_prefill_events),
                decode_events=int(current_decode_events),
                request_class=int(dominant_expert % num_classes),
            )
        )
        current_req_idx = None
        current_experts = []
        current_phases = []
        current_histogram = {}
        current_prefill_events = 0
        current_decode_events = 0

    for row in iter_jsonl_bytes(joined_indep_path):
        req_idx = int(row["req_idx"])
        if req_idx >= request_limit:
            break
        if current_req_idx is None:
            current_req_idx = req_idx
        elif req_idx != current_req_idx:
            flush_current()
            current_req_idx = req_idx

        expert_object_id = pack_expert_object(int(row["layer_id"]), int(row["expert_id"]))
        phase_id = phase_name_to_id(row.get("phase"))
        current_experts.append(expert_object_id)
        current_phases.append(phase_id)
        current_histogram[expert_object_id] = current_histogram.get(expert_object_id, 0) + 1
        if phase_id == PHASE_PREFILL:
            current_prefill_events += 1
        elif phase_id == PHASE_DECODE:
            current_decode_events += 1

    flush_current()
    if not templates:
        raise ValueError(f"failed to load any request templates from {joined_indep_path}")
    return templates


def build_synthetic_schedule(template_count: int, synthetic_request_count: int, seed: int) -> np.ndarray:
    if synthetic_request_count <= 0:
        raise ValueError("synthetic_request_count must be positive")
    if template_count <= 0:
        raise ValueError("template_count must be positive")
    repeats = int(math.ceil(float(synthetic_request_count) / float(template_count)))
    schedule = np.tile(np.arange(template_count, dtype=np.int32), repeats)
    rng = np.random.default_rng(seed)
    rng.shuffle(schedule)
    return schedule[:synthetic_request_count]


def build_base_stream(templates: Sequence[RequestTemplate], schedule: np.ndarray) -> SyntheticBaseStream:
    request_count = int(schedule.shape[0])
    request_ids = [int(req_idx) for req_idx in range(request_count)]
    request_template_ids = np.asarray(schedule, dtype=np.int32)
    request_classes = np.asarray([templates[int(template_idx)].request_class for template_idx in schedule], dtype=np.int16)
    request_lengths = np.asarray(
        [int(templates[int(template_idx)].expert_access_ids.shape[0]) for template_idx in schedule],
        dtype=np.int64,
    )
    request_offsets = np.zeros(request_count + 1, dtype=np.int64)
    request_offsets[1:] = np.cumsum(request_lengths, dtype=np.int64)
    total_events = int(request_offsets[-1])
    if total_events <= 0:
        raise ValueError("synthetic schedule produced zero replay events")

    expert_access_ids = np.empty(total_events, dtype=np.uint32)
    phase_ids = np.empty(total_events, dtype=np.uint8)
    per_request_prefill_events = np.empty(request_count, dtype=np.int64)
    per_request_decode_events = np.empty(request_count, dtype=np.int64)

    cursor = 0
    max_expert_object_id = 0
    for request_ordinal, template_idx in enumerate(schedule):
        template = templates[int(template_idx)]
        event_count = int(template.expert_access_ids.shape[0])
        next_cursor = cursor + event_count
        expert_access_ids[cursor:next_cursor] = template.expert_access_ids
        phase_ids[cursor:next_cursor] = template.phase_ids
        per_request_prefill_events[request_ordinal] = int(template.prefill_events)
        per_request_decode_events[request_ordinal] = int(template.decode_events)
        if event_count > 0:
            max_expert_object_id = max(max_expert_object_id, int(np.max(template.expert_access_ids)))
        cursor = next_cursor

    return SyntheticBaseStream(
        request_ids=request_ids,
        request_template_ids=request_template_ids,
        request_classes=request_classes,
        request_offsets=request_offsets,
        request_lengths=request_lengths,
        expert_access_ids=expert_access_ids,
        expert_access_ids_shifted=(expert_access_ids.astype(np.uint64) * ADAPTER_SLOT_STRIDE).astype(np.uint32),
        phase_ids=phase_ids,
        per_request_prefill_events=per_request_prefill_events,
        per_request_decode_events=per_request_decode_events,
        total_events=total_events,
        max_expert_object_id=int(max_expert_object_id),
    )


def make_rank_weights(count: int, skew: float) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    if skew <= 0.0:
        return np.full(count, 1.0 / float(count), dtype=np.float64)
    ranks = np.arange(1, count + 1, dtype=np.float64)
    weights = np.power(ranks, -float(skew))
    return weights / np.sum(weights)


def sample_run_lengths(request_count: int, burstiness: float, rng: np.random.Generator) -> np.ndarray:
    if burstiness <= 1.0:
        return np.ones(request_count, dtype=np.int32)
    probability = min(max(1.0 / float(burstiness), 1e-6), 1.0)
    return rng.geometric(probability, size=request_count).astype(np.int32)


def class_local_pool(class_id: int, num_loras: int, num_classes: int) -> np.ndarray:
    pool_size = max(1, int(math.ceil(float(num_loras) / float(max(num_classes, 1)))))
    start = int((class_id * pool_size) % num_loras)
    pool = np.asarray([(start + offset) % num_loras for offset in range(min(pool_size, num_loras))], dtype=np.int32)
    return np.unique(pool)


def generate_adapter_slots(
    request_classes: np.ndarray,
    num_loras: int,
    skew: float,
    burstiness: float,
    corr_strength: float,
    num_classes: int,
    seed: int,
    correlated: bool,
) -> np.ndarray:
    if num_loras <= 0 or num_loras >= ADAPTER_SLOT_STRIDE:
        raise ValueError(
            f"num_loras must be in [1, {ADAPTER_SLOT_STRIDE - 1}] so object-key packing stays valid, got {num_loras}"
        )
    rng = np.random.default_rng(seed)
    run_lengths = sample_run_lengths(int(request_classes.shape[0]), burstiness, rng)
    global_weights = make_rank_weights(num_loras, skew)
    slot_ids = np.arange(num_loras, dtype=np.int32)
    pool_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def sample_slot(class_id: int) -> int:
        if correlated and corr_strength > 0.0 and rng.random() < corr_strength:
            cached = pool_cache.get(int(class_id))
            if cached is None:
                pool_slots = class_local_pool(int(class_id), num_loras, num_classes)
                pool_weights = global_weights[pool_slots.astype(np.int64)]
                pool_weights = pool_weights / np.sum(pool_weights)
                cached = (pool_slots, pool_weights)
                pool_cache[int(class_id)] = cached
            return int(rng.choice(cached[0], p=cached[1]))
        return int(rng.choice(slot_ids, p=global_weights))

    slots = np.empty(int(request_classes.shape[0]), dtype=np.uint16)
    session_remaining = 0
    current_slot = 0
    for request_ordinal, class_id in enumerate(request_classes.tolist()):
        if session_remaining <= 0:
            current_slot = sample_slot(int(class_id))
            session_remaining = int(run_lengths[request_ordinal])
        slots[request_ordinal] = np.uint16(current_slot)
        session_remaining -= 1
    return slots


def build_joint_access_ids(base_stream: SyntheticBaseStream, slots: np.ndarray) -> np.ndarray:
    slot_vector = np.repeat(slots.astype(np.uint32), base_stream.request_lengths.astype(np.int64))
    return (base_stream.expert_access_ids_shifted + slot_vector).astype(np.uint32)


def fit_model_parameters(
    templates: Sequence[RequestTemplate],
    request_latencies_ms: Mapping[int, float],
    tail_quantile: float,
    prefill_miss_ratio: float,
) -> dict:
    request_ids = [int(template.template_req_idx) for template in templates]
    fitted = fit_base_compute_model(
        request_ids=request_ids,
        per_request_prefill_events=np.asarray([template.prefill_events for template in templates], dtype=np.int64),
        per_request_decode_events=np.asarray([template.decode_events for template in templates], dtype=np.int64),
        request_latencies_ms=request_latencies_ms,
        base_request_ms_override=None,
        prefill_event_cost_ms_override=None,
        decode_event_cost_ms_override=None,
    )
    prefill_event_cost_ms = float(fitted["prefill_event_cost_ms"])
    decode_event_cost_ms = float(fitted["decode_event_cost_ms"])
    return {
        "phase_aware": True,
        "tail_quantile": float(tail_quantile),
        "base_request_ms": float(fitted["base_request_ms"]),
        "prefill_event_cost_ms": prefill_event_cost_ms,
        "decode_event_cost_ms": decode_event_cost_ms,
        "event_cost_ms": float((prefill_event_cost_ms + decode_event_cost_ms) / 2.0),
        "hit_cost_ms": 0.0,
        "miss_penalty_ms": float(decode_event_cost_ms),
        "miss_penalty_prefill_ms": float(max(prefill_event_cost_ms, prefill_miss_ratio * decode_event_cost_ms)),
        "miss_penalty_decode_ms": float(decode_event_cost_ms),
        "prefill_miss_ratio": float(prefill_miss_ratio),
        "base_compute_fit": fitted,
    }


def evaluate_condition(
    condition: str,
    access_ids: np.ndarray,
    base_stream: SyntheticBaseStream,
    cache_budgets: Sequence[int],
    model_parameters: Mapping[str, object],
) -> Dict[int, dict]:
    access_buffer = ConditionAccessBuffer(
        condition=condition,
        access_ids=access_ids,
        total_events=int(base_stream.total_events),
        max_object_id=int(np.max(access_ids)) if access_ids.size else 0,
    )
    metrics_by_budget: Dict[int, dict] = {}
    for cache_budget in cache_budgets:
        summary = simulate_phase_replay(
            condition_buffer=access_buffer,
            phase_ids=base_stream.phase_ids,
            request_offsets=base_stream.request_offsets,
            cache_budget=int(cache_budget),
        )
        _per_request_rows, quantiles, _tail_rows, _phase_rows = compute_latency_proxy_rows(
            condition=condition,
            cache_budget=int(cache_budget),
            request_ids=base_stream.request_ids,
            summary=summary,
            per_request_prefill_events=base_stream.per_request_prefill_events,
            per_request_decode_events=base_stream.per_request_decode_events,
            model_parameters=model_parameters,
            phase_available=True,
        )
        metrics_by_budget[int(cache_budget)] = {
            "miss_rate": float(summary.misses / summary.total_events) if summary.total_events else 0.0,
            "p95": float(quantiles["p95"]),
            "p99": float(quantiles["p99"]),
        }
    return metrics_by_budget


def build_sweep_points(
    num_loras: Sequence[int],
    skew_levels: Sequence[float],
    burstiness_levels: Sequence[float],
    corr_levels: Sequence[float],
    cache_budgets: Sequence[int],
    base_seed: int,
) -> List[SweepPoint]:
    points: List[SweepPoint] = []
    seen_run_ids = set()

    def append_point(family: str, num_loras_value: int, skew_value: float, burstiness_value: float, corr_value: float, cache_budget: int) -> None:
        token = (
            f"family={family}|nl={num_loras_value}|sk={skew_value:.4f}|"
            f"bu={burstiness_value:.4f}|co={corr_value:.4f}|cb={cache_budget}"
        )
        indep_seed = bounded_seed("indep", token, base_seed)
        corr_seed = bounded_seed("corr", token, base_seed)
        run_id = (
            f"{family}_nl{num_loras_value:03d}_sk{float_token(skew_value)}_"
            f"bu{float_token(burstiness_value)}_co{float_token(corr_value)}_cb{cache_budget:05d}"
        )
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate sweep run_id generated: {run_id}")
        seen_run_ids.add(run_id)
        points.append(
            SweepPoint(
                family=family,
                run_id=run_id,
                num_loras=int(num_loras_value),
                skew=float(skew_value),
                burstiness=float(burstiness_value),
                corr_strength=float(corr_value),
                cache_budget=int(cache_budget),
                indep_seed=indep_seed,
                corr_seed=corr_seed,
            )
        )

    for num_loras_value in num_loras:
        for cache_budget in cache_budgets:
            append_point(
                family="num_loras",
                num_loras_value=int(num_loras_value),
                skew_value=DEFAULT_PRIMARY_SKEW,
                burstiness_value=DEFAULT_PRIMARY_BURSTINESS,
                corr_value=DEFAULT_PRIMARY_CORR,
                cache_budget=int(cache_budget),
            )

    for skew_value in skew_levels:
        for burstiness_value in burstiness_levels:
            for corr_value in corr_levels:
                for cache_budget in cache_budgets:
                    append_point(
                        family="grid",
                        num_loras_value=DEFAULT_PRIMARY_NUM_LORAS,
                        skew_value=float(skew_value),
                        burstiness_value=float(burstiness_value),
                        corr_value=float(corr_value),
                        cache_budget=int(cache_budget),
                    )

    return points


def summarize_p99_drivers(
    result_rows: Sequence[Mapping[str, object]],
    cache_budgets: Sequence[int],
) -> List[dict]:
    rows = [dict(row) for row in result_rows]
    primary_budget = int(cache_budgets[min(1, len(cache_budgets) - 1)])
    joint_conditions = (CONDITION_JOINT_INDEP, CONDITION_JOINT_CORR)
    summaries: List[dict] = []

    def p99_range(filtered_rows: Sequence[Mapping[str, object]]) -> Tuple[float, List[dict]]:
        per_condition = []
        for condition in joint_conditions:
            condition_rows = [row for row in filtered_rows if row["condition"] == condition]
            if not condition_rows:
                continue
            p99_values = [float(row["p99"]) for row in condition_rows]
            per_condition.append(
                {
                    "condition": condition,
                    "min_p99": min(p99_values),
                    "max_p99": max(p99_values),
                    "delta_p99": max(p99_values) - min(p99_values),
                }
            )
        mean_delta = float(sum(item["delta_p99"] for item in per_condition) / len(per_condition)) if per_condition else 0.0
        return mean_delta, per_condition

    analyses = [
        (
            "cache_budget",
            [
                row
                for row in rows
                if row["num_loras"] == DEFAULT_PRIMARY_NUM_LORAS
                and abs(float(row["skew"]) - DEFAULT_PRIMARY_SKEW) < 1e-9
                and abs(float(row["burstiness"]) - DEFAULT_PRIMARY_BURSTINESS) < 1e-9
                and abs(float(row["corr_strength"]) - DEFAULT_PRIMARY_CORR) < 1e-9
            ],
        ),
        (
            "num_loras",
            [
                row
                for row in rows
                if int(row["cache_budget"]) == primary_budget
                and abs(float(row["skew"]) - DEFAULT_PRIMARY_SKEW) < 1e-9
                and abs(float(row["burstiness"]) - DEFAULT_PRIMARY_BURSTINESS) < 1e-9
                and abs(float(row["corr_strength"]) - DEFAULT_PRIMARY_CORR) < 1e-9
            ],
        ),
        (
            "skew",
            [
                row
                for row in rows
                if int(row["num_loras"]) == DEFAULT_PRIMARY_NUM_LORAS
                and int(row["cache_budget"]) == primary_budget
                and abs(float(row["burstiness"]) - DEFAULT_PRIMARY_BURSTINESS) < 1e-9
                and abs(float(row["corr_strength"]) - DEFAULT_PRIMARY_CORR) < 1e-9
            ],
        ),
        (
            "burstiness",
            [
                row
                for row in rows
                if int(row["num_loras"]) == DEFAULT_PRIMARY_NUM_LORAS
                and int(row["cache_budget"]) == primary_budget
                and abs(float(row["skew"]) - DEFAULT_PRIMARY_SKEW) < 1e-9
                and abs(float(row["corr_strength"]) - DEFAULT_PRIMARY_CORR) < 1e-9
            ],
        ),
        (
            "corr_strength",
            [
                row
                for row in rows
                if int(row["num_loras"]) == DEFAULT_PRIMARY_NUM_LORAS
                and int(row["cache_budget"]) == primary_budget
                and abs(float(row["skew"]) - DEFAULT_PRIMARY_SKEW) < 1e-9
                and abs(float(row["burstiness"]) - DEFAULT_PRIMARY_BURSTINESS) < 1e-9
            ],
        ),
    ]

    for knob_name, filtered_rows in analyses:
        mean_delta, per_condition = p99_range(filtered_rows)
        summaries.append(
            {
                "knob": knob_name,
                "primary_cache_budget": primary_budget,
                "mean_joint_delta_p99": mean_delta,
                "per_condition": per_condition,
            }
        )
    summaries.sort(key=lambda item: (-float(item["mean_joint_delta_p99"]), str(item["knob"])))
    return summaries


def run_synthetic_control_sweeps(
    joined_indep_path: Path,
    router_request_log_path: Path,
    output_dir: Path,
    template_request_count: int,
    synthetic_request_count: int,
    num_classes: int,
    num_loras: Sequence[int],
    skew_levels: Sequence[float],
    burstiness_levels: Sequence[float],
    corr_levels: Sequence[float],
    cache_budgets: Sequence[int],
    tail_quantile: float,
    config: Mapping[str, object],
    seeds: Mapping[str, object],
    source_run_id: str,
) -> dict:
    ensure_dir(output_dir)

    template_seed = int(seeds.get("synthetic_trace_seed", seeds.get("global_seed", 7)))
    points_seed = int(seeds.get("replay_seed", seeds.get("global_seed", 7)))

    templates = load_request_templates(
        joined_indep_path=joined_indep_path,
        request_limit=template_request_count,
        num_classes=num_classes,
    )
    schedule = build_synthetic_schedule(len(templates), synthetic_request_count, template_seed)
    base_stream = build_base_stream(templates, schedule)

    request_latencies_ms = load_request_latencies(router_request_log_path)
    model_parameters = fit_model_parameters(
        templates=templates,
        request_latencies_ms=request_latencies_ms,
        tail_quantile=tail_quantile,
        prefill_miss_ratio=DEFAULT_PREFILL_MISS_RATIO,
    )

    points = build_sweep_points(
        num_loras=num_loras,
        skew_levels=skew_levels,
        burstiness_levels=burstiness_levels,
        corr_levels=corr_levels,
        cache_budgets=cache_budgets,
        base_seed=points_seed,
    )

    unique_budgets = sorted({int(point.cache_budget) for point in points})
    expert_metrics = evaluate_condition(
        condition=CONDITION_EXPERT_ONLY,
        access_ids=base_stream.expert_access_ids,
        base_stream=base_stream,
        cache_budgets=unique_budgets,
        model_parameters=model_parameters,
    )

    indep_metric_cache: Dict[Tuple[int, float, float], Dict[int, dict]] = {}
    corr_metric_cache: Dict[Tuple[int, float, float, float], Dict[int, dict]] = {}

    result_rows: List[dict] = []
    run_manifest_rows: List[dict] = []

    for point in points:
        indep_key = (int(point.num_loras), float(point.skew), float(point.burstiness))
        if indep_key not in indep_metric_cache:
            indep_slots = generate_adapter_slots(
                request_classes=base_stream.request_classes,
                num_loras=int(point.num_loras),
                skew=float(point.skew),
                burstiness=float(point.burstiness),
                corr_strength=0.0,
                num_classes=num_classes,
                seed=int(point.indep_seed),
                correlated=False,
            )
            indep_metric_cache[indep_key] = evaluate_condition(
                condition=CONDITION_JOINT_INDEP,
                access_ids=build_joint_access_ids(base_stream, indep_slots),
                base_stream=base_stream,
                cache_budgets=unique_budgets,
                model_parameters=model_parameters,
            )

        corr_key = (int(point.num_loras), float(point.skew), float(point.burstiness), float(point.corr_strength))
        if corr_key not in corr_metric_cache:
            corr_slots = generate_adapter_slots(
                request_classes=base_stream.request_classes,
                num_loras=int(point.num_loras),
                skew=float(point.skew),
                burstiness=float(point.burstiness),
                corr_strength=float(point.corr_strength),
                num_classes=num_classes,
                seed=int(point.corr_seed),
                correlated=True,
            )
            corr_metric_cache[corr_key] = evaluate_condition(
                condition=CONDITION_JOINT_CORR,
                access_ids=build_joint_access_ids(base_stream, corr_slots),
                base_stream=base_stream,
                cache_budgets=unique_budgets,
                model_parameters=model_parameters,
            )

        run_manifest_rows.append(
            {
                "family": point.family,
                "run_id": point.run_id,
                "num_loras": int(point.num_loras),
                "skew": float(point.skew),
                "burstiness": float(point.burstiness),
                "corr_strength": float(point.corr_strength),
                "cache_budget": int(point.cache_budget),
                "indep_seed": int(point.indep_seed),
                "corr_seed": int(point.corr_seed),
            }
        )

        point_metrics = {
            CONDITION_EXPERT_ONLY: expert_metrics[int(point.cache_budget)],
            CONDITION_JOINT_INDEP: indep_metric_cache[indep_key][int(point.cache_budget)],
            CONDITION_JOINT_CORR: corr_metric_cache[corr_key][int(point.cache_budget)],
        }
        for condition, metrics in point_metrics.items():
            result_rows.append(
                {
                    "run_id": point.run_id,
                    "condition": condition,
                    "num_loras": int(point.num_loras),
                    "skew": float(point.skew),
                    "burstiness": float(point.burstiness),
                    "corr_strength": float(point.corr_strength),
                    "cache_budget": int(point.cache_budget),
                    "miss_rate": float(metrics["miss_rate"]),
                    "p95": float(metrics["p95"]),
                    "p99": float(metrics["p99"]),
                }
            )

    result_rows.sort(
        key=lambda row: (
            str(row["run_id"]),
            str(row["condition"]),
        )
    )
    num_loras_rows = [
        {
            "condition": row["condition"],
            "num_loras": row["num_loras"],
            "cache_budget": row["cache_budget"],
            "p99": row["p99"],
        }
        for row in result_rows
        if any(point["run_id"] == row["run_id"] and point["family"] == "num_loras" for point in run_manifest_rows)
    ]
    grid_rows = [
        {
            "condition": row["condition"],
            "skew": row["skew"],
            "burstiness": row["burstiness"],
            "corr_strength": row["corr_strength"],
            "cache_budget": row["cache_budget"],
            "miss_rate": row["miss_rate"],
            "p99": row["p99"],
        }
        for row in result_rows
        if any(point["run_id"] == row["run_id"] and point["family"] == "grid" for point in run_manifest_rows)
    ]

    strongest_p99_drivers = summarize_p99_drivers(result_rows, cache_budgets)
    manifest = {
        "model_name": "case_study_synthetic_control_sweeps_v1",
        "goal": "B10 robustness sweeps over synthetic LoRA workloads while reusing the B8/B9 replay core.",
        "source_run_id": source_run_id,
        "inputs": {
            "joined_indep_path": str(joined_indep_path),
            "router_request_log_path": str(router_request_log_path),
        },
        "template_config": {
            "template_request_count": int(template_request_count),
            "synthetic_request_count": int(synthetic_request_count),
            "num_classes": int(num_classes),
            "schedule_seed": int(template_seed),
            "schedule_template_ordinals": [int(value) for value in schedule.tolist()],
            "template_request_ids": [int(template.template_req_idx) for template in templates],
            "template_request_classes": [int(template.request_class) for template in templates],
            "template_prefill_events": [int(template.prefill_events) for template in templates],
            "template_decode_events": [int(template.decode_events) for template in templates],
            "total_events_per_sweep": int(base_stream.total_events),
            "adapter_slot_stride": int(ADAPTER_SLOT_STRIDE),
        },
        "parameter_grid": {
            "num_loras": [int(value) for value in num_loras],
            "skew_levels": [float(value) for value in skew_levels],
            "burstiness_levels": [float(value) for value in burstiness_levels],
            "corr_strength_levels": [float(value) for value in corr_levels],
            "cache_budgets": [int(value) for value in cache_budgets],
            "primary_slice": {
                "num_loras": DEFAULT_PRIMARY_NUM_LORAS,
                "skew": DEFAULT_PRIMARY_SKEW,
                "burstiness": DEFAULT_PRIMARY_BURSTINESS,
                "corr_strength": DEFAULT_PRIMARY_CORR,
            },
        },
        "reproducibility": {
            "seed_config": dict(seeds),
            "run_points": run_manifest_rows,
            "adapter_assignment_rules": {
                "joint_indep": "Sample adapters from a global Zipf/uniform pool with burstiness-controlled session lengths.",
                "joint_corr": "At session boundaries, sample from a class-local adapter pool with probability corr_strength; otherwise use the global pool.",
            },
        },
        "latency_model_parameters": model_parameters,
        "condition_labels": dict(CONDITION_LABELS),
        "strongest_p99_drivers": strongest_p99_drivers,
        "outputs": {
            "sweep_manifest_json": str(output_dir / "sweep_manifest.json"),
            "sweep_results_csv": str(output_dir / "sweep_results.csv"),
            "num_loras_vs_p99_csv": str(output_dir / "num_loras_vs_p99.csv"),
            "skew_burst_corr_grid_csv": str(output_dir / "skew_burst_corr_grid.csv"),
        },
    }

    write_json(output_dir / "sweep_manifest.json", manifest)
    write_csv(output_dir / "sweep_results.csv", RESULT_FIELDS, result_rows)
    write_csv(output_dir / "num_loras_vs_p99.csv", ["condition", "num_loras", "cache_budget", "p99"], num_loras_rows)
    write_csv(
        output_dir / "skew_burst_corr_grid.csv",
        ["condition", "skew", "burstiness", "corr_strength", "cache_budget", "miss_rate", "p99"],
        grid_rows,
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config(args.seeds)

    source_run_id = str(args.run_id or config.get("case_study", {}).get("default_run_id", "router_lora_case_v1"))
    joined_indep_path = (
        Path(args.joined_indep_path)
        if args.joined_indep_path
        else artifact_root(config) / "case_study" / source_run_id / "joined_trace" / "joined_trace_indep.jsonl"
    )
    router_request_log_path = (
        Path(args.router_request_log_path)
        if args.router_request_log_path
        else artifact_root(config) / "case_study" / source_run_id / "router_trace" / "router_request_log.jsonl"
    )
    output_dir = (
        ensure_dir(Path(args.output_dir))
        if args.output_dir
        else ensure_dir(artifact_root(config) / "case_study" / source_run_id / "sweeps")
    )

    manifest = run_synthetic_control_sweeps(
        joined_indep_path=joined_indep_path,
        router_request_log_path=router_request_log_path,
        output_dir=output_dir,
        template_request_count=int(args.template_request_count),
        synthetic_request_count=int(args.synthetic_request_count),
        num_classes=int(args.num_classes),
        num_loras=parse_int_grid(args.num_loras),
        skew_levels=parse_float_grid(args.skew_levels),
        burstiness_levels=parse_float_grid(args.burstiness_levels),
        corr_levels=parse_float_grid(args.corr_strength_levels),
        cache_budgets=parse_int_grid(args.cache_budgets),
        tail_quantile=float(args.tail_quantile),
        config=config,
        seeds=seeds,
        source_run_id=source_run_id,
    )
    print(
        "wrote sweep artifacts: "
        f"{manifest['outputs']['sweep_manifest_json']}, "
        f"{manifest['outputs']['sweep_results_csv']}, "
        f"{manifest['outputs']['num_loras_vs_p99_csv']}, "
        f"{manifest['outputs']['skew_burst_corr_grid_csv']}"
    )


if __name__ == "__main__":
    main()
