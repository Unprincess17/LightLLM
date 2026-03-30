#!/usr/bin/env python3
"""Higher-level helpers for calibrated system-level TPOT simulation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import percentile
from replay_core import PhaseReplayBitmapSummary
from system_tpot_core import (
    CURVE_STAT_KEYS,
    CONDITION_LABELS,
    ComponentSizeBytes,
    LayerTransferDecision,
    MISS_HANDLING_EXECUTION_FIRST,
    MISS_HANDLING_LOAD_THEN_RUN,
    MISS_HANDLING_NO_CPU_PATH,
    MISS_HANDLING_NO_DEFERRED_SYNC,
    OVERLAP_POLICY_CALIBRATED,
    OVERLAP_POLICY_DISABLED,
    SchedulerBatchStats,
    TokenStructuredStream,
    TokenTemplate,
    _interp_from_rows,
    calibration_service_ms,
    summarize_values,
)


@dataclass(frozen=True)
class Stage1TPOTOutputs:
    token_rows: List[dict]
    quantiles: Dict[str, float]
    tail_rows: List[dict]
    layer_rows: List[dict]
    request_rows: List[dict]
    tail_threshold_ms: float


@dataclass(frozen=True)
class SchedulerSimulationSummary:
    summary_row: dict
    batch_stats: List[SchedulerBatchStats]


@dataclass
class _SchedulerRequestState:
    req_idx: int
    request_ordinal: int
    tokens: Sequence[TokenTemplate]
    next_token_index: int


def _interp_from_pairs(points: Sequence[Tuple[float, float]], x_value: float) -> float:
    if not points:
        raise ValueError("interpolation points must not be empty")
    ordered = sorted((float(x), float(y)) for x, y in points)
    first_x = ordered[0][0]
    last_x = ordered[-1][0]
    if x_value < first_x - 1e-9:
        raise ValueError(f"batch interpolation underflow: requested={x_value}, min={first_x}")
    if x_value > last_x + 1e-9:
        raise ValueError(f"batch interpolation overflow: requested={x_value}, max={last_x}")
    for index, (x_coord, y_coord) in enumerate(ordered):
        if abs(x_coord - x_value) <= 1e-9:
            return float(y_coord)
        if x_coord > x_value:
            lower_x, lower_y = ordered[index - 1]
            weight = (x_value - lower_x) / max(x_coord - lower_x, 1e-9)
            return float(lower_y) * (1.0 - weight) + float(y_coord) * weight
    return float(ordered[-1][1])


def _mapping_points(raw_mapping: Mapping[object, object]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for raw_key, raw_value in raw_mapping.items():
        points.append((float(raw_key), float(raw_value)))
    return points


def calibration_base_tpot_ms_interp(calibration: Mapping[str, object], system_batch: int) -> float:
    values = calibration.get("base_tpot_ms", {})
    if not isinstance(values, Mapping) or not values:
        raise ValueError("calibration is missing base_tpot_ms")
    return _interp_from_pairs(_mapping_points(values), float(system_batch))


def calibration_overlap_window_ms_interp(
    calibration: Mapping[str, object],
    system_batch: int,
    window_name: str,
    stat_key: str,
) -> float:
    if stat_key not in CURVE_STAT_KEYS:
        raise ValueError(f"unsupported curve stat key: {stat_key}")
    windows = calibration.get("overlap_windows_ms", {})
    if not isinstance(windows, Mapping) or not windows:
        raise ValueError("calibration is missing overlap_windows_ms")

    points: List[Tuple[float, float]] = []
    for raw_batch, payload in windows.items():
        if not isinstance(payload, Mapping):
            continue
        window_payload = payload.get(window_name)
        if not isinstance(window_payload, Mapping):
            continue
        if stat_key not in window_payload:
            raise ValueError(
                f"overlap window {window_name!r} for batch={raw_batch!r} is missing stat {stat_key!r}"
            )
        points.append((float(raw_batch), float(window_payload[stat_key])))
    if not points:
        raise ValueError(f"no overlap-window points found for window={window_name!r}")
    return _interp_from_pairs(points, float(system_batch))


def resolve_overlap_window_ms(
    calibration: Mapping[str, object],
    system_batch: int,
    window_name: str,
    stat_key: str,
    overlap_policy: str,
) -> float:
    if overlap_policy == OVERLAP_POLICY_DISABLED:
        return 0.0
    return calibration_overlap_window_ms_interp(calibration, system_batch, window_name, stat_key)


def calibration_cold_path_component_ms_interp(
    calibration: Mapping[str, object],
    load_profile: str,
    stage_name: str,
    component_name: str,
    x_key: str,
    x_value: float,
    stat_key: str,
) -> float:
    if stat_key not in CURVE_STAT_KEYS:
        raise ValueError(f"unsupported curve stat key: {stat_key}")
    if x_value <= 0.0:
        return 0.0
    cold_path_curves = calibration.get("cold_path_curves", {})
    if not isinstance(cold_path_curves, Mapping):
        raise ValueError("calibration is missing cold_path_curves")
    profiles = cold_path_curves.get("profiles", {})
    if not isinstance(profiles, Mapping) or load_profile not in profiles:
        raise ValueError(f"load profile {load_profile!r} is missing from cold_path_curves")
    profile_payload = profiles[load_profile]
    if not isinstance(profile_payload, Mapping):
        raise ValueError(f"unexpected cold_path profile payload for {load_profile!r}")
    stage_payload = profile_payload.get(stage_name)
    if not isinstance(stage_payload, Mapping):
        raise ValueError(f"cold-path stage {stage_name!r} is missing for profile {load_profile!r}")
    rows = stage_payload.get(component_name)
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"cold-path component {component_name!r} is missing for profile {load_profile!r}, stage {stage_name!r}"
        )
    return _interp_from_rows(rows, x_key, stat_key, x_value)


def choose_layer_transfer_strategy_interp(
    miss_objects: int,
    cold_miss_objects: int,
    object_sizes: ComponentSizeBytes,
    calibration: Mapping[str, object],
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    system_batch: int,
    overlap_policy: str = OVERLAP_POLICY_CALIBRATED,
) -> LayerTransferDecision:
    capacity_miss_objects = max(int(miss_objects) - int(cold_miss_objects), 0)
    early_overlap_window_ms = resolve_overlap_window_ms(
        calibration, system_batch, "early", stat_key, overlap_policy
    )
    late_overlap_window_ms = resolve_overlap_window_ms(
        calibration, system_batch, "late", stat_key, overlap_policy
    )

    if miss_objects <= 0:
        return LayerTransferDecision(
            strategy="hit_only",
            miss_objects=0,
            cold_miss_objects=0,
            capacity_miss_objects=0,
            early_bytes=0,
            late_bytes=0,
            total_bytes=0,
            early_slices=0,
            late_slices=0,
            total_slices=0,
            early_service_ms=0.0,
            late_service_ms=0.0,
            whole_service_ms=0.0,
            early_overlap_window_ms=early_overlap_window_ms,
            late_overlap_window_ms=late_overlap_window_ms,
            early_exposed_stall_ms=0.0,
            late_exposed_stall_ms=0.0,
            total_exposed_stall_ms=0.0,
        )

    early_bytes = int(miss_objects) * int(object_sizes.early_object_bytes)
    late_bytes = int(miss_objects) * int(object_sizes.late_object_bytes)
    total_bytes = int(miss_objects) * int(object_sizes.total_object_bytes)
    early_slices = int(math.ceil(float(early_bytes) / float(object_sizes.slice_bytes)))
    late_slices = int(math.ceil(float(late_bytes) / float(object_sizes.slice_bytes)))
    total_slices = int(math.ceil(float(total_bytes) / float(object_sizes.slice_bytes)))

    early_service_ms = calibration_service_ms(calibration, transfer_mode, load_profile, early_bytes, stat_key)
    late_service_ms = calibration_service_ms(calibration, transfer_mode, load_profile, late_bytes, stat_key)
    whole_service_ms = calibration_service_ms(calibration, transfer_mode, load_profile, total_bytes, stat_key)

    split_early_stall_ms = max(0.0, early_service_ms - early_overlap_window_ms)
    split_late_stall_ms = max(0.0, late_service_ms - late_overlap_window_ms)
    split_total_stall_ms = split_early_stall_ms + split_late_stall_ms
    whole_total_stall_ms = max(0.0, whole_service_ms - early_overlap_window_ms)

    if whole_total_stall_ms <= split_total_stall_ms:
        return LayerTransferDecision(
            strategy="whole_object_early",
            miss_objects=int(miss_objects),
            cold_miss_objects=int(cold_miss_objects),
            capacity_miss_objects=int(capacity_miss_objects),
            early_bytes=0,
            late_bytes=0,
            total_bytes=total_bytes,
            early_slices=0,
            late_slices=0,
            total_slices=total_slices,
            early_service_ms=0.0,
            late_service_ms=0.0,
            whole_service_ms=whole_service_ms,
            early_overlap_window_ms=early_overlap_window_ms,
            late_overlap_window_ms=late_overlap_window_ms,
            early_exposed_stall_ms=whole_total_stall_ms,
            late_exposed_stall_ms=0.0,
            total_exposed_stall_ms=whole_total_stall_ms,
        )

    return LayerTransferDecision(
        strategy="staged_split",
        miss_objects=int(miss_objects),
        cold_miss_objects=int(cold_miss_objects),
        capacity_miss_objects=int(capacity_miss_objects),
        early_bytes=early_bytes,
        late_bytes=late_bytes,
        total_bytes=total_bytes,
        early_slices=early_slices,
        late_slices=late_slices,
        total_slices=total_slices,
        early_service_ms=early_service_ms,
        late_service_ms=late_service_ms,
        whole_service_ms=whole_service_ms,
        early_overlap_window_ms=early_overlap_window_ms,
        late_overlap_window_ms=late_overlap_window_ms,
        early_exposed_stall_ms=split_early_stall_ms,
        late_exposed_stall_ms=split_late_stall_ms,
        total_exposed_stall_ms=split_total_stall_ms,
    )


def choose_execution_first_strategy_interp(
    miss_objects: int,
    cold_miss_objects: int,
    prefetched_miss_objects: int,
    object_sizes: ComponentSizeBytes,
    calibration: Mapping[str, object],
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    system_batch: int,
    prefetch_cpu_discount: float,
    overlap_policy: str = OVERLAP_POLICY_CALIBRATED,
    sync_promotion: bool = False,
) -> LayerTransferDecision:
    capacity_miss_objects = max(int(miss_objects) - int(cold_miss_objects), 0)
    early_overlap_window_ms = resolve_overlap_window_ms(
        calibration, system_batch, "early", stat_key, overlap_policy
    )
    late_overlap_window_ms = resolve_overlap_window_ms(
        calibration, system_batch, "late", stat_key, overlap_policy
    )

    if miss_objects <= 0:
        return LayerTransferDecision(
            strategy="hit_only",
            miss_objects=0,
            cold_miss_objects=0,
            capacity_miss_objects=0,
            early_bytes=0,
            late_bytes=0,
            total_bytes=0,
            early_slices=0,
            late_slices=0,
            total_slices=0,
            early_service_ms=0.0,
            late_service_ms=0.0,
            whole_service_ms=0.0,
            early_overlap_window_ms=early_overlap_window_ms,
            late_overlap_window_ms=late_overlap_window_ms,
            early_exposed_stall_ms=0.0,
            late_exposed_stall_ms=0.0,
            total_exposed_stall_ms=0.0,
        )

    prefetched_miss_objects = min(max(int(prefetched_miss_objects), 0), int(miss_objects))
    prefetch_fraction = float(prefetched_miss_objects) / float(max(int(miss_objects), 1))
    cpu_discount = max(0.0, min(float(prefetch_cpu_discount), 1.0)) * prefetch_fraction

    early_bytes = int(miss_objects) * int(object_sizes.early_activation_bytes + object_sizes.early_result_bytes)
    late_bytes = int(miss_objects) * int(object_sizes.late_activation_bytes + object_sizes.late_result_bytes)
    weight_bytes = int(miss_objects) * int(object_sizes.total_object_bytes) if sync_promotion else 0
    total_bytes = int(early_bytes + late_bytes + weight_bytes)
    early_slices = int(math.ceil(float(early_bytes) / float(object_sizes.slice_bytes)))
    late_slices = int(math.ceil(float(late_bytes) / float(object_sizes.slice_bytes)))
    total_slices = int(math.ceil(float(total_bytes) / float(object_sizes.slice_bytes)))

    early_pack_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="early",
        component_name="pack_rows",
        x_key="row_count",
        x_value=float(miss_objects),
        stat_key=stat_key,
    )
    early_d2h_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="early",
        component_name="d2h_rows",
        x_key="payload_bytes",
        x_value=float(int(miss_objects) * int(object_sizes.early_activation_bytes)),
        stat_key=stat_key,
    )
    early_cpu_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="early",
        component_name="cpu_rows",
        x_key="row_count",
        x_value=float(miss_objects),
        stat_key=stat_key,
    )
    early_cpu_ms *= 1.0 - cpu_discount
    early_h2d_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="early",
        component_name="h2d_rows",
        x_key="payload_bytes",
        x_value=float(int(miss_objects) * int(object_sizes.early_result_bytes)),
        stat_key=stat_key,
    )
    early_merge_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="early",
        component_name="merge_rows",
        x_key="row_count",
        x_value=float(miss_objects),
        stat_key=stat_key,
    )

    late_pack_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="late",
        component_name="pack_rows",
        x_key="row_count",
        x_value=float(miss_objects),
        stat_key=stat_key,
    )
    late_d2h_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="late",
        component_name="d2h_rows",
        x_key="payload_bytes",
        x_value=float(int(miss_objects) * int(object_sizes.late_activation_bytes)),
        stat_key=stat_key,
    )
    late_cpu_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="late",
        component_name="cpu_rows",
        x_key="row_count",
        x_value=float(miss_objects),
        stat_key=stat_key,
    )
    late_cpu_ms *= 1.0 - cpu_discount
    late_h2d_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="late",
        component_name="h2d_rows",
        x_key="payload_bytes",
        x_value=float(int(miss_objects) * int(object_sizes.late_result_bytes)),
        stat_key=stat_key,
    )
    late_merge_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="late",
        component_name="merge_rows",
        x_key="row_count",
        x_value=float(miss_objects),
        stat_key=stat_key,
    )

    early_service_ms = early_pack_ms + early_d2h_ms + early_cpu_ms + early_h2d_ms + early_merge_ms
    late_service_ms = late_pack_ms + late_d2h_ms + late_cpu_ms + late_h2d_ms + late_merge_ms
    early_exposed_stall_ms = max(0.0, early_service_ms - early_overlap_window_ms)
    late_exposed_stall_ms = max(0.0, late_service_ms - late_overlap_window_ms)
    sync_promotion_ms = 0.0
    if sync_promotion:
        sync_promotion_ms = float(
            calibration_service_ms(
                calibration=calibration,
                transfer_mode=transfer_mode,
                load_profile=load_profile,
                payload_bytes=weight_bytes,
                stat_key=stat_key,
            )
        )
        early_exposed_stall_ms += sync_promotion_ms

    return LayerTransferDecision(
        strategy="execution_first_sync_promotion" if sync_promotion else "staged_split",
        miss_objects=int(miss_objects),
        cold_miss_objects=int(cold_miss_objects),
        capacity_miss_objects=int(capacity_miss_objects),
        early_bytes=early_bytes,
        late_bytes=late_bytes,
        total_bytes=total_bytes,
        early_slices=early_slices,
        late_slices=late_slices,
        total_slices=total_slices,
        early_service_ms=float(early_service_ms),
        late_service_ms=float(late_service_ms),
        whole_service_ms=float(early_service_ms + late_service_ms + sync_promotion_ms),
        early_overlap_window_ms=float(early_overlap_window_ms),
        late_overlap_window_ms=float(late_overlap_window_ms),
        early_exposed_stall_ms=float(early_exposed_stall_ms),
        late_exposed_stall_ms=float(late_exposed_stall_ms),
        total_exposed_stall_ms=float(early_exposed_stall_ms + late_exposed_stall_ms),
    )


def choose_blocking_promotion_first_strategy_interp(
    miss_objects: int,
    cold_miss_objects: int,
    object_sizes: ComponentSizeBytes,
    calibration: Mapping[str, object],
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
) -> LayerTransferDecision:
    capacity_miss_objects = max(int(miss_objects) - int(cold_miss_objects), 0)
    if miss_objects <= 0:
        return LayerTransferDecision(
            strategy="hit_only",
            miss_objects=0,
            cold_miss_objects=0,
            capacity_miss_objects=0,
            early_bytes=0,
            late_bytes=0,
            total_bytes=0,
            early_slices=0,
            late_slices=0,
            total_slices=0,
            early_service_ms=0.0,
            late_service_ms=0.0,
            whole_service_ms=0.0,
            early_overlap_window_ms=0.0,
            late_overlap_window_ms=0.0,
            early_exposed_stall_ms=0.0,
            late_exposed_stall_ms=0.0,
            total_exposed_stall_ms=0.0,
        )

    total_bytes = int(miss_objects) * int(object_sizes.total_object_bytes)
    total_slices = int(math.ceil(float(total_bytes) / float(object_sizes.slice_bytes)))
    blocking_ms = float(
        calibration_service_ms(
            calibration=calibration,
            transfer_mode=transfer_mode,
            load_profile=load_profile,
            payload_bytes=total_bytes,
            stat_key=stat_key,
        )
    )
    return LayerTransferDecision(
        strategy="blocking_promotion_first",
        miss_objects=int(miss_objects),
        cold_miss_objects=int(cold_miss_objects),
        capacity_miss_objects=int(capacity_miss_objects),
        early_bytes=0,
        late_bytes=0,
        total_bytes=total_bytes,
        early_slices=0,
        late_slices=0,
        total_slices=total_slices,
        early_service_ms=0.0,
        late_service_ms=0.0,
        whole_service_ms=blocking_ms,
        early_overlap_window_ms=0.0,
        late_overlap_window_ms=0.0,
        early_exposed_stall_ms=blocking_ms,
        late_exposed_stall_ms=0.0,
        total_exposed_stall_ms=blocking_ms,
    )


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
                "system TPOT replay diverged from B8 per-request misses: "
                f"condition={condition}, budget={cache_budget}, req_idx={req_idx}, "
                f"expected={expected}, observed={observed}"
            )


def compute_stage1_token_tpot(
    condition: str,
    cache_budget: int,
    stream: TokenStructuredStream,
    replay_summary: PhaseReplayBitmapSummary,
    calibration: Mapping[str, object],
    object_sizes: ComponentSizeBytes,
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    system_batch: int,
    tail_quantile: float,
    miss_handling_mode: str = MISS_HANDLING_LOAD_THEN_RUN,
    prefetch_cpu_discount: float = 0.0,
    overlap_policy: str = OVERLAP_POLICY_CALIBRATED,
) -> Stage1TPOTOutputs:
    if tail_quantile <= 0.0 or tail_quantile >= 1.0:
        raise ValueError(f"tail_quantile must lie in (0, 1), got {tail_quantile}")

    condition_label = CONDITION_LABELS[condition]
    base_tpot_ms = calibration_base_tpot_ms_interp(calibration, system_batch)
    token_rows: List[dict] = []
    request_tokens: Dict[int, List[dict]] = defaultdict(list)
    layer_metrics: Dict[int, Dict[str, object]] = {}

    for token in stream.decode_tokens:
        token_exposed_ms = 0.0
        token_miss_objects = 0
        token_cold_miss_objects = 0
        token_capacity_miss_objects = 0
        token_miss_bytes = 0
        token_miss_slices = 0
        token_prefetched_miss_objects = 0

        for step in stream.decode_layer_steps[token.step_start:token.step_end]:
            miss_objects = int(np.sum(replay_summary.miss_flags[step.start_index:step.end_index]))
            cold_miss_objects = int(np.sum(replay_summary.cold_miss_flags[step.start_index:step.end_index]))
            prefetched_miss_objects = (
                int(np.sum(replay_summary.prefetch_hit_flags[step.start_index:step.end_index]))
                if replay_summary.prefetch_hit_flags is not None
                else 0
            )
            if miss_handling_mode == MISS_HANDLING_NO_CPU_PATH:
                decision = choose_blocking_promotion_first_strategy_interp(
                    miss_objects=miss_objects,
                    cold_miss_objects=cold_miss_objects,
                    object_sizes=object_sizes,
                    calibration=calibration,
                    transfer_mode=transfer_mode,
                    load_profile=load_profile,
                    stat_key=stat_key,
                )
            elif miss_handling_mode in (MISS_HANDLING_EXECUTION_FIRST, MISS_HANDLING_NO_DEFERRED_SYNC):
                decision = choose_execution_first_strategy_interp(
                    miss_objects=miss_objects,
                    cold_miss_objects=cold_miss_objects,
                    prefetched_miss_objects=prefetched_miss_objects,
                    object_sizes=object_sizes,
                    calibration=calibration,
                    transfer_mode=transfer_mode,
                    load_profile=load_profile,
                    stat_key=stat_key,
                    system_batch=system_batch,
                    prefetch_cpu_discount=prefetch_cpu_discount,
                    overlap_policy=overlap_policy,
                    sync_promotion=miss_handling_mode == MISS_HANDLING_NO_DEFERRED_SYNC,
                )
            else:
                decision = choose_layer_transfer_strategy_interp(
                    miss_objects=miss_objects,
                    cold_miss_objects=cold_miss_objects,
                    object_sizes=object_sizes,
                    calibration=calibration,
                    transfer_mode=transfer_mode,
                    load_profile=load_profile,
                    stat_key=stat_key,
                    system_batch=system_batch,
                    overlap_policy=overlap_policy,
                )
            token_exposed_ms += float(decision.total_exposed_stall_ms)
            token_miss_objects += int(decision.miss_objects)
            token_cold_miss_objects += int(decision.cold_miss_objects)
            token_capacity_miss_objects += int(decision.capacity_miss_objects)
            token_miss_bytes += int(decision.total_bytes)
            token_miss_slices += int(decision.total_slices)
            token_prefetched_miss_objects += int(prefetched_miss_objects)

            metrics = layer_metrics.setdefault(
                int(step.layer_id),
                {
                    "strategy_counts": defaultdict(int),
                    "miss_objects": [],
                    "cold_miss_objects": [],
                    "capacity_miss_objects": [],
                    "prefetched_miss_objects": [],
                    "miss_bytes": [],
                    "miss_slices": [],
                    "total_exposed_stall_ms": [],
                    "early_service_ms": [],
                    "late_service_ms": [],
                    "whole_service_ms": [],
                    "early_overlap_window_ms": [],
                    "late_overlap_window_ms": [],
                },
            )
            metrics["strategy_counts"][decision.strategy] += 1
            metrics["miss_objects"].append(float(decision.miss_objects))
            metrics["cold_miss_objects"].append(float(decision.cold_miss_objects))
            metrics["capacity_miss_objects"].append(float(decision.capacity_miss_objects))
            metrics["prefetched_miss_objects"].append(float(prefetched_miss_objects))
            metrics["miss_bytes"].append(float(decision.total_bytes))
            metrics["miss_slices"].append(float(decision.total_slices))
            metrics["total_exposed_stall_ms"].append(float(decision.total_exposed_stall_ms))
            metrics["early_service_ms"].append(float(decision.early_service_ms))
            metrics["late_service_ms"].append(float(decision.late_service_ms))
            metrics["whole_service_ms"].append(float(decision.whole_service_ms))
            metrics["early_overlap_window_ms"].append(float(decision.early_overlap_window_ms))
            metrics["late_overlap_window_ms"].append(float(decision.late_overlap_window_ms))

        tpot_ms = float(base_tpot_ms + token_exposed_ms)
        row = {
            "condition": condition,
            "condition_label": condition_label,
            "cache_budget": int(cache_budget),
            "miss_handling_mode": str(miss_handling_mode),
            "overlap_policy": str(overlap_policy),
            "req_idx": int(token.req_idx),
            "token_pos": int(token.token_pos),
            "token_ordinal": int(token.token_ordinal),
            "base_tpot_ms": float(base_tpot_ms),
            "tpot_ms": float(tpot_ms),
            "exposed_miss_latency_ms": float(token_exposed_ms),
            "miss_objects": int(token_miss_objects),
            "cold_miss_objects": int(token_cold_miss_objects),
            "capacity_miss_objects": int(token_capacity_miss_objects),
            "prefetched_miss_objects": int(token_prefetched_miss_objects),
            "miss_bytes": int(token_miss_bytes),
            "miss_slices": int(token_miss_slices),
            "layer_count": int(token.step_end - token.step_start),
        }
        token_rows.append(row)
        request_tokens[int(token.req_idx)].append(row)

    tpot_values = [float(row["tpot_ms"]) for row in token_rows]
    quantiles = summarize_values(tpot_values)
    tail_threshold_ms = float(percentile(tpot_values, tail_quantile)) if tpot_values else 0.0
    tail_rows = [
        row
        for row in token_rows
        if float(row["tpot_ms"]) >= tail_threshold_ms
    ]
    tail_rows.sort(key=lambda row: (-float(row["tpot_ms"]), int(row["req_idx"]), int(row["token_ordinal"])))

    request_rows: List[dict] = []
    for req_idx in sorted(request_tokens):
        rows = request_tokens[req_idx]
        tpot_req = [float(row["tpot_ms"]) for row in rows]
        exposed_req = [float(row["exposed_miss_latency_ms"]) for row in rows]
        request_rows.append(
            {
                "condition": condition,
                "condition_label": condition_label,
                "cache_budget": int(cache_budget),
                "miss_handling_mode": str(miss_handling_mode),
                "overlap_policy": str(overlap_policy),
                "req_idx": int(req_idx),
                "decode_tokens": int(len(rows)),
                "mean_tpot_ms": float(sum(tpot_req) / len(tpot_req)) if tpot_req else 0.0,
                "p50_tpot_ms": float(percentile(tpot_req, 0.50)) if tpot_req else 0.0,
                "p95_tpot_ms": float(percentile(tpot_req, 0.95)) if tpot_req else 0.0,
                "p99_tpot_ms": float(percentile(tpot_req, 0.99)) if tpot_req else 0.0,
                "max_tpot_ms": float(max(tpot_req)) if tpot_req else 0.0,
                "mean_exposed_miss_latency_ms": float(sum(exposed_req) / len(exposed_req)) if exposed_req else 0.0,
                "total_miss_objects": int(sum(int(row["miss_objects"]) for row in rows)),
                "total_cold_miss_objects": int(sum(int(row["cold_miss_objects"]) for row in rows)),
                "total_capacity_miss_objects": int(sum(int(row["capacity_miss_objects"]) for row in rows)),
                "total_prefetched_miss_objects": int(sum(int(row["prefetched_miss_objects"]) for row in rows)),
            }
        )

    layer_rows: List[dict] = []
    for layer_id in sorted(layer_metrics):
        metrics = layer_metrics[layer_id]
        strategy_counts = dict(metrics["strategy_counts"])
        total_steps = int(sum(strategy_counts.values()))
        layer_rows.append(
            {
                "condition": condition,
                "condition_label": condition_label,
                "cache_budget": int(cache_budget),
                "miss_handling_mode": str(miss_handling_mode),
                "overlap_policy": str(overlap_policy),
                "layer_id": int(layer_id),
                "steps": int(total_steps),
                "mean_miss_objects": float(sum(metrics["miss_objects"]) / total_steps) if total_steps else 0.0,
                "p95_miss_objects": float(percentile(metrics["miss_objects"], 0.95)) if total_steps else 0.0,
                "mean_prefetched_miss_objects": float(sum(metrics["prefetched_miss_objects"]) / total_steps) if total_steps else 0.0,
                "mean_miss_bytes": float(sum(metrics["miss_bytes"]) / total_steps) if total_steps else 0.0,
                "mean_miss_slices": float(sum(metrics["miss_slices"]) / total_steps) if total_steps else 0.0,
                "mean_exposed_stall_ms": float(sum(metrics["total_exposed_stall_ms"]) / total_steps) if total_steps else 0.0,
                "p95_exposed_stall_ms": float(percentile(metrics["total_exposed_stall_ms"], 0.95)) if total_steps else 0.0,
                "mean_early_service_ms": float(sum(metrics["early_service_ms"]) / total_steps) if total_steps else 0.0,
                "mean_late_service_ms": float(sum(metrics["late_service_ms"]) / total_steps) if total_steps else 0.0,
                "mean_whole_service_ms": float(sum(metrics["whole_service_ms"]) / total_steps) if total_steps else 0.0,
                "mean_early_overlap_window_ms": float(sum(metrics["early_overlap_window_ms"]) / total_steps) if total_steps else 0.0,
                "mean_late_overlap_window_ms": float(sum(metrics["late_overlap_window_ms"]) / total_steps) if total_steps else 0.0,
                "share_hit_only": float(strategy_counts.get("hit_only", 0) / total_steps) if total_steps else 0.0,
                "share_whole_object_early": float(strategy_counts.get("whole_object_early", 0) / total_steps) if total_steps else 0.0,
                "share_staged_split": float(strategy_counts.get("staged_split", 0) / total_steps) if total_steps else 0.0,
                "share_blocking_promotion_first": float(strategy_counts.get("blocking_promotion_first", 0) / total_steps) if total_steps else 0.0,
                "share_execution_first_sync_promotion": float(strategy_counts.get("execution_first_sync_promotion", 0) / total_steps) if total_steps else 0.0,
            }
        )

    return Stage1TPOTOutputs(
        token_rows=token_rows,
        quantiles=quantiles,
        tail_rows=tail_rows,
        layer_rows=layer_rows,
        request_rows=request_rows,
        tail_threshold_ms=float(tail_threshold_ms),
    )


def resolve_request_arrival_times_ms(stream: TokenStructuredStream) -> np.ndarray:
    if stream.request_start_ts.size > 0 and np.all(stream.request_start_ts >= 0):
        starts = stream.request_start_ts.astype(np.float64)
        if np.all(starts[1:] >= starts[:-1]):
            return starts - float(starts[0])
    if stream.request_arrival_idx.size > 0:
        arrivals = stream.request_arrival_idx.astype(np.float64)
        return arrivals - float(arrivals[0])
    return np.zeros(len(stream.request_ids), dtype=np.float64)


def _simulate_batch_service_ms(
    batch_tokens: Sequence[TokenTemplate],
    calibration: Mapping[str, object],
    object_sizes: ComponentSizeBytes,
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    miss_handling_mode: str,
    prefetch_cpu_discount: float,
    overlap_policy: str,
) -> Tuple[float, float]:
    if not batch_tokens:
        return 0.0, 0.0

    actual_batch = int(len(batch_tokens))
    base_tpot_ms = calibration_base_tpot_ms_interp(calibration, actual_batch)
    total_stall_ms = 0.0
    total_queue_wait_ms = 0.0
    total_jobs = 0
    max_layers = max(len(token.layers) for token in batch_tokens)

    for layer_ordinal in range(max_layers):
        decisions: List[LayerTransferDecision] = []
        for token in batch_tokens:
            if layer_ordinal >= len(token.layers):
                continue
            layer = token.layers[layer_ordinal]
            if miss_handling_mode == MISS_HANDLING_NO_CPU_PATH:
                decisions.append(
                    choose_blocking_promotion_first_strategy_interp(
                        miss_objects=int(layer.miss_objects),
                        cold_miss_objects=int(layer.cold_miss_objects),
                        object_sizes=object_sizes,
                        calibration=calibration,
                        transfer_mode=transfer_mode,
                        load_profile=load_profile,
                        stat_key=stat_key,
                    )
                )
            elif miss_handling_mode in (MISS_HANDLING_EXECUTION_FIRST, MISS_HANDLING_NO_DEFERRED_SYNC):
                decisions.append(
                    choose_execution_first_strategy_interp(
                        miss_objects=int(layer.miss_objects),
                        cold_miss_objects=int(layer.cold_miss_objects),
                        prefetched_miss_objects=int(layer.prefetched_miss_objects),
                        object_sizes=object_sizes,
                        calibration=calibration,
                        transfer_mode=transfer_mode,
                        load_profile=load_profile,
                        stat_key=stat_key,
                        system_batch=actual_batch,
                        prefetch_cpu_discount=prefetch_cpu_discount,
                        overlap_policy=overlap_policy,
                        sync_promotion=miss_handling_mode == MISS_HANDLING_NO_DEFERRED_SYNC,
                    )
                )
            else:
                decisions.append(
                    choose_layer_transfer_strategy_interp(
                        miss_objects=int(layer.miss_objects),
                        cold_miss_objects=int(layer.cold_miss_objects),
                        object_sizes=object_sizes,
                        calibration=calibration,
                        transfer_mode=transfer_mode,
                        load_profile=load_profile,
                        stat_key=stat_key,
                        system_batch=actual_batch,
                        overlap_policy=overlap_policy,
                    )
                )

        early_window = resolve_overlap_window_ms(calibration, actual_batch, "early", stat_key, overlap_policy)
        cursor_ms = 0.0
        early_completion_ms = 0.0
        for decision in decisions:
            if decision.strategy == "whole_object_early":
                service_ms = float(decision.whole_service_ms)
            elif decision.strategy in ("blocking_promotion_first", "execution_first_sync_promotion"):
                service_ms = float(decision.whole_service_ms)
            elif decision.strategy == "staged_split":
                service_ms = float(decision.early_service_ms)
            else:
                service_ms = 0.0
            if service_ms <= 0.0:
                continue
            total_queue_wait_ms += float(cursor_ms)
            total_jobs += 1
            cursor_ms += service_ms
            early_completion_ms = max(early_completion_ms, cursor_ms)
        total_stall_ms += max(0.0, early_completion_ms - early_window)

        late_window = resolve_overlap_window_ms(calibration, actual_batch, "late", stat_key, overlap_policy)
        cursor_ms = 0.0
        late_completion_ms = 0.0
        for decision in decisions:
            service_ms = float(decision.late_service_ms) if decision.strategy == "staged_split" else 0.0
            if service_ms <= 0.0:
                continue
            total_queue_wait_ms += float(cursor_ms)
            total_jobs += 1
            cursor_ms += service_ms
            late_completion_ms = max(late_completion_ms, cursor_ms)
        total_stall_ms += max(0.0, late_completion_ms - late_window)

    mean_pcie_queue_wait_ms = float(total_queue_wait_ms / total_jobs) if total_jobs > 0 else 0.0
    return float(base_tpot_ms + total_stall_ms), mean_pcie_queue_wait_ms


def simulate_scheduler_sensitivity(
    condition: str,
    cache_budget: int,
    request_ids: Sequence[int],
    token_templates_by_req: Mapping[int, Sequence[TokenTemplate]],
    request_arrival_times_ms: Sequence[float],
    calibration: Mapping[str, object],
    object_sizes: ComponentSizeBytes,
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    max_system_batch: int,
    miss_handling_mode: str = MISS_HANDLING_LOAD_THEN_RUN,
    prefetch_cpu_discount: float = 0.0,
    overlap_policy: str = OVERLAP_POLICY_CALIBRATED,
) -> SchedulerSimulationSummary:
    if max_system_batch <= 0:
        raise ValueError(f"max_system_batch must be positive, got {max_system_batch}")

    condition_label = CONDITION_LABELS[condition]
    arrival_map = {int(req_idx): float(request_arrival_times_ms[ordinal]) for ordinal, req_idx in enumerate(request_ids)}
    future: List[Tuple[float, int, _SchedulerRequestState]] = []
    for request_ordinal, req_idx in enumerate(request_ids):
        tokens = list(token_templates_by_req.get(int(req_idx), ()))
        if not tokens:
            continue
        future.append(
            (
                float(arrival_map[int(req_idx)]),
                int(request_ordinal),
                _SchedulerRequestState(
                    req_idx=int(req_idx),
                    request_ordinal=int(request_ordinal),
                    tokens=tokens,
                    next_token_index=0,
                ),
            )
        )
    future.sort(key=lambda item: (item[0], item[1]))

    ready: List[Tuple[float, int, _SchedulerRequestState]] = []
    current_time_ms = float(future[0][0]) if future else 0.0
    future_index = 0
    service_tpots: List[float] = []
    completion_intervals: List[float] = []
    scheduler_waits: List[float] = []
    pcie_queue_waits: List[float] = []
    active_batch_sizes: List[float] = []
    batch_stats: List[SchedulerBatchStats] = []

    while future_index < len(future) or ready:
        while future_index < len(future) and future[future_index][0] <= current_time_ms + 1e-9:
            ready.append(future[future_index])
            future_index += 1

        if not ready:
            current_time_ms = float(future[future_index][0])
            continue

        ready.sort(key=lambda item: (item[0], item[1]))
        batch_entries = ready[:max_system_batch]
        ready = ready[max_system_batch:]
        batch_tokens = [entry[2].tokens[entry[2].next_token_index] for entry in batch_entries]
        batch_service_ms, batch_pcie_queue_wait_ms = _simulate_batch_service_ms(
            batch_tokens=batch_tokens,
            calibration=calibration,
            object_sizes=object_sizes,
            transfer_mode=transfer_mode,
            load_profile=load_profile,
            stat_key=stat_key,
            miss_handling_mode=miss_handling_mode,
            prefetch_cpu_discount=prefetch_cpu_discount,
            overlap_policy=overlap_policy,
        )

        batch_start_ms = float(current_time_ms)
        batch_end_ms = float(batch_start_ms + batch_service_ms)
        batch_scheduler_waits = [max(0.0, batch_start_ms - float(entry[0])) for entry in batch_entries]
        batch_request_ids = tuple(int(entry[2].req_idx) for entry in batch_entries)
        batch_token_ordinals = tuple(int(token.token_ordinal) for token in batch_tokens)
        batch_stats.append(
            SchedulerBatchStats(
                request_ids=batch_request_ids,
                token_ordinals=batch_token_ordinals,
                batch_start_ms=float(batch_start_ms),
                batch_duration_ms=float(batch_service_ms),
                scheduler_queue_wait_ms=float(sum(batch_scheduler_waits) / len(batch_scheduler_waits)),
                pcie_queue_wait_ms=float(batch_pcie_queue_wait_ms),
                active_request_count=int(len(batch_entries)),
            )
        )

        for entry, scheduler_wait_ms, token in zip(batch_entries, batch_scheduler_waits, batch_tokens):
            state = entry[2]
            service_tpots.append(float(batch_service_ms))
            completion_intervals.append(float(scheduler_wait_ms + batch_service_ms))
            scheduler_waits.append(float(scheduler_wait_ms))
            pcie_queue_waits.append(float(batch_pcie_queue_wait_ms))
            active_batch_sizes.append(float(len(batch_entries)))
            state.next_token_index += 1
            if state.next_token_index < len(state.tokens):
                ready.append((float(batch_end_ms), int(state.request_ordinal), state))

        current_time_ms = batch_end_ms

    service_summary = summarize_values(service_tpots)
    completion_summary = summarize_values(completion_intervals)
    scheduler_wait_summary = summarize_values(scheduler_waits)
    pcie_queue_summary = summarize_values(pcie_queue_waits)

    return SchedulerSimulationSummary(
        summary_row={
            "condition": condition,
            "condition_label": condition_label,
            "cache_budget": int(cache_budget),
            "miss_handling_mode": str(miss_handling_mode),
            "overlap_policy": str(overlap_policy),
            "system_batch": int(max_system_batch),
            "token_count": int(len(service_tpots)),
            "batch_count": int(len(batch_stats)),
            "mean_service_tpot_ms": float(service_summary["mean"]),
            "p50_service_tpot_ms": float(service_summary["p50"]),
            "p95_service_tpot_ms": float(service_summary["p95"]),
            "p99_service_tpot_ms": float(service_summary["p99"]),
            "max_service_tpot_ms": float(service_summary["max"]),
            "mean_completion_interval_ms": float(completion_summary["mean"]),
            "p50_completion_interval_ms": float(completion_summary["p50"]),
            "p95_completion_interval_ms": float(completion_summary["p95"]),
            "p99_completion_interval_ms": float(completion_summary["p99"]),
            "max_completion_interval_ms": float(completion_summary["max"]),
            "mean_scheduler_queue_wait_ms": float(scheduler_wait_summary["mean"]),
            "p95_scheduler_queue_wait_ms": float(scheduler_wait_summary["p95"]),
            "mean_pcie_queue_wait_ms": float(pcie_queue_summary["mean"]),
            "p95_pcie_queue_wait_ms": float(pcie_queue_summary["p95"]),
            "mean_active_batch_size": float(sum(active_batch_sizes) / len(active_batch_sizes)) if active_batch_sizes else 0.0,
        },
        batch_stats=batch_stats,
    )
