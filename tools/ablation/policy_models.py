#!/usr/bin/env python3
"""Ablation-local replay and service models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = THIS_DIR.parent / "case_study"

import sys

if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(CASE_STUDY_DIR) not in sys.path:
    sys.path.append(str(CASE_STUDY_DIR))

from replay_core import PHASE_DECODE, PhaseReplayBitmapSummary, simulate_phase_replay_with_bitmaps  # type: ignore
from system_tpot_core import (  # type: ignore
    ComponentSizeBytes,
    TokenStructuredStream,
    apply_temporal_prefetch_hits,
    build_replay_policy_state,
    calibration_service_ms,
    compute_component_sizes,
    dtype_name_to_torch_dtype,
    load_model_text_config,
    simulate_miss_path_replay_with_bitmaps,
    validate_execution_first_calibration,
)
from system_tpot_sim import (  # type: ignore
    calibration_base_tpot_ms_interp,
    calibration_cold_path_component_ms_interp,
    calibration_overlap_window_ms_interp,
)

from ablation_utils import quantile_summary
from variants import (
    AblationVariant,
    OVERLAP_POLICY_DISABLED,
    SERVICE_MODEL_BLOCKING_PROMOTION_FIRST,
    SERVICE_MODEL_EXECUTION_FIRST,
    SERVICE_MODEL_EXECUTION_FIRST_SYNC_PROMOTION,
    REPLAY_POLICY_ALWAYS_ADMIT,
)


@dataclass(frozen=True)
class LayerServiceBreakdown:
    strategy: str
    miss_objects: int
    cold_miss_objects: int
    capacity_miss_objects: int
    prefetched_miss_objects: int
    total_service_ms: float
    exposed_stall_ms: float
    gpu_wait_ms: float
    cpu_compute_ms: float
    d2h_h2d_ms: float
    merge_ms: float
    overlap_hidden_ms: float
    d2h_bytes: int
    h2d_bytes: int
    weight_h2d_bytes: int
    cpu_busy_ms: float
    total_bytes: int
    total_slices: int
    early_overlap_window_ms: float
    late_overlap_window_ms: float
    early_pack_ms: float
    early_d2h_ms: float
    early_cpu_ms: float
    early_h2d_ms: float
    early_merge_ms: float
    late_pack_ms: float
    late_d2h_ms: float
    late_cpu_ms: float
    late_h2d_ms: float
    late_merge_ms: float
    weight_h2d_ms: float


@dataclass(frozen=True)
class VariantSimulationResult:
    token_rows: List[dict]
    request_rows: List[dict]
    step_rows: List[dict]
    quantiles: Dict[str, float]
    tail_threshold_ms: float


def validate_calibration_for_variant(
    calibration: Mapping[str, object],
    variant: AblationVariant,
    load_profile: str,
) -> None:
    if variant.service_model in (SERVICE_MODEL_EXECUTION_FIRST, SERVICE_MODEL_EXECUTION_FIRST_SYNC_PROMOTION):
        validate_execution_first_calibration(
            calibration=calibration,
            load_profile=load_profile,
            require_overlap_windows=variant.overlap_policy != OVERLAP_POLICY_DISABLED,
            tool_name="tools/ablation/run_ablation_suite.py",
        )


def resolve_object_sizes(
    calibration: Mapping[str, object],
    model_dir: Path,
    dtype_name: str,
    lora_rank: int,
    slice_bytes: int,
) -> ComponentSizeBytes:
    object_sizes_payload = calibration.get("object_sizes", {})
    if isinstance(object_sizes_payload, Mapping) and object_sizes_payload.get("hidden_size") is not None:
        return compute_component_sizes(
            hidden_size=int(object_sizes_payload["hidden_size"]),
            moe_intermediate_size=int(object_sizes_payload["moe_intermediate_size"]),
            lora_rank=int(object_sizes_payload.get("lora_rank", lora_rank)),
            dtype=dtype_name_to_torch_dtype(str(object_sizes_payload.get("dtype", dtype_name))),
            slice_bytes=int(object_sizes_payload.get("slice_bytes", slice_bytes)),
            early_component_label=str(object_sizes_payload.get("early_component_label", "up_proj")),
            late_component_label=str(object_sizes_payload.get("late_component_label", "down_proj")),
            gate_lora_excluded=bool(object_sizes_payload.get("gate_lora_excluded", True)),
        )

    text_config = load_model_text_config(model_dir)
    return compute_component_sizes(
        hidden_size=int(text_config["hidden_size"]),
        moe_intermediate_size=int(text_config["moe_intermediate_size"]),
        lora_rank=int(lora_rank),
        dtype=dtype_name_to_torch_dtype(dtype_name),
        slice_bytes=int(slice_bytes),
    )


def build_variant_replay_summary(
    variant: AblationVariant,
    stream: TokenStructuredStream,
    cache_budget: int,
) -> PhaseReplayBitmapSummary:
    condition_buffer = stream.condition_buffers[variant.trace_condition]
    if variant.replay_policy == REPLAY_POLICY_ALWAYS_ADMIT:
        replay_summary = simulate_phase_replay_with_bitmaps(
            condition_buffer=condition_buffer,
            phase_ids=stream.phase_ids,
            request_offsets=stream.request_offsets,
            cache_budget=int(cache_budget),
        )
        replay_summary.prefetch_hit_flags = np.zeros(stream.total_events, dtype=np.uint8)
        return replay_summary

    policy_state = build_replay_policy_state(
        stream=stream,
        access_buffer=condition_buffer,
        enable_temporal_prefetch=bool(variant.temporal_prefetch),
    )
    replay_summary = simulate_miss_path_replay_with_bitmaps(
        condition_buffer=condition_buffer,
        stream=stream,
        cache_budget=int(cache_budget),
        miss_handling_mode="execution_first",
        deferred_promotion_delta_steps=int(variant.deferred_promotion_delta_steps),
        policy_state=policy_state,
    )
    if variant.temporal_prefetch:
        replay_summary = apply_temporal_prefetch_hits(
            replay_summary=replay_summary,
            plan=policy_state.temporal_prefetch_plan,
        )
    else:
        replay_summary.prefetch_hit_flags = np.zeros(stream.total_events, dtype=np.uint8)
    return replay_summary


def _execution_first_components(
    miss_objects: int,
    object_sizes: ComponentSizeBytes,
    calibration: Mapping[str, object],
    load_profile: str,
    stat_key: str,
    prefetched_miss_objects: int,
) -> dict:
    if miss_objects <= 0:
        return {
            "early_pack_ms": 0.0,
            "early_d2h_ms": 0.0,
            "early_cpu_ms": 0.0,
            "early_h2d_ms": 0.0,
            "early_merge_ms": 0.0,
            "late_pack_ms": 0.0,
            "late_d2h_ms": 0.0,
            "late_cpu_ms": 0.0,
            "late_h2d_ms": 0.0,
            "late_merge_ms": 0.0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
        }

    prefetched = min(max(int(prefetched_miss_objects), 0), int(miss_objects))
    prefetch_fraction = float(prefetched) / float(max(int(miss_objects), 1))
    cpu_discount = 0.20 * prefetch_fraction

    early_activation_bytes = int(miss_objects) * int(object_sizes.early_activation_bytes)
    late_activation_bytes = int(miss_objects) * int(object_sizes.late_activation_bytes)
    early_result_bytes = int(miss_objects) * int(object_sizes.early_result_bytes)
    late_result_bytes = int(miss_objects) * int(object_sizes.late_result_bytes)

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
        x_value=float(early_activation_bytes),
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
    ) * (1.0 - cpu_discount)
    early_h2d_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="early",
        component_name="h2d_rows",
        x_key="payload_bytes",
        x_value=float(early_result_bytes),
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
        x_value=float(late_activation_bytes),
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
    ) * (1.0 - cpu_discount)
    late_h2d_ms = calibration_cold_path_component_ms_interp(
        calibration=calibration,
        load_profile=load_profile,
        stage_name="late",
        component_name="h2d_rows",
        x_key="payload_bytes",
        x_value=float(late_result_bytes),
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

    return {
        "early_pack_ms": float(early_pack_ms),
        "early_d2h_ms": float(early_d2h_ms),
        "early_cpu_ms": float(early_cpu_ms),
        "early_h2d_ms": float(early_h2d_ms),
        "early_merge_ms": float(early_merge_ms),
        "late_pack_ms": float(late_pack_ms),
        "late_d2h_ms": float(late_d2h_ms),
        "late_cpu_ms": float(late_cpu_ms),
        "late_h2d_ms": float(late_h2d_ms),
        "late_merge_ms": float(late_merge_ms),
        "d2h_bytes": int(early_activation_bytes + late_activation_bytes),
        "h2d_bytes": int(early_result_bytes + late_result_bytes),
    }


def simulate_layer_service(
    variant: AblationVariant,
    miss_objects: int,
    cold_miss_objects: int,
    prefetched_miss_objects: int,
    object_sizes: ComponentSizeBytes,
    calibration: Mapping[str, object],
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    system_batch: int,
) -> LayerServiceBreakdown:
    miss_objects = int(miss_objects)
    cold_miss_objects = int(cold_miss_objects)
    prefetched_miss_objects = min(max(int(prefetched_miss_objects), 0), miss_objects)
    capacity_miss_objects = max(miss_objects - cold_miss_objects, 0)
    total_weight_bytes = int(miss_objects) * int(object_sizes.total_object_bytes)
    total_weight_slices = int(math.ceil(float(total_weight_bytes) / float(object_sizes.slice_bytes))) if total_weight_bytes else 0

    if miss_objects <= 0:
        return LayerServiceBreakdown(
            strategy="hit_only",
            miss_objects=0,
            cold_miss_objects=0,
            capacity_miss_objects=0,
            prefetched_miss_objects=0,
            total_service_ms=0.0,
            exposed_stall_ms=0.0,
            gpu_wait_ms=0.0,
            cpu_compute_ms=0.0,
            d2h_h2d_ms=0.0,
            merge_ms=0.0,
            overlap_hidden_ms=0.0,
            d2h_bytes=0,
            h2d_bytes=0,
            weight_h2d_bytes=0,
            cpu_busy_ms=0.0,
            total_bytes=0,
            total_slices=0,
            early_overlap_window_ms=0.0,
            late_overlap_window_ms=0.0,
            early_pack_ms=0.0,
            early_d2h_ms=0.0,
            early_cpu_ms=0.0,
            early_h2d_ms=0.0,
            early_merge_ms=0.0,
            late_pack_ms=0.0,
            late_d2h_ms=0.0,
            late_cpu_ms=0.0,
            late_h2d_ms=0.0,
            late_merge_ms=0.0,
            weight_h2d_ms=0.0,
        )

    if variant.service_model == SERVICE_MODEL_BLOCKING_PROMOTION_FIRST:
        weight_h2d_ms = calibration_service_ms(
            calibration=calibration,
            transfer_mode=transfer_mode,
            load_profile=load_profile,
            payload_bytes=total_weight_bytes,
            stat_key=stat_key,
        )
        return LayerServiceBreakdown(
            strategy="blocking_promotion_first",
            miss_objects=miss_objects,
            cold_miss_objects=cold_miss_objects,
            capacity_miss_objects=capacity_miss_objects,
            prefetched_miss_objects=0,
            total_service_ms=float(weight_h2d_ms),
            exposed_stall_ms=float(weight_h2d_ms),
            gpu_wait_ms=float(weight_h2d_ms),
            cpu_compute_ms=0.0,
            d2h_h2d_ms=0.0,
            merge_ms=0.0,
            overlap_hidden_ms=0.0,
            d2h_bytes=0,
            h2d_bytes=0,
            weight_h2d_bytes=total_weight_bytes,
            cpu_busy_ms=0.0,
            total_bytes=total_weight_bytes,
            total_slices=total_weight_slices,
            early_overlap_window_ms=0.0,
            late_overlap_window_ms=0.0,
            early_pack_ms=0.0,
            early_d2h_ms=0.0,
            early_cpu_ms=0.0,
            early_h2d_ms=0.0,
            early_merge_ms=0.0,
            late_pack_ms=0.0,
            late_d2h_ms=0.0,
            late_cpu_ms=0.0,
            late_h2d_ms=0.0,
            late_merge_ms=0.0,
            weight_h2d_ms=float(weight_h2d_ms),
        )

    overlap_enabled = variant.overlap_policy != OVERLAP_POLICY_DISABLED
    components = _execution_first_components(
        miss_objects=miss_objects,
        object_sizes=object_sizes,
        calibration=calibration,
        load_profile=load_profile,
        stat_key=stat_key,
        prefetched_miss_objects=prefetched_miss_objects,
    )
    early_overlap_window_ms = (
        calibration_overlap_window_ms_interp(calibration, system_batch, "early", stat_key) if overlap_enabled else 0.0
    )
    late_overlap_window_ms = (
        calibration_overlap_window_ms_interp(calibration, system_batch, "late", stat_key) if overlap_enabled else 0.0
    )
    early_service_ms = (
        components["early_pack_ms"]
        + components["early_d2h_ms"]
        + components["early_cpu_ms"]
        + components["early_h2d_ms"]
        + components["early_merge_ms"]
    )
    late_service_ms = (
        components["late_pack_ms"]
        + components["late_d2h_ms"]
        + components["late_cpu_ms"]
        + components["late_h2d_ms"]
        + components["late_merge_ms"]
    )
    early_exposed_ms = max(0.0, float(early_service_ms) - float(early_overlap_window_ms))
    late_exposed_ms = max(0.0, float(late_service_ms) - float(late_overlap_window_ms))
    weight_h2d_ms = 0.0
    weight_h2d_bytes = 0
    total_service_ms = float(early_service_ms + late_service_ms)
    exposed_stall_ms = float(early_exposed_ms + late_exposed_ms)
    overlap_hidden_ms = float(min(early_service_ms, early_overlap_window_ms) + min(late_service_ms, late_overlap_window_ms))
    gpu_wait_ms = float(components["early_pack_ms"] + components["late_pack_ms"] + exposed_stall_ms)
    if variant.service_model == SERVICE_MODEL_EXECUTION_FIRST_SYNC_PROMOTION:
        weight_h2d_ms = float(
            calibration_service_ms(
                calibration=calibration,
                transfer_mode=transfer_mode,
                load_profile=load_profile,
                payload_bytes=total_weight_bytes,
                stat_key=stat_key,
            )
        )
        weight_h2d_bytes = total_weight_bytes
        total_service_ms += weight_h2d_ms
        exposed_stall_ms += weight_h2d_ms
        gpu_wait_ms += weight_h2d_ms

    activation_total_bytes = int(components["d2h_bytes"] + components["h2d_bytes"])
    activation_total_slices = int(math.ceil(float(activation_total_bytes) / float(object_sizes.slice_bytes))) if activation_total_bytes else 0
    total_bytes = int(activation_total_bytes + weight_h2d_bytes)
    total_slices = int(activation_total_slices + total_weight_slices)
    return LayerServiceBreakdown(
        strategy=variant.service_model,
        miss_objects=miss_objects,
        cold_miss_objects=cold_miss_objects,
        capacity_miss_objects=capacity_miss_objects,
        prefetched_miss_objects=prefetched_miss_objects,
        total_service_ms=float(total_service_ms),
        exposed_stall_ms=float(exposed_stall_ms),
        gpu_wait_ms=float(gpu_wait_ms),
        cpu_compute_ms=float(components["early_cpu_ms"] + components["late_cpu_ms"]),
        d2h_h2d_ms=float(
            components["early_d2h_ms"]
            + components["late_d2h_ms"]
            + components["early_h2d_ms"]
            + components["late_h2d_ms"]
        ),
        merge_ms=float(components["early_merge_ms"] + components["late_merge_ms"]),
        overlap_hidden_ms=float(overlap_hidden_ms),
        d2h_bytes=int(components["d2h_bytes"]),
        h2d_bytes=int(components["h2d_bytes"]),
        weight_h2d_bytes=int(weight_h2d_bytes),
        cpu_busy_ms=float(components["early_cpu_ms"] + components["late_cpu_ms"]),
        total_bytes=int(total_bytes),
        total_slices=int(total_slices),
        early_overlap_window_ms=float(early_overlap_window_ms),
        late_overlap_window_ms=float(late_overlap_window_ms),
        early_pack_ms=float(components["early_pack_ms"]),
        early_d2h_ms=float(components["early_d2h_ms"]),
        early_cpu_ms=float(components["early_cpu_ms"]),
        early_h2d_ms=float(components["early_h2d_ms"]),
        early_merge_ms=float(components["early_merge_ms"]),
        late_pack_ms=float(components["late_pack_ms"]),
        late_d2h_ms=float(components["late_d2h_ms"]),
        late_cpu_ms=float(components["late_cpu_ms"]),
        late_h2d_ms=float(components["late_h2d_ms"]),
        late_merge_ms=float(components["late_merge_ms"]),
        weight_h2d_ms=float(weight_h2d_ms),
    )


def simulate_variant_stream(
    variant: AblationVariant,
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
) -> VariantSimulationResult:
    if tail_quantile <= 0.0 or tail_quantile >= 1.0:
        raise ValueError(f"tail_quantile must lie in (0, 1), got {tail_quantile}")

    base_tpot_ms = calibration_base_tpot_ms_interp(calibration, int(system_batch))
    token_rows: List[dict] = []
    request_rows_by_id: Dict[int, List[dict]] = {}
    step_rows: List[dict] = []

    for token in stream.decode_tokens:
        token_exposed_ms = 0.0
        token_miss_objects = 0
        token_cold_miss_objects = 0
        token_capacity_miss_objects = 0
        token_prefetched_miss_objects = 0
        token_gpu_wait_ms = 0.0
        token_cpu_compute_ms = 0.0
        token_d2h_h2d_ms = 0.0
        token_merge_ms = 0.0
        token_overlap_hidden_ms = 0.0
        token_d2h_bytes = 0
        token_h2d_bytes = 0
        token_weight_h2d_bytes = 0
        token_cpu_busy_ms = 0.0
        token_total_bytes = 0
        token_total_slices = 0
        token_decode_event_count = 0

        layer_steps = stream.decode_layer_steps[token.step_start:token.step_end]
        base_step_slot_ms = float(base_tpot_ms / max(len(layer_steps), 1))

        for step in layer_steps:
            miss_objects = int(np.sum(replay_summary.miss_flags[step.start_index:step.end_index]))
            cold_miss_objects = int(np.sum(replay_summary.cold_miss_flags[step.start_index:step.end_index]))
            prefetched_miss_objects = (
                int(np.sum(replay_summary.prefetch_hit_flags[step.start_index:step.end_index]))
                if replay_summary.prefetch_hit_flags is not None
                else 0
            )
            breakdown = simulate_layer_service(
                variant=variant,
                miss_objects=miss_objects,
                cold_miss_objects=cold_miss_objects,
                prefetched_miss_objects=prefetched_miss_objects,
                object_sizes=object_sizes,
                calibration=calibration,
                transfer_mode=transfer_mode,
                load_profile=load_profile,
                stat_key=stat_key,
                system_batch=system_batch,
            )
            token_exposed_ms += breakdown.exposed_stall_ms
            token_miss_objects += breakdown.miss_objects
            token_cold_miss_objects += breakdown.cold_miss_objects
            token_capacity_miss_objects += breakdown.capacity_miss_objects
            token_prefetched_miss_objects += breakdown.prefetched_miss_objects
            token_gpu_wait_ms += breakdown.gpu_wait_ms
            token_cpu_compute_ms += breakdown.cpu_compute_ms
            token_d2h_h2d_ms += breakdown.d2h_h2d_ms
            token_merge_ms += breakdown.merge_ms
            token_overlap_hidden_ms += breakdown.overlap_hidden_ms
            token_d2h_bytes += breakdown.d2h_bytes
            token_h2d_bytes += breakdown.h2d_bytes
            token_weight_h2d_bytes += breakdown.weight_h2d_bytes
            token_cpu_busy_ms += breakdown.cpu_busy_ms
            token_total_bytes += breakdown.total_bytes
            token_total_slices += breakdown.total_slices
            token_decode_event_count += int(step.end_index - step.start_index)
            step_rows.append(
                {
                    "req_idx": int(step.req_idx),
                    "token_ordinal": int(step.token_ordinal),
                    "token_pos": int(step.token_pos),
                    "layer_id": int(step.layer_id),
                    "event_idx": int(step.event_idx),
                    "base_step_slot_ms": float(base_step_slot_ms),
                    "miss_objects": int(breakdown.miss_objects),
                    "cold_miss_objects": int(breakdown.cold_miss_objects),
                    "prefetched_miss_objects": int(breakdown.prefetched_miss_objects),
                    "exposed_stall_ms": float(breakdown.exposed_stall_ms),
                    "gpu_wait_ms": float(breakdown.gpu_wait_ms),
                    "cpu_compute_ms": float(breakdown.cpu_compute_ms),
                    "d2h_h2d_ms": float(breakdown.d2h_h2d_ms),
                    "merge_ms": float(breakdown.merge_ms),
                    "overlap_hidden_ms": float(breakdown.overlap_hidden_ms),
                    "d2h_bytes": int(breakdown.d2h_bytes),
                    "h2d_bytes": int(breakdown.h2d_bytes),
                    "weight_h2d_bytes": int(breakdown.weight_h2d_bytes),
                    "cpu_busy_ms": float(breakdown.cpu_busy_ms),
                    "total_bytes": int(breakdown.total_bytes),
                    "total_slices": int(breakdown.total_slices),
                    "early_overlap_window_ms": float(breakdown.early_overlap_window_ms),
                    "late_overlap_window_ms": float(breakdown.late_overlap_window_ms),
                    "early_pack_ms": float(breakdown.early_pack_ms),
                    "early_d2h_ms": float(breakdown.early_d2h_ms),
                    "early_cpu_ms": float(breakdown.early_cpu_ms),
                    "early_h2d_ms": float(breakdown.early_h2d_ms),
                    "early_merge_ms": float(breakdown.early_merge_ms),
                    "late_pack_ms": float(breakdown.late_pack_ms),
                    "late_d2h_ms": float(breakdown.late_d2h_ms),
                    "late_cpu_ms": float(breakdown.late_cpu_ms),
                    "late_h2d_ms": float(breakdown.late_h2d_ms),
                    "late_merge_ms": float(breakdown.late_merge_ms),
                    "weight_h2d_ms": float(breakdown.weight_h2d_ms),
                    "service_model": str(variant.service_model),
                    "strategy": str(breakdown.strategy),
                }
            )

        token_tpot_ms = float(base_tpot_ms + token_exposed_ms)
        token_row = {
            "req_idx": int(token.req_idx),
            "token_ordinal": int(token.token_ordinal),
            "token_pos": int(token.token_pos),
            "layer_count": int(token.step_end - token.step_start),
            "base_tpot_ms": float(base_tpot_ms),
            "tpot_ms": float(token_tpot_ms),
            "exposed_miss_latency_ms": float(token_exposed_ms),
            "miss_objects": int(token_miss_objects),
            "cold_miss_objects": int(token_cold_miss_objects),
            "capacity_miss_objects": int(token_capacity_miss_objects),
            "prefetched_miss_objects": int(token_prefetched_miss_objects),
            "gpu_wait_ms": float(token_gpu_wait_ms),
            "cpu_compute_ms": float(token_cpu_compute_ms),
            "d2h_h2d_ms": float(token_d2h_h2d_ms),
            "merge_ms": float(token_merge_ms),
            "overlap_hidden_ms": float(token_overlap_hidden_ms),
            "d2h_bytes": int(token_d2h_bytes),
            "h2d_bytes": int(token_h2d_bytes),
            "weight_h2d_bytes": int(token_weight_h2d_bytes),
            "cpu_busy_ms": float(token_cpu_busy_ms),
            "total_transfer_bytes": int(token_total_bytes),
            "total_slices": int(token_total_slices),
            "decode_event_count": int(token_decode_event_count),
        }
        token_rows.append(token_row)
        request_rows_by_id.setdefault(int(token.req_idx), []).append(token_row)

    tpot_values = [float(row["tpot_ms"]) for row in token_rows]
    quantiles = quantile_summary(tpot_values)
    tail_threshold_ms = float(np.asarray(tpot_values).size and np.quantile(np.asarray(tpot_values, dtype=np.float64), tail_quantile) or 0.0)

    request_rows: List[dict] = []
    for req_idx in sorted(request_rows_by_id):
        rows = request_rows_by_id[req_idx]
        request_tpots = [float(row["tpot_ms"]) for row in rows]
        request_rows.append(
            {
                "req_idx": int(req_idx),
                "decode_tokens": int(len(rows)),
                "mean_tpot_ms": float(sum(request_tpots) / len(request_tpots)) if request_tpots else 0.0,
                "p50_tpot_ms": float(np.quantile(np.asarray(request_tpots, dtype=np.float64), 0.50)) if request_tpots else 0.0,
                "p95_tpot_ms": float(np.quantile(np.asarray(request_tpots, dtype=np.float64), 0.95)) if request_tpots else 0.0,
                "p99_tpot_ms": float(np.quantile(np.asarray(request_tpots, dtype=np.float64), 0.99)) if request_tpots else 0.0,
                "max_tpot_ms": float(max(request_tpots)) if request_tpots else 0.0,
                "mean_gpu_wait_ms": float(sum(float(row["gpu_wait_ms"]) for row in rows) / len(rows)) if rows else 0.0,
                "mean_cpu_compute_ms": float(sum(float(row["cpu_compute_ms"]) for row in rows) / len(rows)) if rows else 0.0,
                "mean_d2h_h2d_ms": float(sum(float(row["d2h_h2d_ms"]) for row in rows) / len(rows)) if rows else 0.0,
                "mean_merge_ms": float(sum(float(row["merge_ms"]) for row in rows) / len(rows)) if rows else 0.0,
                "mean_overlap_hidden_ms": float(sum(float(row["overlap_hidden_ms"]) for row in rows) / len(rows)) if rows else 0.0,
            }
        )

    return VariantSimulationResult(
        token_rows=token_rows,
        request_rows=request_rows,
        step_rows=step_rows,
        quantiles=quantiles,
        tail_threshold_ms=float(tail_threshold_ms),
    )


def build_decode_event_windows(
    stream: TokenStructuredStream,
    replay_summary: PhaseReplayBitmapSummary,
    window_size: int,
) -> List[dict]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    decode_indices = np.flatnonzero(stream.phase_ids == PHASE_DECODE)
    rows: List[dict] = []
    if decode_indices.size == 0:
        return rows
    for window_id, start in enumerate(range(0, decode_indices.size, window_size)):
        end = min(start + window_size, decode_indices.size)
        window_indices = decode_indices[start:end]
        miss_count = int(np.sum(replay_summary.miss_flags[window_indices]))
        cold_miss_count = int(np.sum(replay_summary.cold_miss_flags[window_indices]))
        total = int(window_indices.size)
        rows.append(
            {
                "window_id": int(window_id),
                "decode_events": total,
                "hit_rate": float((total - miss_count) / total) if total else 0.0,
                "cold_miss_rate": float(cold_miss_count / total) if total else 0.0,
            }
        )
    return rows


def build_cpu_utilization_windows(
    token_rows: Sequence[Mapping[str, object]],
    window_decode_events: int,
) -> List[dict]:
    if window_decode_events <= 0:
        raise ValueError(f"window_decode_events must be positive, got {window_decode_events}")
    rows: List[dict] = []
    current_events = 0
    current_cpu_ms = 0.0
    current_total_ms = 0.0
    window_id = 0
    for row in token_rows:
        current_events += int(row["decode_event_count"])
        current_cpu_ms += float(row["cpu_busy_ms"])
        current_total_ms += float(row["tpot_ms"])
        if current_events < window_decode_events:
            continue
        rows.append(
            {
                "window_id": int(window_id),
                "decode_events": int(current_events),
                "cpu_busy_ratio": float(current_cpu_ms / current_total_ms) if current_total_ms > 0.0 else 0.0,
            }
        )
        window_id += 1
        current_events = 0
        current_cpu_ms = 0.0
        current_total_ms = 0.0
    if current_events > 0:
        rows.append(
            {
                "window_id": int(window_id),
                "decode_events": int(current_events),
                "cpu_busy_ratio": float(current_cpu_ms / current_total_ms) if current_total_ms > 0.0 else 0.0,
            }
        )
    return rows


def select_tail_request(request_rows: Sequence[Mapping[str, object]]) -> Optional[int]:
    if not request_rows:
        return None
    winner = max(
        request_rows,
        key=lambda row: (
            float(row["max_tpot_ms"]),
            float(row["p99_tpot_ms"]),
            float(row["mean_tpot_ms"]),
            -int(row["req_idx"]),
        ),
    )
    return int(winner["req_idx"])
