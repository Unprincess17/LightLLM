#!/usr/bin/env python3
"""Shared helpers for calibrated system-level TPOT simulation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from analyze_locality import (
    CONDITION_EXPERT_ONLY,
    CONDITION_JOINT_CORR,
    CONDITION_JOINT_INDEP,
    CONDITION_ORDER,
    iter_jsonl_bytes,
    pack_expert_object,
    pack_joint_object,
    paired_row_alignment_fields,
    parse_adapter_slot,
)
from common import percentile
from replay_core import (
    ConditionAccessBuffer,
    PHASE_DECODE,
    PHASE_PREFILL,
    PHASE_UNKNOWN,
)


CONDITION_LABELS = {
    CONDITION_EXPERT_ONLY: "B0",
    CONDITION_JOINT_INDEP: "B1",
    CONDITION_JOINT_CORR: "B2",
}
DEFAULT_SLICE_BYTES = 256 * 1024
TRANSFER_MODE_STAGED_PAGEABLE_PACKED = "staged_pageable_packed"
TRANSFER_MODE_DIRECT_PINNED_PACKED = "direct_pinned_packed"
TRANSFER_MODE_DIRECT_PAGEABLE_FRAGMENTED = "direct_pageable_fragmented"
TRANSFER_MODE_ORDER = (
    TRANSFER_MODE_STAGED_PAGEABLE_PACKED,
    TRANSFER_MODE_DIRECT_PINNED_PACKED,
    TRANSFER_MODE_DIRECT_PAGEABLE_FRAGMENTED,
)
CURVE_STAT_KEYS = ("mean_ms", "p50_ms", "p90_ms")


@dataclass(frozen=True)
class DecodeLayerStep:
    request_ordinal: int
    req_idx: int
    token_ordinal: int
    token_pos: int
    layer_id: int
    event_idx: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class DecodeToken:
    request_ordinal: int
    req_idx: int
    token_ordinal: int
    token_pos: int
    step_start: int
    step_end: int


@dataclass(frozen=True)
class TokenStructuredStream:
    request_ids: List[int]
    request_offsets: np.ndarray
    condition_buffers: Dict[str, ConditionAccessBuffer]
    phase_ids: np.ndarray
    per_request_prefill_events: np.ndarray
    per_request_decode_events: np.ndarray
    decode_layer_steps: List[DecodeLayerStep]
    decode_tokens: List[DecodeToken]
    request_arrival_idx: np.ndarray
    request_start_ts: np.ndarray
    total_events: int
    phase_available: bool
    position_field_counts: Dict[str, int]
    invariant_checked_fields: List[str]


@dataclass(frozen=True)
class ComponentSizeBytes:
    early_component_label: str
    late_component_label: str
    early_object_bytes: int
    late_object_bytes: int
    total_object_bytes: int
    early_object_slices: int
    late_object_slices: int
    total_object_slices: int
    gate_lora_excluded: bool
    slice_bytes: int


@dataclass(frozen=True)
class LayerTransferDecision:
    strategy: str
    miss_objects: int
    cold_miss_objects: int
    capacity_miss_objects: int
    early_bytes: int
    late_bytes: int
    total_bytes: int
    early_slices: int
    late_slices: int
    total_slices: int
    early_service_ms: float
    late_service_ms: float
    whole_service_ms: float
    early_overlap_window_ms: float
    late_overlap_window_ms: float
    early_exposed_stall_ms: float
    late_exposed_stall_ms: float
    total_exposed_stall_ms: float


@dataclass(frozen=True)
class TokenTPOTResult:
    req_idx: int
    token_pos: int
    token_ordinal: int
    tpot_ms: float
    exposed_miss_latency_ms: float
    miss_objects: int
    cold_miss_objects: int
    capacity_miss_objects: int
    miss_bytes: int
    miss_slices: int


@dataclass(frozen=True)
class SchedulerBatchStats:
    request_ids: Tuple[int, ...]
    token_ordinals: Tuple[int, ...]
    batch_start_ms: float
    batch_duration_ms: float
    scheduler_queue_wait_ms: float
    pcie_queue_wait_ms: float
    active_request_count: int


@dataclass(frozen=True)
class TokenTemplateLayer:
    layer_id: int
    miss_objects: int
    cold_miss_objects: int


@dataclass(frozen=True)
class TokenTemplate:
    req_idx: int
    token_pos: int
    token_ordinal: int
    layers: Tuple[TokenTemplateLayer, ...]


def phase_name_to_id(phase: object) -> int:
    phase_name = str(phase or "").strip().lower()
    if phase_name == "prefill":
        return PHASE_PREFILL
    if phase_name == "decode":
        return PHASE_DECODE
    return PHASE_UNKNOWN


def dtype_name_to_torch_dtype(raw_dtype: str) -> torch.dtype:
    normalized = str(raw_dtype).strip().lower()
    if normalized in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half", "torch.float16"):
        return torch.float16
    if normalized in ("fp32", "float32", "torch.float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype name: {raw_dtype}")


def torch_dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float16:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype for size accounting: {dtype}")


def load_model_text_config(model_dir: Path) -> dict:
    payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("text_config"), Mapping):
        return dict(payload["text_config"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"unexpected model config format in {model_dir / 'config.json'}")


def parse_batch_grid(raw_value: object) -> List[int]:
    if isinstance(raw_value, int):
        return [int(raw_value)]
    if isinstance(raw_value, (list, tuple)):
        return [int(value) for value in raw_value]
    return [int(token.strip()) for token in str(raw_value).split(",") if token.strip()]


def parse_batch_value_map(raw_value: object) -> Dict[int, float]:
    if isinstance(raw_value, Mapping):
        return {int(key): float(value) for key, value in raw_value.items()}
    values: Dict[int, float] = {}
    for token in str(raw_value).split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"expected batch:value entry, got {item!r}")
        batch_token, value_token = item.split(":", 1)
        values[int(batch_token.strip())] = float(value_token.strip())
    if not values:
        raise ValueError("batch value map must not be empty")
    return values


def compute_component_sizes(
    hidden_size: int,
    moe_intermediate_size: int,
    lora_rank: int,
    dtype: torch.dtype,
    slice_bytes: int = DEFAULT_SLICE_BYTES,
    early_component_label: str = "up_proj",
    late_component_label: str = "down_proj",
    gate_lora_excluded: bool = True,
) -> ComponentSizeBytes:
    dtype_nbytes = torch_dtype_nbytes(dtype)
    early_object_bytes = int(lora_rank * (hidden_size + moe_intermediate_size) * dtype_nbytes)
    late_object_bytes = int(lora_rank * (moe_intermediate_size + hidden_size) * dtype_nbytes)
    total_object_bytes = int(early_object_bytes + late_object_bytes)
    return ComponentSizeBytes(
        early_component_label=str(early_component_label),
        late_component_label=str(late_component_label),
        early_object_bytes=early_object_bytes,
        late_object_bytes=late_object_bytes,
        total_object_bytes=total_object_bytes,
        early_object_slices=int(math.ceil(float(early_object_bytes) / float(slice_bytes))),
        late_object_slices=int(math.ceil(float(late_object_bytes) / float(slice_bytes))),
        total_object_slices=int(math.ceil(float(total_object_bytes) / float(slice_bytes))),
        gate_lora_excluded=bool(gate_lora_excluded),
        slice_bytes=int(slice_bytes),
    )


def _interp_from_rows(rows: Sequence[Mapping[str, object]], x_key: str, y_key: str, x_value: float) -> float:
    if x_value <= 0.0:
        return 0.0
    sorted_rows = sorted(rows, key=lambda row: float(row[x_key]))
    if not sorted_rows:
        raise ValueError(f"curve rows for {x_key}->{y_key} must not be empty")
    first_x = float(sorted_rows[0][x_key])
    if x_value < first_x - 1e-9:
        raise ValueError(f"curve lookup underflow: requested {x_key}={x_value}, min={first_x}")
    last_x = float(sorted_rows[-1][x_key])
    if x_value > last_x + 1e-9:
        raise ValueError(f"curve lookup overflow: requested {x_key}={x_value}, max={last_x}")
    for index, row in enumerate(sorted_rows):
        row_x = float(row[x_key])
        if abs(row_x - x_value) <= 1e-9:
            return float(row[y_key])
        if row_x > x_value:
            lower = sorted_rows[index - 1]
            upper = row
            lower_x = float(lower[x_key])
            upper_x = float(upper[x_key])
            weight = (x_value - lower_x) / max(upper_x - lower_x, 1e-9)
            return float(lower[y_key]) * (1.0 - weight) + float(upper[y_key]) * weight
    return float(sorted_rows[-1][y_key])


def calibration_service_ms(
    calibration: Mapping[str, object],
    transfer_mode: str,
    load_profile: str,
    payload_bytes: int,
    stat_key: str,
) -> float:
    if payload_bytes <= 0:
        return 0.0
    if stat_key not in CURVE_STAT_KEYS:
        raise ValueError(f"unsupported curve stat key: {stat_key}")
    transfer_curves = calibration.get("transfer_modes", {})
    if transfer_mode not in transfer_curves:
        raise ValueError(f"transfer mode {transfer_mode!r} is missing from calibration")
    mode_payload = transfer_curves[transfer_mode]
    profiles = mode_payload.get("profiles", {})
    if load_profile not in profiles:
        raise ValueError(f"load profile {load_profile!r} is missing for mode {transfer_mode!r}")
    profile_payload = profiles[load_profile]

    if transfer_mode == TRANSFER_MODE_STAGED_PAGEABLE_PACKED:
        gather_rows = profile_payload.get("packed_gather_rows", [])
        h2d_rows = profile_payload.get("packed_h2d_rows", [])
        launch_rows = profile_payload.get("packed_launch_rows", [])
        return (
            _interp_from_rows(gather_rows, "payload_bytes", stat_key, float(payload_bytes))
            + _interp_from_rows(h2d_rows, "payload_bytes", stat_key, float(payload_bytes))
            + _interp_from_rows(launch_rows, "payload_bytes", stat_key, float(payload_bytes))
        )
    if transfer_mode == TRANSFER_MODE_DIRECT_PINNED_PACKED:
        h2d_rows = profile_payload.get("packed_h2d_rows", [])
        launch_rows = profile_payload.get("packed_launch_rows", [])
        return (
            _interp_from_rows(h2d_rows, "payload_bytes", stat_key, float(payload_bytes))
            + _interp_from_rows(launch_rows, "payload_bytes", stat_key, float(payload_bytes))
        )
    if transfer_mode == TRANSFER_MODE_DIRECT_PAGEABLE_FRAGMENTED:
        slice_bytes = int(calibration["object_sizes"]["slice_bytes"])
        slice_count = int(math.ceil(float(payload_bytes) / float(slice_bytes)))
        fragmented_rows = profile_payload.get("fragmented_total_rows", [])
        return _interp_from_rows(fragmented_rows, "slice_count", stat_key, float(slice_count))
    raise ValueError(f"unsupported transfer mode: {transfer_mode}")


def calibration_base_tpot_ms(calibration: Mapping[str, object], system_batch: int) -> float:
    values = calibration.get("base_tpot_ms", {})
    if str(system_batch) in values:
        return float(values[str(system_batch)])
    if int(system_batch) in values:
        return float(values[int(system_batch)])
    raise ValueError(f"base_tpot_ms is missing system_batch={system_batch}")


def calibration_overlap_window_ms(
    calibration: Mapping[str, object],
    system_batch: int,
    window_name: str,
    stat_key: str,
) -> float:
    if stat_key not in CURVE_STAT_KEYS:
        raise ValueError(f"unsupported curve stat key: {stat_key}")
    windows = calibration.get("overlap_windows_ms", {})
    batch_payload = windows.get(str(system_batch), windows.get(system_batch))
    if not isinstance(batch_payload, Mapping):
        raise ValueError(f"overlap windows are missing system_batch={system_batch}")
    window_payload = batch_payload.get(window_name)
    if not isinstance(window_payload, Mapping):
        raise ValueError(f"overlap window {window_name!r} is missing for system_batch={system_batch}")
    return float(window_payload[stat_key])


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = [float(value) for value in values]
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "p50": float(percentile(ordered, 0.50)),
        "p90": float(percentile(ordered, 0.90)),
        "p95": float(percentile(ordered, 0.95)),
        "p99": float(percentile(ordered, 0.99)),
        "max": float(max(ordered)),
    }


def choose_layer_transfer_strategy(
    miss_objects: int,
    cold_miss_objects: int,
    object_sizes: ComponentSizeBytes,
    calibration: Mapping[str, object],
    transfer_mode: str,
    load_profile: str,
    stat_key: str,
    system_batch: int,
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
            early_overlap_window_ms=calibration_overlap_window_ms(calibration, system_batch, "early", stat_key),
            late_overlap_window_ms=calibration_overlap_window_ms(calibration, system_batch, "late", stat_key),
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
    early_overlap_window_ms = calibration_overlap_window_ms(calibration, system_batch, "early", stat_key)
    late_overlap_window_ms = calibration_overlap_window_ms(calibration, system_batch, "late", stat_key)

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


def build_system_streams(
    joined_indep_path: Path,
    joined_corr_path: Path,
    total_events: int,
    progress_every: int,
) -> TokenStructuredStream:
    access_buffers = {
        CONDITION_EXPERT_ONLY: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_INDEP: np.empty(total_events, dtype=np.uint32),
        CONDITION_JOINT_CORR: np.empty(total_events, dtype=np.uint32),
    }
    phase_ids = np.empty(total_events, dtype=np.uint8)
    max_object_ids = {condition: 0 for condition in CONDITION_ORDER}

    request_offsets = [0]
    request_ids: List[int] = []
    request_arrival_idx: List[int] = []
    request_start_ts: List[int] = []
    per_request_prefill_events: List[int] = []
    per_request_decode_events: List[int] = []
    current_prefill_events = 0
    current_decode_events = 0
    current_req_idx: Optional[int] = None
    current_request_ordinal = -1
    current_token_ordinal = 0
    phase_available = True
    position_field_counts: Dict[str, int] = {}
    invariant_checked_fields: List[str] = []

    decode_layer_steps: List[DecodeLayerStep] = []
    decode_tokens: List[DecodeToken] = []
    active_step_req_idx: Optional[int] = None
    active_step_token_pos: Optional[int] = None
    active_step_layer_id: Optional[int] = None
    active_step_event_idx: Optional[int] = None
    active_step_start_index: Optional[int] = None
    active_step_request_ordinal = -1
    active_step_token_ordinal = -1
    active_token_req_idx: Optional[int] = None
    active_token_pos: Optional[int] = None
    active_token_step_start = 0
    active_token_request_ordinal = -1
    active_token_ordinal = -1

    def flush_active_step(end_index: int) -> None:
        nonlocal active_step_req_idx, active_step_token_pos, active_step_layer_id, active_step_event_idx
        nonlocal active_step_start_index, active_step_request_ordinal, active_step_token_ordinal
        if active_step_start_index is None:
            return
        decode_layer_steps.append(
            DecodeLayerStep(
                request_ordinal=int(active_step_request_ordinal),
                req_idx=int(active_step_req_idx),
                token_ordinal=int(active_step_token_ordinal),
                token_pos=int(active_step_token_pos),
                layer_id=int(active_step_layer_id),
                event_idx=int(active_step_event_idx),
                start_index=int(active_step_start_index),
                end_index=int(end_index),
            )
        )
        active_step_req_idx = None
        active_step_token_pos = None
        active_step_layer_id = None
        active_step_event_idx = None
        active_step_start_index = None
        active_step_request_ordinal = -1
        active_step_token_ordinal = -1

    def flush_active_token() -> None:
        nonlocal active_token_req_idx, active_token_pos, active_token_step_start
        nonlocal active_token_request_ordinal, active_token_ordinal
        if active_token_req_idx is None:
            return
        decode_tokens.append(
            DecodeToken(
                request_ordinal=int(active_token_request_ordinal),
                req_idx=int(active_token_req_idx),
                token_ordinal=int(active_token_ordinal),
                token_pos=int(active_token_pos),
                step_start=int(active_token_step_start),
                step_end=len(decode_layer_steps),
            )
        )
        active_token_req_idx = None
        active_token_pos = None
        active_token_request_ordinal = -1
        active_token_ordinal = -1
        active_token_step_start = len(decode_layer_steps)

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
            current_request_ordinal = 0
            request_ids.append(req_idx)
            request_arrival_idx.append(int(indep_row.get("arrival_idx", req_idx)))
            request_start_ts.append(int(indep_row["start_ts"]) if indep_row.get("start_ts") is not None else -1)
        elif req_idx < current_req_idx:
            raise ValueError(
                "joined trace request order regressed; system TPOT replay requires canonical non-decreasing req_idx order: "
                f"previous={current_req_idx}, current={req_idx}"
            )
        elif req_idx > current_req_idx:
            if req_idx != current_req_idx + 1:
                raise ValueError(
                    "joined trace req_idx order is not contiguous; system TPOT replay requires the same canonical request stream "
                    f"across conditions: previous={current_req_idx}, current={req_idx}"
                )
            flush_active_step(rows_seen - 1)
            flush_active_token()
            request_offsets.append(rows_seen - 1)
            per_request_prefill_events.append(current_prefill_events)
            per_request_decode_events.append(current_decode_events)
            current_prefill_events = 0
            current_decode_events = 0
            current_req_idx = req_idx
            current_request_ordinal += 1
            current_token_ordinal = 0
            request_ids.append(req_idx)
            request_arrival_idx.append(int(indep_row.get("arrival_idx", req_idx)))
            request_start_ts.append(int(indep_row["start_ts"]) if indep_row.get("start_ts") is not None else -1)

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

        if phase_id == PHASE_DECODE:
            token_pos = int(indep_row.get("token_pos", indep_row.get("chunk_idx", -1)))
            event_idx = int(indep_row["event_idx"])
            step_key = (req_idx, token_pos, layer_id, event_idx)
            token_key = (req_idx, token_pos)
            active_step_key = (
                active_step_req_idx,
                active_step_token_pos,
                active_step_layer_id,
                active_step_event_idx,
            )
            if active_step_start_index is None:
                active_token_req_idx = req_idx
                active_token_pos = token_pos
                active_token_step_start = len(decode_layer_steps)
                active_token_request_ordinal = current_request_ordinal
                active_token_ordinal = current_token_ordinal
                active_step_req_idx = req_idx
                active_step_token_pos = token_pos
                active_step_layer_id = layer_id
                active_step_event_idx = event_idx
                active_step_start_index = buffer_index
                active_step_request_ordinal = current_request_ordinal
                active_step_token_ordinal = current_token_ordinal
            elif step_key != active_step_key:
                flush_active_step(buffer_index)
                if token_key != (active_token_req_idx, active_token_pos):
                    flush_active_token()
                    current_token_ordinal += 1
                    active_token_req_idx = req_idx
                    active_token_pos = token_pos
                    active_token_step_start = len(decode_layer_steps)
                    active_token_request_ordinal = current_request_ordinal
                    active_token_ordinal = current_token_ordinal
                active_step_req_idx = req_idx
                active_step_token_pos = token_pos
                active_step_layer_id = layer_id
                active_step_event_idx = event_idx
                active_step_start_index = buffer_index
                active_step_request_ordinal = current_request_ordinal
                active_step_token_ordinal = current_token_ordinal

        if progress_every > 0 and rows_seen % progress_every == 0:
            print(f"materialized {rows_seen}/{total_events} aligned cache-access objects for system TPOT replay")

    if rows_seen != total_events:
        raise ValueError(
            f"joined trace row count mismatch against join_qc_report.json: expected {total_events}, observed {rows_seen}"
        )

    flush_active_step(total_events)
    flush_active_token()

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

    return TokenStructuredStream(
        request_ids=request_ids,
        request_offsets=np.asarray(request_offsets, dtype=np.int64),
        condition_buffers=condition_buffers,
        phase_ids=phase_ids,
        per_request_prefill_events=np.asarray(per_request_prefill_events, dtype=np.int64),
        per_request_decode_events=np.asarray(per_request_decode_events, dtype=np.int64),
        decode_layer_steps=decode_layer_steps,
        decode_tokens=decode_tokens,
        request_arrival_idx=np.asarray(request_arrival_idx, dtype=np.int64),
        request_start_ts=np.asarray(request_start_ts, dtype=np.int64),
        total_events=total_events,
        phase_available=phase_available,
        position_field_counts=position_field_counts,
        invariant_checked_fields=invariant_checked_fields,
    )


def build_token_templates(
    stream: TokenStructuredStream,
    miss_flags: np.ndarray,
    cold_miss_flags: np.ndarray,
) -> Dict[int, List[TokenTemplate]]:
    templates: Dict[int, List[TokenTemplate]] = {int(req_idx): [] for req_idx in stream.request_ids}
    for token in stream.decode_tokens:
        layers: List[TokenTemplateLayer] = []
        for step in stream.decode_layer_steps[token.step_start:token.step_end]:
            miss_objects = int(np.sum(miss_flags[step.start_index:step.end_index]))
            cold_miss_objects = int(np.sum(cold_miss_flags[step.start_index:step.end_index]))
            layers.append(
                TokenTemplateLayer(
                    layer_id=int(step.layer_id),
                    miss_objects=miss_objects,
                    cold_miss_objects=cold_miss_objects,
                )
            )
        templates[int(token.req_idx)].append(
            TokenTemplate(
                req_idx=int(token.req_idx),
                token_pos=int(token.token_pos),
                token_ordinal=int(token.token_ordinal),
                layers=tuple(layers),
            )
        )
    return templates
