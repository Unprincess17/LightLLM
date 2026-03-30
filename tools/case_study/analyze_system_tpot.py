#!/usr/bin/env python3
"""Compute calibrated system-level TPOT from B8 cache replay outputs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from analyze_locality import (
    CONDITION_ORDER,
    OBJECT_KEY_DEFINITIONS,
    OBJECT_KEY_NOTES,
    resolve_total_events,
)
from common import (
    ensure_dir,
    ensure_parent_dir,
    finalize_condition_merged_csv,
    load_global_config,
    load_seed_config,
    parse_cardinalities,
    parse_condition_subset,
    prepare_condition_merged_csv,
    read_condition_filtered_csv_rows,
    stage_output_dir,
    write_json,
)
from replay_core import simulate_phase_replay_with_bitmaps
from system_tpot_core import (
    CONDITION_LABELS,
    MISS_HANDLING_EXECUTION_FIRST,
    MISS_HANDLING_LOAD_THEN_RUN,
    MISS_HANDLING_NO_CPU_PATH,
    MISS_HANDLING_NO_DEFERRED_SYNC,
    MISS_HANDLING_MODE_ORDER,
    OVERLAP_POLICY_CALIBRATED,
    OVERLAP_POLICY_ORDER,
    TRANSFER_MODE_ORDER,
    TRANSFER_MODE_STAGED_PAGEABLE_PACKED,
    apply_temporal_prefetch_hits,
    build_replay_policy_state,
    build_system_streams,
    build_token_templates,
    compute_component_sizes,
    dtype_name_to_torch_dtype,
    load_model_text_config,
    simulate_miss_path_replay_with_bitmaps,
    validate_execution_first_calibration,
)
from system_tpot_sim import (
    compute_stage1_token_tpot,
    resolve_request_arrival_times_ms,
    simulate_scheduler_sensitivity,
    validate_against_b8_misses,
)


DEFAULT_PROGRESS_EVERY_ROWS = 1_000_000
DEFAULT_TAIL_QUANTILE = 0.95
DEFAULT_STAGE1_SYSTEM_BATCH = 1
DEFAULT_SCHEDULER_BATCH_GRID = (1, 2, 4)
DEFAULT_CALIBRATION_STAT = "p50_ms"
TOKEN_TPOT_FIELDS = [
    "condition",
    "condition_label",
    "cache_budget",
    "miss_handling_mode",
    "overlap_policy",
    "req_idx",
    "token_pos",
    "token_ordinal",
    "base_tpot_ms",
    "tpot_ms",
    "exposed_miss_latency_ms",
    "miss_objects",
    "cold_miss_objects",
    "capacity_miss_objects",
    "prefetched_miss_objects",
    "miss_bytes",
    "miss_slices",
    "layer_count",
]
TPOT_QUANTILE_FIELDS = [
    "condition",
    "condition_label",
    "cache_budget",
    "miss_handling_mode",
    "overlap_policy",
    "mean",
    "p50",
    "p90",
    "p95",
    "p99",
    "max",
    "tail_threshold_ms",
    "token_count",
]
TAIL_TOKEN_FIELDS = TOKEN_TPOT_FIELDS
LAYER_BARRIER_FIELDS = [
    "condition",
    "condition_label",
    "cache_budget",
    "miss_handling_mode",
    "overlap_policy",
    "layer_id",
    "steps",
    "mean_miss_objects",
    "p95_miss_objects",
    "mean_prefetched_miss_objects",
    "mean_miss_bytes",
    "mean_miss_slices",
    "mean_exposed_stall_ms",
    "p95_exposed_stall_ms",
    "mean_early_service_ms",
    "mean_late_service_ms",
    "mean_whole_service_ms",
    "mean_early_overlap_window_ms",
    "mean_late_overlap_window_ms",
    "share_hit_only",
    "share_whole_object_early",
    "share_staged_split",
    "share_blocking_promotion_first",
    "share_execution_first_sync_promotion",
]
REQUEST_DECODE_FIELDS = [
    "condition",
    "condition_label",
    "cache_budget",
    "miss_handling_mode",
    "overlap_policy",
    "req_idx",
    "decode_tokens",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "max_tpot_ms",
    "mean_exposed_miss_latency_ms",
    "total_miss_objects",
    "total_cold_miss_objects",
    "total_capacity_miss_objects",
    "total_prefetched_miss_objects",
]
SCHEDULER_FIELDS = [
    "condition",
    "condition_label",
    "cache_budget",
    "miss_handling_mode",
    "overlap_policy",
    "system_batch",
    "token_count",
    "batch_count",
    "mean_service_tpot_ms",
    "p50_service_tpot_ms",
    "p95_service_tpot_ms",
    "p99_service_tpot_ms",
    "max_service_tpot_ms",
    "mean_completion_interval_ms",
    "p50_completion_interval_ms",
    "p95_completion_interval_ms",
    "p99_completion_interval_ms",
    "max_completion_interval_ms",
    "mean_scheduler_queue_wait_ms",
    "p95_scheduler_queue_wait_ms",
    "mean_pcie_queue_wait_ms",
    "p95_pcie_queue_wait_ms",
    "mean_active_batch_size",
]
BACKGROUND_POLICY_FIELDS = [
    "condition",
    "condition_label",
    "cache_budget",
    "miss_handling_mode",
    "overlap_policy",
    "deferred_promotion_delta_steps",
    "promotion_admitted",
    "promotion_hits",
    "prefetch_predictions",
    "prefetch_matches",
    "prefetch_false_positives",
    "prefetch_miss_hits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute calibrated TPOT from trace-driven cache replay outputs")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory for replay/system_tpot artifacts")
    parser.add_argument("--joined_indep_path", type=str, default=None, help="Override joined_trace_indep.jsonl input path")
    parser.add_argument("--joined_corr_path", type=str, default=None, help="Override joined_trace_corr.jsonl input path")
    parser.add_argument("--qc_report_path", type=str, default=None, help="Override join_qc_report.json input path")
    parser.add_argument("--cache_curve_path", type=str, default=None, help="Override replay/cache/cache_curve.csv input path")
    parser.add_argument("--per_request_miss_path", type=str, default=None, help="Override replay/cache/per_request_miss_count.csv input path")
    parser.add_argument("--calibration_path", type=str, required=True, help="Calibration JSON emitted by calibrate_system_baseline.py")
    parser.add_argument("--model_dir", type=str, default=None, help="Override HF model directory used for dimension lookup")
    parser.add_argument("--cache_budgets", type=str, default=None, help="Optional comma-separated cache-budget override")
    parser.add_argument("--transfer_mode", type=str, default=TRANSFER_MODE_STAGED_PAGEABLE_PACKED, choices=list(TRANSFER_MODE_ORDER), help="Transfer mode from the calibration manifest")
    parser.add_argument("--load_profile", type=str, default="stressed", help="Load profile from the calibration manifest")
    parser.add_argument("--calibration_stat", type=str, default=DEFAULT_CALIBRATION_STAT, choices=["mean_ms", "p50_ms", "p90_ms"], help="Statistic to pull from the calibration curves")
    parser.add_argument("--miss_handling_mode", type=str, default=MISS_HANDLING_LOAD_THEN_RUN, choices=list(MISS_HANDLING_MODE_ORDER), help="Miss handling model: baseline load-then-run or COLoRA execution-first")
    parser.add_argument("--overlap_policy", type=str, default=OVERLAP_POLICY_CALIBRATED, choices=list(OVERLAP_POLICY_ORDER), help="Whether calibrated overlap windows are active for replay service modeling")
    parser.add_argument("--deferred_promotion_delta_steps", type=int, default=4, help="Static decode-step reuse threshold delta for execution-first deferred promotion; 0 disables promotion admission")
    parser.add_argument("--temporal_prefetch", action="store_true", help="Enable the lightweight previous-top1 temporal prefetch model for execution-first mode")
    parser.add_argument("--temporal_prefetch_cpu_discount", type=float, default=0.20, help="Fractional cold-path CPU-time discount applied to prefetched miss objects in execution-first mode")
    parser.add_argument("--stage1_system_batch", type=int, default=DEFAULT_STAGE1_SYSTEM_BATCH, help="Effective decode batch for stage-1 per-sequence TPOT")
    parser.add_argument("--scheduler_batch_grid", type=str, default=",".join(str(value) for value in DEFAULT_SCHEDULER_BATCH_GRID), help="Comma-separated max system-batch sweep for scheduler sensitivity")
    parser.add_argument("--skip_scheduler", action="store_true", help="Skip stage-2 scheduler sensitivity")
    parser.add_argument("--tail_quantile", type=float, default=DEFAULT_TAIL_QUANTILE, help="Tail threshold quantile for tail_token_breakdown.csv")
    parser.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY_ROWS, help="Print progress every N aligned rows while materializing the system stream")
    parser.add_argument("--dtype", type=str, default="bf16", help="LoRA dtype used for byte accounting")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank used for byte accounting")
    parser.add_argument("--slice_bytes", type=int, default=256 * 1024, help="Audit slice size in bytes")
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated subset of conditions to recompute: expert_only,joint_indep,joint_corr",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def load_cache_budget_grid_from_curve(path: Path) -> List[int]:
    budgets: List[int] = []
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
    expected: Dict[Tuple[str, int, int], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            expected[(row["condition"], int(row["cache_budget"]), int(row["req_idx"]))] = int(row["miss_count"])
    return expected


def open_csv_writer(path: Path, fieldnames: Sequence[str]):
    ensure_parent_dir(path)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
    writer.writeheader()
    return handle, writer


def write_condition_merged_csv(
    output_path: Path,
    fieldnames: Sequence[str],
    selected_conditions: Sequence[str],
    preserve_existing: bool,
    rows: Sequence[Mapping[str, object]],
) -> None:
    temp_path, handle, writer = prepare_condition_merged_csv(
        output_path=output_path,
        fieldnames=fieldnames,
        selected_conditions=selected_conditions,
        preserve_existing=preserve_existing,
    )
    try:
        writer.writerows(rows)
    finally:
        handle.close()
    finalize_condition_merged_csv(temp_path, output_path)


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config()

    output_dir = stage_output_dir("replay/system_tpot", config, args.run_id, args.output_dir)
    joined_dir = stage_output_dir("joined_trace", config, args.run_id)
    replay_cache_dir = stage_output_dir("replay/cache", config, args.run_id)

    joined_indep_path = Path(args.joined_indep_path) if args.joined_indep_path else joined_dir / "joined_trace_indep.jsonl"
    joined_corr_path = Path(args.joined_corr_path) if args.joined_corr_path else joined_dir / "joined_trace_corr.jsonl"
    qc_report_path = Path(args.qc_report_path) if args.qc_report_path else joined_dir / "join_qc_report.json"
    cache_curve_path = Path(args.cache_curve_path) if args.cache_curve_path else replay_cache_dir / "cache_curve.csv"
    per_request_miss_path = Path(args.per_request_miss_path) if args.per_request_miss_path else replay_cache_dir / "per_request_miss_count.csv"
    calibration_path = Path(args.calibration_path)
    model_dir = Path(args.model_dir) if args.model_dir else Path(config.get("paths", {}).get("model_dir"))

    if args.cache_budgets is not None:
        cache_budgets = parse_cardinalities(args.cache_budgets)
    else:
        cache_budgets = load_cache_budget_grid_from_curve(cache_curve_path)
    scheduler_batch_grid = parse_cardinalities(args.scheduler_batch_grid)
    if not scheduler_batch_grid:
        raise ValueError("scheduler_batch_grid must not be empty")
    selected_conditions = parse_condition_subset(args.conditions, CONDITION_ORDER)
    preserve_existing = len(selected_conditions) != len(CONDITION_ORDER)

    token_output_path = output_dir / "token_tpot.csv"
    quantiles_output_path = output_dir / "tpot_quantiles.csv"
    tail_output_path = output_dir / "tail_token_breakdown.csv"
    layer_output_path = output_dir / "layer_barrier_breakdown.csv"
    request_output_path = output_dir / "request_decode_summary.csv"
    scheduler_output_path = output_dir / "scheduler_sensitivity.csv"
    background_output_path = output_dir / "background_policy_summary.csv"
    manifest_output_path = output_dir / "system_tpot_manifest.json"

    calibration = load_json(calibration_path)
    effective_overlap_policy = (
        str(args.overlap_policy)
        if str(args.miss_handling_mode) != MISS_HANDLING_NO_CPU_PATH
        else "disabled"
    )
    if str(args.miss_handling_mode) in (MISS_HANDLING_EXECUTION_FIRST, MISS_HANDLING_NO_DEFERRED_SYNC):
        validate_execution_first_calibration(
            calibration=calibration,
            load_profile=str(args.load_profile),
            require_overlap_windows=effective_overlap_policy == OVERLAP_POLICY_CALIBRATED,
            tool_name="tools/case_study/analyze_system_tpot.py",
        )
    qc_payload = load_json(qc_report_path)
    total_events = resolve_total_events(qc_payload)
    expected_b8_misses = load_expected_per_request_misses(per_request_miss_path) if per_request_miss_path.exists() else None
    existing_manifest = load_json(manifest_output_path) if preserve_existing else {}

    text_config = load_model_text_config(model_dir)
    object_sizes = compute_component_sizes(
        hidden_size=int(text_config["hidden_size"]),
        moe_intermediate_size=int(text_config["moe_intermediate_size"]),
        lora_rank=int(args.lora_rank),
        dtype=dtype_name_to_torch_dtype(args.dtype),
        slice_bytes=int(args.slice_bytes),
    )

    stream = build_system_streams(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        total_events=total_events,
        progress_every=int(args.progress_every),
    )
    request_arrival_times_ms = resolve_request_arrival_times_ms(stream)

    policy_states = {
        condition: (
            build_replay_policy_state(
                stream=stream,
                access_buffer=stream.condition_buffers[condition],
                enable_temporal_prefetch=bool(args.temporal_prefetch),
            )
            if str(args.miss_handling_mode) == MISS_HANDLING_EXECUTION_FIRST
            else None
        )
        for condition in CONDITION_ORDER
    }

    token_temp_path, token_handle, token_writer = prepare_condition_merged_csv(
        output_path=token_output_path,
        fieldnames=TOKEN_TPOT_FIELDS,
        selected_conditions=selected_conditions,
        preserve_existing=preserve_existing,
    )
    try:
        quantile_rows: List[dict] = (
            read_condition_filtered_csv_rows(quantiles_output_path, selected_conditions) if preserve_existing else []
        )
        tail_rows: List[dict] = (
            read_condition_filtered_csv_rows(tail_output_path, selected_conditions) if preserve_existing else []
        )
        layer_rows: List[dict] = (
            read_condition_filtered_csv_rows(layer_output_path, selected_conditions) if preserve_existing else []
        )
        request_rows: List[dict] = (
            read_condition_filtered_csv_rows(request_output_path, selected_conditions) if preserve_existing else []
        )
        scheduler_rows: List[dict] = (
            read_condition_filtered_csv_rows(scheduler_output_path, selected_conditions) if preserve_existing else []
        )
        background_policy_rows: List[dict] = (
            read_condition_filtered_csv_rows(background_output_path, selected_conditions) if preserve_existing else []
        )
        per_condition_payloads: Dict[str, dict] = {}
        if preserve_existing:
            existing_conditions = existing_manifest.get("conditions", {})
            if not isinstance(existing_conditions, Mapping):
                raise ValueError(f"expected conditions map in {manifest_output_path}")
            for condition in CONDITION_ORDER:
                if condition in selected_conditions:
                    continue
                if condition not in existing_conditions:
                    raise KeyError(f"missing preserved condition {condition} in {manifest_output_path}")
                per_condition_payloads[condition] = dict(existing_conditions[condition])

        for condition in CONDITION_ORDER:
            if condition not in selected_conditions:
                continue
            print(f"replaying calibrated TPOT for {condition}")
            access_buffer = stream.condition_buffers[condition]
            budget_payloads: Dict[str, dict] = {}
            policy_state = policy_states[condition]
            for cache_budget in cache_budgets:
                print(f"  replay budget={cache_budget}")
                replay_summary = simulate_miss_path_replay_with_bitmaps(
                    condition_buffer=access_buffer,
                    stream=stream,
                    cache_budget=int(cache_budget),
                    miss_handling_mode=str(args.miss_handling_mode),
                    deferred_promotion_delta_steps=int(args.deferred_promotion_delta_steps),
                    policy_state=policy_state,
                )
                if str(args.miss_handling_mode) in (
                    MISS_HANDLING_LOAD_THEN_RUN,
                    MISS_HANDLING_NO_CPU_PATH,
                    MISS_HANDLING_NO_DEFERRED_SYNC,
                ):
                    validate_against_b8_misses(
                        condition=condition,
                        cache_budget=int(cache_budget),
                        request_ids=stream.request_ids,
                        observed_per_request_misses=replay_summary.per_request_misses,
                        expected_b8_misses=expected_b8_misses,
                    )
                replay_summary = apply_temporal_prefetch_hits(
                    replay_summary=replay_summary,
                    plan=policy_state.temporal_prefetch_plan if bool(args.temporal_prefetch) else None,
                )
                stage1 = compute_stage1_token_tpot(
                    condition=condition,
                    cache_budget=int(cache_budget),
                    stream=stream,
                    replay_summary=replay_summary,
                    calibration=calibration,
                    object_sizes=object_sizes,
                    transfer_mode=args.transfer_mode,
                    load_profile=args.load_profile,
                    stat_key=args.calibration_stat,
                    system_batch=int(args.stage1_system_batch),
                    tail_quantile=float(args.tail_quantile),
                    miss_handling_mode=str(args.miss_handling_mode),
                    prefetch_cpu_discount=float(args.temporal_prefetch_cpu_discount),
                    overlap_policy=effective_overlap_policy,
                )
                for row in stage1.token_rows:
                    token_writer.writerow(row)
                quantile_rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "cache_budget": int(cache_budget),
                        "miss_handling_mode": str(args.miss_handling_mode),
                        "overlap_policy": effective_overlap_policy,
                        "mean": float(stage1.quantiles["mean"]),
                        "p50": float(stage1.quantiles["p50"]),
                        "p90": float(stage1.quantiles["p90"]),
                        "p95": float(stage1.quantiles["p95"]),
                        "p99": float(stage1.quantiles["p99"]),
                        "max": float(stage1.quantiles["max"]),
                        "tail_threshold_ms": float(stage1.tail_threshold_ms),
                        "token_count": int(len(stage1.token_rows)),
                    }
                )
                tail_rows.extend(stage1.tail_rows)
                layer_rows.extend(stage1.layer_rows)
                request_rows.extend(stage1.request_rows)
                background_policy_rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "cache_budget": int(cache_budget),
                        "miss_handling_mode": str(args.miss_handling_mode),
                        "overlap_policy": effective_overlap_policy,
                        "deferred_promotion_delta_steps": int(args.deferred_promotion_delta_steps),
                        "promotion_admitted": int(replay_summary.promotion_admitted),
                        "promotion_hits": int(replay_summary.promotion_hits),
                        "prefetch_predictions": int(replay_summary.prefetch_predictions),
                        "prefetch_matches": int(replay_summary.prefetch_matches),
                        "prefetch_false_positives": int(replay_summary.prefetch_false_positives),
                        "prefetch_miss_hits": int(replay_summary.prefetch_miss_hits),
                    }
                )

                scheduler_payload: Dict[str, dict] = {}
                if not args.skip_scheduler:
                    token_templates = build_token_templates(
                        stream=stream,
                        miss_flags=replay_summary.miss_flags,
                        cold_miss_flags=replay_summary.cold_miss_flags,
                        prefetch_hit_flags=replay_summary.prefetch_hit_flags,
                    )
                    for system_batch in scheduler_batch_grid:
                        scheduler = simulate_scheduler_sensitivity(
                            condition=condition,
                            cache_budget=int(cache_budget),
                            request_ids=stream.request_ids,
                            token_templates_by_req=token_templates,
                            request_arrival_times_ms=request_arrival_times_ms,
                            calibration=calibration,
                            object_sizes=object_sizes,
                            transfer_mode=args.transfer_mode,
                            load_profile=args.load_profile,
                            stat_key=args.calibration_stat,
                            max_system_batch=int(system_batch),
                            miss_handling_mode=str(args.miss_handling_mode),
                            prefetch_cpu_discount=float(args.temporal_prefetch_cpu_discount),
                            overlap_policy=effective_overlap_policy,
                        )
                        scheduler_rows.append(scheduler.summary_row)
                        scheduler_payload[str(system_batch)] = dict(scheduler.summary_row)

                budget_payloads[str(cache_budget)] = {
                    "cache_budget": int(cache_budget),
                    "hits": int(replay_summary.hits),
                    "misses": int(replay_summary.misses),
                    "cold_misses": int(replay_summary.cold_misses),
                    "capacity_misses": int(replay_summary.capacity_misses),
                    "stage1_token_tpot_summary": dict(stage1.quantiles),
                    "stage1_tail_threshold_ms": float(stage1.tail_threshold_ms),
                    "decode_token_count": int(len(stage1.token_rows)),
                    "background_policy": {
                        "overlap_policy": effective_overlap_policy,
                        "promotion_admitted": int(replay_summary.promotion_admitted),
                        "promotion_hits": int(replay_summary.promotion_hits),
                        "prefetch_predictions": int(replay_summary.prefetch_predictions),
                        "prefetch_matches": int(replay_summary.prefetch_matches),
                        "prefetch_false_positives": int(replay_summary.prefetch_false_positives),
                        "prefetch_miss_hits": int(replay_summary.prefetch_miss_hits),
                    },
                    "scheduler_sensitivity": scheduler_payload,
                }

            per_condition_payloads[condition] = {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "object_key_definition": OBJECT_KEY_DEFINITIONS[condition],
                "object_key_note": OBJECT_KEY_NOTES[condition],
                "total_events": int(access_buffer.total_events),
                "max_object_id": int(access_buffer.max_object_id),
                "budgets": budget_payloads,
            }
            stream.condition_buffers[condition].access_ids = np.empty(0, dtype=np.uint32)
            gc.collect()
    finally:
        token_handle.close()
    finalize_condition_merged_csv(token_temp_path, token_output_path)

    quantile_rows.sort(key=lambda row: (CONDITION_ORDER.index(str(row["condition"])), int(row["cache_budget"])))
    tail_rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(str(row["condition"])),
            int(row["cache_budget"]),
            int(row["req_idx"]),
            int(row["token_ordinal"]),
        )
    )
    layer_rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(str(row["condition"])),
            int(row["cache_budget"]),
            int(row["layer_id"]),
        )
    )
    request_rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(str(row["condition"])),
            int(row["cache_budget"]),
            int(row["req_idx"]),
        )
    )
    scheduler_rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(str(row["condition"])),
            int(row["cache_budget"]),
            int(row["system_batch"]),
        )
    )
    background_policy_rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(str(row["condition"])),
            int(row["cache_budget"]),
        )
    )

    write_condition_merged_csv(quantiles_output_path, TPOT_QUANTILE_FIELDS, selected_conditions, preserve_existing, quantile_rows)
    write_condition_merged_csv(tail_output_path, TAIL_TOKEN_FIELDS, selected_conditions, preserve_existing, tail_rows)
    write_condition_merged_csv(layer_output_path, LAYER_BARRIER_FIELDS, selected_conditions, preserve_existing, layer_rows)
    write_condition_merged_csv(request_output_path, REQUEST_DECODE_FIELDS, selected_conditions, preserve_existing, request_rows)
    write_condition_merged_csv(scheduler_output_path, SCHEDULER_FIELDS, selected_conditions, preserve_existing, scheduler_rows)
    write_condition_merged_csv(background_output_path, BACKGROUND_POLICY_FIELDS, selected_conditions, preserve_existing, background_policy_rows)

    manifest = {
        "model_name": "case_study_system_tpot_v1",
        "goal": "Convert B8 miss bitmaps into calibrated token-level TPOT under a causal per-layer system baseline.",
        "inputs": {
            "joined_indep_path": str(joined_indep_path),
            "joined_corr_path": str(joined_corr_path),
            "qc_report_path": str(qc_report_path),
            "cache_curve_path": str(cache_curve_path),
            "per_request_miss_path": str(per_request_miss_path),
            "calibration_path": str(calibration_path),
            "model_dir": str(model_dir),
        },
        "assumptions": {
            "causal_packing": "per-layer only; no token-wide future-aware packing",
            "gate_lora_excluded": bool(object_sizes.gate_lora_excluded),
            "stage1_system_batch": int(args.stage1_system_batch),
            "scheduler_batch_grid": [int(value) for value in scheduler_batch_grid],
            "transfer_mode": str(args.transfer_mode),
            "load_profile": str(args.load_profile),
            "calibration_stat": str(args.calibration_stat),
            "tail_quantile": float(args.tail_quantile),
            "miss_handling_mode": str(args.miss_handling_mode),
            "overlap_policy": effective_overlap_policy,
            "deferred_promotion_delta_steps": int(args.deferred_promotion_delta_steps),
            "temporal_prefetch_enabled": bool(args.temporal_prefetch),
            "temporal_prefetch_cpu_discount": float(args.temporal_prefetch_cpu_discount),
        },
        "object_sizes": {
            "early_component_label": object_sizes.early_component_label,
            "late_component_label": object_sizes.late_component_label,
            "early_object_bytes": int(object_sizes.early_object_bytes),
            "late_object_bytes": int(object_sizes.late_object_bytes),
            "total_object_bytes": int(object_sizes.total_object_bytes),
            "early_activation_bytes": int(object_sizes.early_activation_bytes),
            "early_result_bytes": int(object_sizes.early_result_bytes),
            "late_activation_bytes": int(object_sizes.late_activation_bytes),
            "late_result_bytes": int(object_sizes.late_result_bytes),
            "slice_bytes": int(object_sizes.slice_bytes),
            "gate_lora_excluded": bool(object_sizes.gate_lora_excluded),
        },
        "request_count": int(len(stream.request_ids)),
        "total_events": int(total_events),
        "request_stream_alignment": {
            "checked_fields": list(stream.invariant_checked_fields),
            "position_field_counts": dict(stream.position_field_counts),
        },
        "condition_labels": dict(CONDITION_LABELS),
        "conditions": per_condition_payloads,
        "seeds": dict(seeds),
        "outputs": {
            "token_tpot_csv": str(token_output_path),
            "tpot_quantiles_csv": str(quantiles_output_path),
            "tail_token_breakdown_csv": str(tail_output_path),
            "layer_barrier_breakdown_csv": str(layer_output_path),
            "request_decode_summary_csv": str(request_output_path),
            "scheduler_sensitivity_csv": str(scheduler_output_path),
            "background_policy_summary_csv": str(background_output_path),
            "manifest_json": str(manifest_output_path),
        },
    }
    write_json(manifest_output_path, manifest)
    print(
        "wrote calibrated TPOT artifacts: "
        f"{token_output_path}, "
        f"{quantiles_output_path}, "
        f"{tail_output_path}, "
        f"{layer_output_path}, "
            f"{request_output_path}, "
            f"{scheduler_output_path}, "
            f"{background_output_path}, "
            f"{manifest_output_path}"
    )


if __name__ == "__main__":
    main()
