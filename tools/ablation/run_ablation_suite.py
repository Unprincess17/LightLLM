#!/usr/bin/env python3
"""Run the offline COLoRA ablation suite."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = THIS_DIR.parent / "case_study"

import sys

if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(CASE_STUDY_DIR) not in sys.path:
    sys.path.append(str(CASE_STUDY_DIR))

from ablation_utils import (
    DEFAULT_EVENT_WINDOW,
    DEFAULT_REPEAT_WINDOWS,
    DEFAULT_TIMELINE_BUDGET,
    build_repeat_windows,
    detect_request_count,
    ensure_inputs_exist,
    load_json,
    materialize_repeat_window,
    quantile_summary,
    resolve_ablation_output_paths,
    resolve_case_study_inputs,
    resolve_config_and_run_id,
    resolve_total_events_from_qc,
    write_manifest,
)
from common import parse_cardinalities, write_csv  # type: ignore
from export_timeline import build_timeline_rows_from_step_rows
from policy_models import (
    build_cpu_utilization_windows,
    build_decode_event_windows,
    build_variant_replay_summary,
    resolve_object_sizes,
    select_tail_request,
    simulate_variant_stream,
    validate_calibration_for_variant,
)
from system_tpot_core import build_system_streams, TRANSFER_MODE_STAGED_PAGEABLE_PACKED  # type: ignore
from variants import VARIANTS, parse_variant_csv, resolve_variants


TOKEN_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "req_idx",
    "token_ordinal",
    "token_pos",
    "layer_count",
    "base_tpot_ms",
    "tpot_ms",
    "exposed_miss_latency_ms",
    "miss_objects",
    "cold_miss_objects",
    "capacity_miss_objects",
    "prefetched_miss_objects",
    "gpu_wait_ms",
    "cpu_compute_ms",
    "d2h_h2d_ms",
    "merge_ms",
    "overlap_hidden_ms",
    "d2h_bytes",
    "h2d_bytes",
    "weight_h2d_bytes",
    "cpu_busy_ms",
    "total_transfer_bytes",
    "total_slices",
    "decode_event_count",
]
REQUEST_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "req_idx",
    "decode_tokens",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "max_tpot_ms",
    "mean_gpu_wait_ms",
    "mean_cpu_compute_ms",
    "mean_d2h_h2d_ms",
    "mean_merge_ms",
    "mean_overlap_hidden_ms",
]
QUANTILE_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "mean",
    "p50",
    "p90",
    "p95",
    "p99",
    "p999",
    "max",
    "tail_threshold_ms",
    "token_count",
]
BREAKDOWN_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "gpu_wait_ms",
    "cpu_compute_ms",
    "d2h_h2d_ms",
    "merge_ms",
    "overlap_hidden_ms",
]
TRANSFER_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "d2h_bytes",
    "h2d_bytes",
    "weight_h2d_bytes",
    "effective_bw_gbps",
]
CPU_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "window_id",
    "decode_events",
    "cpu_busy_ratio",
]
HIT_WINDOW_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "window_id",
    "decode_events",
    "hit_rate",
    "cold_miss_rate",
]
POLICY_FIELDS = [
    "run_id",
    "suite_id",
    "variant",
    "variant_label",
    "condition",
    "repeat_id",
    "cache_budget",
    "promotion_admitted",
    "promotion_hits",
    "prefetch_predictions",
    "prefetch_matches",
    "prefetch_false_positives",
    "prefetch_miss_hits",
]
TIMELINE_FIELDS = [
    "variant",
    "cache_budget",
    "repeat_id",
    "req_idx",
    "token_ordinal",
    "token_pos",
    "layer_id",
    "state",
    "resource",
    "start_ms",
    "end_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COLoRA offline ablation suite")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Run id")
    parser.add_argument("--suite_id", type=str, default="core", help="Ablation suite: core/background/granularity/all")
    parser.add_argument("--variants", type=str, default=None, help="Optional comma-separated variant subset")
    parser.add_argument("--output_root", type=str, default=None, help="Optional output root override")
    parser.add_argument("--trace_source", type=str, default="real", choices=["real", "synthetic"], help="Trace source label")
    parser.add_argument("--joined_indep_path", type=str, default=None, help="Override joined_trace_indep.jsonl path")
    parser.add_argument("--joined_corr_path", type=str, default=None, help="Override joined_trace_corr.jsonl path")
    parser.add_argument("--qc_report_path", type=str, default=None, help="Override join_qc_report.json path")
    parser.add_argument("--calibration_path", type=str, default=None, help="Calibration JSON path")
    parser.add_argument("--model_dir", type=str, default=None, help="Override HF model directory")
    parser.add_argument("--cache_budgets", type=str, default=None, help="Comma-separated cache budgets")
    parser.add_argument("--repeat_windows", type=int, default=DEFAULT_REPEAT_WINDOWS, help="How many deterministic request windows to replay")
    parser.add_argument("--window_decode_events", type=int, default=DEFAULT_EVENT_WINDOW, help="Decode-event window for hit-rate and CPU utilization")
    parser.add_argument("--timeline_budget", type=int, default=DEFAULT_TIMELINE_BUDGET, help="Budget used for the timeline export")
    parser.add_argument("--timeline_variants", type=str, default="colora_full,no_overlap", help="Comma-separated variant subset for timeline export")
    parser.add_argument("--transfer_mode", type=str, default=TRANSFER_MODE_STAGED_PAGEABLE_PACKED, help="Transfer mode from the calibration manifest")
    parser.add_argument("--load_profile", type=str, default="stressed", help="Calibration load profile")
    parser.add_argument("--calibration_stat", type=str, default="p50_ms", choices=["mean_ms", "p50_ms", "p90_ms"], help="Statistic to read from the calibration curves")
    parser.add_argument("--system_batch", type=int, default=1, help="Effective decode batch used for token TPOT")
    parser.add_argument("--tail_quantile", type=float, default=0.95, help="Tail quantile used for request selection")
    parser.add_argument("--dtype", type=str, default="bf16", help="LoRA dtype")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--slice_bytes", type=int, default=256 * 1024, help="Slice size in bytes")
    parser.add_argument("--progress_every", type=int, default=1_000_000, help="Replay materialization progress interval")
    return parser.parse_args()


def add_metadata(row: Mapping[str, object], **metadata: object) -> dict:
    payload = dict(metadata)
    payload.update(dict(row))
    return payload


def main() -> None:
    args = parse_args()
    config, run_id = resolve_config_and_run_id(args.config, args.run_id)
    inputs = resolve_case_study_inputs(
        config=config,
        run_id=run_id,
        joined_indep_path=args.joined_indep_path,
        joined_corr_path=args.joined_corr_path,
        qc_report_path=args.qc_report_path,
        calibration_path=args.calibration_path,
    )
    ensure_inputs_exist(
        [
            Path(inputs["joined_indep_path"]),
            Path(inputs["joined_corr_path"]),
            Path(inputs["calibration_path"]),
        ]
    )
    if Path(inputs["qc_report_path"]).exists():
        ensure_inputs_exist([Path(inputs["qc_report_path"])])

    output_paths = resolve_ablation_output_paths(
        config=config,
        run_id=run_id,
        suite_id=str(args.suite_id),
        output_root=args.output_root,
    )
    calibration = load_json(Path(inputs["calibration_path"]))
    model_dir = Path(args.model_dir) if args.model_dir else Path(config.get("paths", {}).get("model_dir"))
    object_sizes = resolve_object_sizes(
        calibration=calibration,
        model_dir=model_dir,
        dtype_name=str(args.dtype),
        lora_rank=int(args.lora_rank),
        slice_bytes=int(args.slice_bytes),
    )

    variant_specs = resolve_variants(str(args.suite_id), parse_variant_csv(args.variants))
    for variant in variant_specs:
        validate_calibration_for_variant(calibration, variant, str(args.load_profile))

    if args.cache_budgets:
        cache_budgets = parse_cardinalities(args.cache_budgets)
    else:
        cache_budgets = parse_cardinalities(config.get("case_study", {}).get("replay_cache", {}).get("budget_grid"))
    if not cache_budgets:
        raise ValueError("cache budget grid must not be empty")

    request_count = detect_request_count(Path(inputs["joined_corr_path"]))
    repeat_windows = build_repeat_windows(request_count, int(args.repeat_windows))
    full_total_events = (
        resolve_total_events_from_qc(Path(inputs["qc_report_path"]))
        if Path(inputs["qc_report_path"]).exists()
        else sum(1 for _ in open(Path(inputs["joined_corr_path"]), "r", encoding="utf-8"))
    )

    token_rows: List[dict] = []
    request_rows: List[dict] = []
    quantile_rows: List[dict] = []
    breakdown_rows: List[dict] = []
    transfer_rows: List[dict] = []
    cpu_rows: List[dict] = []
    hit_window_rows: List[dict] = []
    policy_rows: List[dict] = []

    timeline_variant_ids = [token.strip() for token in str(args.timeline_variants).split(",") if token.strip()]
    stored_timeline_steps: Dict[str, List[dict]] = {}
    stored_timeline_requests: List[dict] = []
    selected_repeat_id = repeat_windows[0].repeat_id if repeat_windows else "r00"

    with tempfile.TemporaryDirectory(prefix="ablation_suite_", dir="/tmp") as temp_root:
        temp_root_path = Path(temp_root)
        for window in repeat_windows:
            if window.request_start == 0 and window.request_end == request_count:
                materialized = {
                    "repeat_id": window.repeat_id,
                    "joined_indep_path": Path(inputs["joined_indep_path"]),
                    "joined_corr_path": Path(inputs["joined_corr_path"]),
                    "total_events": full_total_events,
                    "request_start": window.request_start,
                    "request_end": window.request_end,
                    "request_count": window.request_count,
                }
            else:
                materialized_window = materialize_repeat_window(
                    joined_indep_path=Path(inputs["joined_indep_path"]),
                    joined_corr_path=Path(inputs["joined_corr_path"]),
                    window=window,
                    temp_dir=temp_root_path / window.repeat_id,
                )
                materialized = materialized_window.__dict__

            stream = build_system_streams(
                joined_indep_path=Path(materialized["joined_indep_path"]),
                joined_corr_path=Path(materialized["joined_corr_path"]),
                total_events=int(materialized["total_events"]),
                progress_every=int(args.progress_every),
            )

            for cache_budget in cache_budgets:
                for variant in variant_specs:
                    replay_summary = build_variant_replay_summary(
                        variant=variant,
                        stream=stream,
                        cache_budget=int(cache_budget),
                    )
                    result = simulate_variant_stream(
                        variant=variant,
                        cache_budget=int(cache_budget),
                        stream=stream,
                        replay_summary=replay_summary,
                        calibration=calibration,
                        object_sizes=object_sizes,
                        transfer_mode=str(args.transfer_mode),
                        load_profile=str(args.load_profile),
                        stat_key=str(args.calibration_stat),
                        system_batch=int(args.system_batch),
                        tail_quantile=float(args.tail_quantile),
                    )

                    metadata = {
                        "run_id": run_id,
                        "suite_id": str(args.suite_id),
                        "variant": variant.variant_id,
                        "variant_label": variant.label,
                        "condition": variant.trace_condition,
                        "repeat_id": str(materialized["repeat_id"]),
                        "cache_budget": int(cache_budget),
                    }
                    token_rows.extend(add_metadata(row, **metadata) for row in result.token_rows)
                    request_rows.extend(add_metadata(row, **metadata) for row in result.request_rows)
                    quantile_rows.append(
                        {
                            **metadata,
                            **result.quantiles,
                            "tail_threshold_ms": float(result.tail_threshold_ms),
                            "token_count": int(len(result.token_rows)),
                        }
                    )

                    if result.token_rows:
                        breakdown_rows.append(
                            {
                                **metadata,
                                "gpu_wait_ms": float(sum(float(row["gpu_wait_ms"]) for row in result.token_rows) / len(result.token_rows)),
                                "cpu_compute_ms": float(sum(float(row["cpu_compute_ms"]) for row in result.token_rows) / len(result.token_rows)),
                                "d2h_h2d_ms": float(sum(float(row["d2h_h2d_ms"]) for row in result.token_rows) / len(result.token_rows)),
                                "merge_ms": float(sum(float(row["merge_ms"]) for row in result.token_rows) / len(result.token_rows)),
                                "overlap_hidden_ms": float(sum(float(row["overlap_hidden_ms"]) for row in result.token_rows) / len(result.token_rows)),
                            }
                        )
                    else:
                        breakdown_rows.append(
                            {
                                **metadata,
                                "gpu_wait_ms": 0.0,
                                "cpu_compute_ms": 0.0,
                                "d2h_h2d_ms": 0.0,
                                "merge_ms": 0.0,
                                "overlap_hidden_ms": 0.0,
                            }
                        )

                    total_d2h_bytes = sum(int(row["d2h_bytes"]) for row in result.step_rows)
                    total_h2d_bytes = sum(int(row["h2d_bytes"]) for row in result.step_rows)
                    total_weight_h2d_bytes = sum(int(row["weight_h2d_bytes"]) for row in result.step_rows)
                    total_transfer_ms = sum(
                        float(row["d2h_h2d_ms"]) + float(row["weight_h2d_ms"]) for row in result.step_rows
                    )
                    effective_bw_gbps = (
                        float((total_d2h_bytes + total_h2d_bytes + total_weight_h2d_bytes) * 8.0 / total_transfer_ms / 1.0e6)
                        if total_transfer_ms > 0.0
                        else 0.0
                    )
                    transfer_rows.append(
                        {
                            **metadata,
                            "d2h_bytes": int(total_d2h_bytes),
                            "h2d_bytes": int(total_h2d_bytes),
                            "weight_h2d_bytes": int(total_weight_h2d_bytes),
                            "effective_bw_gbps": float(effective_bw_gbps),
                        }
                    )

                    hit_windows = build_decode_event_windows(stream, replay_summary, int(args.window_decode_events))
                    hit_window_rows.extend(add_metadata(row, **metadata) for row in hit_windows)
                    cpu_windows = build_cpu_utilization_windows(result.token_rows, int(args.window_decode_events))
                    cpu_rows.extend(add_metadata(row, **metadata) for row in cpu_windows)
                    policy_rows.append(
                        {
                            **metadata,
                            "promotion_admitted": int(replay_summary.promotion_admitted),
                            "promotion_hits": int(replay_summary.promotion_hits),
                            "prefetch_predictions": int(replay_summary.prefetch_predictions),
                            "prefetch_matches": int(replay_summary.prefetch_matches),
                            "prefetch_false_positives": int(replay_summary.prefetch_false_positives),
                            "prefetch_miss_hits": int(replay_summary.prefetch_miss_hits),
                        }
                    )

                    if (
                        int(cache_budget) == int(args.timeline_budget)
                        and str(materialized["repeat_id"]) == selected_repeat_id
                        and variant.variant_id in timeline_variant_ids
                    ):
                        stored_timeline_steps[variant.variant_id] = [add_metadata(row, **metadata) for row in result.step_rows]
                        if variant.variant_id == timeline_variant_ids[0]:
                            stored_timeline_requests = [add_metadata(row, **metadata) for row in result.request_rows]

    timeline_rows: List[dict] = []
    timeline_request_id = select_tail_request(stored_timeline_requests)
    if timeline_request_id is not None:
        for variant_id in timeline_variant_ids:
            step_rows = stored_timeline_steps.get(variant_id, [])
            filtered = [row for row in step_rows if int(row["req_idx"]) == int(timeline_request_id)]
            timeline_rows.extend(
                build_timeline_rows_from_step_rows(
                    step_rows=filtered,
                    variant_id=variant_id,
                    cache_budget=int(args.timeline_budget),
                    repeat_id=selected_repeat_id,
                    req_idx=int(timeline_request_id),
                )
            )

    write_csv(output_paths.metrics_dir / "ablation_token_tpot.csv", TOKEN_FIELDS, token_rows)
    write_csv(output_paths.metrics_dir / "ablation_request_summary.csv", REQUEST_FIELDS, request_rows)
    write_csv(output_paths.metrics_dir / "ablation_quantiles.csv", QUANTILE_FIELDS, quantile_rows)
    write_csv(output_paths.metrics_dir / "ablation_breakdown.csv", BREAKDOWN_FIELDS, breakdown_rows)
    write_csv(output_paths.metrics_dir / "ablation_transfer_usage.csv", TRANSFER_FIELDS, transfer_rows)
    write_csv(output_paths.metrics_dir / "ablation_cpu_utilization.csv", CPU_FIELDS, cpu_rows)
    write_csv(output_paths.metrics_dir / "ablation_hit_rate_over_time.csv", HIT_WINDOW_FIELDS, hit_window_rows)
    write_csv(output_paths.metrics_dir / "ablation_policy_summary.csv", POLICY_FIELDS, policy_rows)
    write_csv(output_paths.timelines_dir / "ablation_timeline.csv", TIMELINE_FIELDS, timeline_rows)

    manifest = {
        "run_id": run_id,
        "suite_id": str(args.suite_id),
        "trace_source": str(args.trace_source),
        "variant_ids": [variant.variant_id for variant in variant_specs],
        "variant_labels": {variant.variant_id: variant.label for variant in variant_specs},
        "cache_budgets": [int(value) for value in cache_budgets],
        "repeat_windows": [
            {
                "repeat_id": window.repeat_id,
                "request_start": int(window.request_start),
                "request_end": int(window.request_end),
                "request_count": int(window.request_count),
            }
            for window in repeat_windows
        ],
        "timeline": {
            "budget": int(args.timeline_budget),
            "variants": list(timeline_variant_ids),
            "repeat_id": selected_repeat_id,
            "req_idx": int(timeline_request_id) if timeline_request_id is not None else None,
        },
        "inputs": {
            "joined_indep_path": str(inputs["joined_indep_path"]),
            "joined_corr_path": str(inputs["joined_corr_path"]),
            "qc_report_path": str(inputs["qc_report_path"]),
            "calibration_path": str(inputs["calibration_path"]),
        },
        "outputs": {
            "token_csv": str(output_paths.metrics_dir / "ablation_token_tpot.csv"),
            "request_csv": str(output_paths.metrics_dir / "ablation_request_summary.csv"),
            "quantiles_csv": str(output_paths.metrics_dir / "ablation_quantiles.csv"),
            "breakdown_csv": str(output_paths.metrics_dir / "ablation_breakdown.csv"),
            "transfer_csv": str(output_paths.metrics_dir / "ablation_transfer_usage.csv"),
            "cpu_util_csv": str(output_paths.metrics_dir / "ablation_cpu_utilization.csv"),
            "hit_rate_csv": str(output_paths.metrics_dir / "ablation_hit_rate_over_time.csv"),
            "policy_csv": str(output_paths.metrics_dir / "ablation_policy_summary.csv"),
            "timeline_csv": str(output_paths.timelines_dir / "ablation_timeline.csv"),
        },
    }
    write_manifest(output_paths.manifests_dir / "ablation_manifest.json", manifest)
    print(f"Wrote ablation outputs to {output_paths.root_dir}")


if __name__ == "__main__":
    main()
