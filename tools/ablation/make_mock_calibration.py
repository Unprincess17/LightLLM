#!/usr/bin/env python3
"""Generate a schema-compatible mock calibration manifest for ablation smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


THIS_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = THIS_DIR.parent / "case_study"

import sys

if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(CASE_STUDY_DIR) not in sys.path:
    sys.path.append(str(CASE_STUDY_DIR))

from ablation_utils import resolve_config_and_run_id
from common import artifact_root, ensure_parent_dir, write_json  # type: ignore
from policy_models import resolve_object_sizes
from system_tpot_core import load_model_text_config, parse_batch_grid, parse_batch_value_map  # type: ignore


DEFAULT_BATCH_GRID = "1,2,4"
DEFAULT_BASE_TPOT = "1:1.20,2:1.45,4:1.90"
DEFAULT_ROW_GRID = "1,2,4,8,16,32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mock calibration manifest for ablation smoke tests")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Run id used to resolve the default output path")
    parser.add_argument("--output_path", type=str, default=None, help="Explicit output path")
    parser.add_argument("--model_dir", type=str, default=None, help="Override HF model directory")
    parser.add_argument("--base_tpot_ms", type=str, default=DEFAULT_BASE_TPOT, help="Batch:value map for hot-cache TPOT")
    parser.add_argument("--batch_grid", type=str, default=DEFAULT_BATCH_GRID, help="Comma-separated batch grid")
    parser.add_argument("--row_grid", type=str, default=DEFAULT_ROW_GRID, help="Comma-separated row-count grid")
    parser.add_argument("--dtype", type=str, default="bf16", help="LoRA dtype")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--slice_bytes", type=int, default=256 * 1024, help="Slice size for fragmented transfers")
    return parser.parse_args()


def metric_payload(value_ms: float) -> dict:
    base = float(value_ms)
    return {
        "mean_ms": base,
        "p50_ms": base,
        "p90_ms": base * 1.12,
        "p95_ms": base * 1.18,
        "max_ms": base * 1.25,
    }


def main() -> None:
    args = parse_args()
    config, run_id = resolve_config_and_run_id(args.config, args.run_id)
    model_dir = Path(args.model_dir) if args.model_dir else Path(config.get("paths", {}).get("model_dir"))
    text_config = load_model_text_config(model_dir)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else (artifact_root(config) / "ablation" / run_id / "mock" / "manifests" / "mock_calibration.json")
    )
    if not output_path.is_absolute():
        output_path = (THIS_DIR.parent.parent / output_path).resolve()

    batch_grid = parse_batch_grid(args.batch_grid)
    base_tpot_ms = parse_batch_value_map(args.base_tpot_ms)
    row_grid = [int(token.strip()) for token in str(args.row_grid).split(",") if token.strip()]
    object_sizes = resolve_object_sizes(
        calibration={},
        model_dir=model_dir,
        dtype_name=str(args.dtype),
        lora_rank=int(args.lora_rank),
        slice_bytes=int(args.slice_bytes),
    )

    payload_byte_points = sorted(
        {
            int(row_count) * int(object_sizes.early_object_bytes)
            for row_count in row_grid
        }
        | {
            int(row_count) * int(object_sizes.late_object_bytes)
            for row_count in row_grid
        }
        | {
            int(row_count) * int(object_sizes.total_object_bytes)
            for row_count in row_grid
        }
    )
    slice_points = sorted(
        {max(1, int((payload_bytes + object_sizes.slice_bytes - 1) // object_sizes.slice_bytes)) for payload_bytes in payload_byte_points}
    )

    transfer_modes: Dict[str, dict] = {}
    for mode_name in (
        "staged_pageable_packed",
        "direct_pinned_packed",
        "direct_pageable_fragmented",
    ):
        profiles: Dict[str, dict] = {}
        for profile_name, factor in (("idle", 1.0), ("stressed", 1.25)):
            if mode_name == "staged_pageable_packed":
                profiles[profile_name] = {
                    "packed_gather_rows": [
                        {"payload_bytes": int(payload_bytes), **metric_payload(0.020 * factor + payload_bytes / 7.0e7 * factor)}
                        for payload_bytes in payload_byte_points
                    ],
                    "packed_h2d_rows": [
                        {"payload_bytes": int(payload_bytes), **metric_payload(0.018 * factor + payload_bytes / 1.4e8 * factor)}
                        for payload_bytes in payload_byte_points
                    ],
                    "packed_launch_rows": [
                        {"payload_bytes": int(payload_bytes), **metric_payload(0.010 * factor + payload_bytes / 8.0e8 * factor)}
                        for payload_bytes in payload_byte_points
                    ],
                }
            elif mode_name == "direct_pinned_packed":
                profiles[profile_name] = {
                    "packed_h2d_rows": [
                        {"payload_bytes": int(payload_bytes), **metric_payload(0.014 * factor + payload_bytes / 1.8e8 * factor)}
                        for payload_bytes in payload_byte_points
                    ],
                    "packed_launch_rows": [
                        {"payload_bytes": int(payload_bytes), **metric_payload(0.008 * factor + payload_bytes / 8.5e8 * factor)}
                        for payload_bytes in payload_byte_points
                    ],
                }
            else:
                profiles[profile_name] = {
                    "fragmented_total_rows": [
                        {"slice_count": int(slice_count), **metric_payload(0.060 * factor + 0.028 * factor * slice_count)}
                        for slice_count in slice_points
                    ]
                }
        transfer_modes[mode_name] = {"profiles": profiles}

    cold_path_curves: Dict[str, dict] = {"profiles": {}}
    for profile_name, factor in (("idle", 1.0), ("stressed", 1.25)):
        cold_path_curves["profiles"][profile_name] = {
            "early": {
                "pack_rows": [
                    {"row_count": int(row_count), **metric_payload(0.010 * factor + 0.0045 * row_count * factor)}
                    for row_count in row_grid
                ],
                "d2h_rows": [
                    {
                        "payload_bytes": int(row_count) * int(object_sizes.early_activation_bytes),
                        **metric_payload(0.022 * factor + 0.009 * row_count * factor),
                    }
                    for row_count in row_grid
                ],
                "cpu_rows": [
                    {"row_count": int(row_count), **metric_payload(0.090 * factor + 0.088 * row_count * factor)}
                    for row_count in row_grid
                ],
                "h2d_rows": [
                    {
                        "payload_bytes": int(row_count) * int(object_sizes.early_result_bytes),
                        **metric_payload(0.018 * factor + 0.008 * row_count * factor),
                    }
                    for row_count in row_grid
                ],
                "merge_rows": [
                    {"row_count": int(row_count), **metric_payload(0.006 * factor + 0.003 * row_count * factor)}
                    for row_count in row_grid
                ],
            },
            "late": {
                "pack_rows": [
                    {"row_count": int(row_count), **metric_payload(0.012 * factor + 0.004 * row_count * factor)}
                    for row_count in row_grid
                ],
                "d2h_rows": [
                    {
                        "payload_bytes": int(row_count) * int(object_sizes.late_activation_bytes),
                        **metric_payload(0.020 * factor + 0.008 * row_count * factor),
                    }
                    for row_count in row_grid
                ],
                "cpu_rows": [
                    {"row_count": int(row_count), **metric_payload(0.082 * factor + 0.080 * row_count * factor)}
                    for row_count in row_grid
                ],
                "h2d_rows": [
                    {
                        "payload_bytes": int(row_count) * int(object_sizes.late_result_bytes),
                        **metric_payload(0.016 * factor + 0.0075 * row_count * factor),
                    }
                    for row_count in row_grid
                ],
                "merge_rows": [
                    {"row_count": int(row_count), **metric_payload(0.006 * factor + 0.0025 * row_count * factor)}
                    for row_count in row_grid
                ],
            },
        }

    overlap_windows_ms = {
        str(batch_size): {
            "early": metric_payload(0.120 + 0.050 * batch_size),
            "late": metric_payload(0.100 + 0.045 * batch_size),
        }
        for batch_size in batch_grid
    }
    lora_compute_ms = {
        str(batch_size): {
            "rows": max(1, int(batch_size) * 8),
            "early": metric_payload(0.080 + 0.020 * batch_size),
            "late": metric_payload(0.075 + 0.018 * batch_size),
            "total": metric_payload(0.160 + 0.040 * batch_size),
        }
        for batch_size in batch_grid
    }

    payload = {
        "model_name": "ablation_mock_calibration_v1",
        "device": {"requested_device": "mock", "resolved_device": "mock", "name": "mock", "capability": [0, 0]},
        "measurement": {"warmup_iters": 0, "measure_iters": 0, "host_profiles": ["idle", "stressed"]},
        "object_sizes": {
            "hidden_size": int(text_config["hidden_size"]),
            "moe_intermediate_size": int(text_config["moe_intermediate_size"]),
            "dtype": str(args.dtype),
            "lora_rank": int(args.lora_rank),
            "slice_bytes": int(args.slice_bytes),
            "early_component_label": object_sizes.early_component_label,
            "late_component_label": object_sizes.late_component_label,
            "early_object_bytes": int(object_sizes.early_object_bytes),
            "late_object_bytes": int(object_sizes.late_object_bytes),
            "total_object_bytes": int(object_sizes.total_object_bytes),
            "early_object_slices": int(object_sizes.early_object_slices),
            "late_object_slices": int(object_sizes.late_object_slices),
            "total_object_slices": int(object_sizes.total_object_slices),
            "gate_lora_excluded": bool(object_sizes.gate_lora_excluded),
            "early_activation_bytes": int(object_sizes.early_activation_bytes),
            "early_result_bytes": int(object_sizes.early_result_bytes),
            "late_activation_bytes": int(object_sizes.late_activation_bytes),
            "late_result_bytes": int(object_sizes.late_result_bytes),
        },
        "assumptions": {
            "base_tpot_source": "mock",
            "cold_path_payload_model": "pack + pinned_d2h + cpu_lora + pinned_h2d + merge",
            "note": "Synthetic calibration for ablation debugging and smoke tests.",
        },
        "cold_path_config": {
            "row_grid": [int(value) for value in row_grid],
            "cpu_threads": 1,
            "cpu_kernel_mode_requested": "mock",
            "cpu_kernel_mode_effective": "mock",
        },
        "base_tpot_ms": {str(key): float(value) for key, value in base_tpot_ms.items()},
        "transfer_modes": transfer_modes,
        "cold_path_curves": cold_path_curves,
        "lora_compute_ms": lora_compute_ms,
        "overlap_windows_ms": overlap_windows_ms,
    }

    ensure_parent_dir(output_path)
    write_json(output_path, payload)
    print(f"Wrote mock calibration to {output_path}")


if __name__ == "__main__":
    main()
