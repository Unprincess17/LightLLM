#!/usr/bin/env python3
"""Convert ablation timeline CSVs into paper-facing figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from variants import VARIANTS


RESOURCE_ORDER = ("gpu", "cpu", "pcie")
STATE_COLORS = {
    "gpu_window_early": "#4C956C",
    "gpu_window_late": "#8AC926",
    "cold_fetch_early": "#277DA1",
    "cold_fetch_late": "#4D908E",
    "cold_cpu_early": "#F8961E",
    "cold_cpu_late": "#F9C74F",
    "reinsert_early": "#577590",
    "reinsert_late": "#90BE6D",
    "sync_promote": "#C8553D",
    "blocking_promotion": "#B56576",
    "ready": "#D9D9D9",
    "done": "#6D6875",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an ablation timeline figure from ablation_timeline.csv")
    parser.add_argument("--timeline_csv", type=str, required=True, help="Timeline CSV emitted by run_ablation_suite.py")
    parser.add_argument("--output_path", type=str, required=True, help="Output PDF/PNG path")
    parser.add_argument("--variants", type=str, default=None, help="Optional comma-separated variant subset")
    parser.add_argument("--request_id", type=int, default=None, help="Optional req_idx filter")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    return parser.parse_args()


def read_timeline_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_timeline_rows(
    rows: Sequence[Mapping[str, str]],
    variants: Optional[Sequence[str]] = None,
    request_id: Optional[int] = None,
) -> List[dict]:
    allowed_variants = {str(variant) for variant in variants} if variants else None
    filtered: List[dict] = []
    for row in rows:
        if allowed_variants is not None and str(row["variant"]) not in allowed_variants:
            continue
        if request_id is not None and int(row["req_idx"]) != int(request_id):
            continue
        filtered.append(dict(row))
    return filtered


def build_timeline_rows_from_step_rows(
    step_rows: Sequence[Mapping[str, object]],
    variant_id: str,
    cache_budget: int,
    repeat_id: str,
    req_idx: int,
) -> List[dict]:
    rows: List[dict] = []
    cursor_ms = 0.0
    ordered_steps = sorted(
        step_rows,
        key=lambda row: (
            int(row["token_ordinal"]),
            int(row["layer_id"]),
            int(row["event_idx"]),
        ),
    )
    for row in ordered_steps:
        token_ordinal = int(row["token_ordinal"])
        token_pos = int(row["token_pos"])
        layer_id = int(row["layer_id"])
        base_step_slot_ms = float(row["base_step_slot_ms"])
        rows.append(
            {
                "variant": variant_id,
                "cache_budget": int(cache_budget),
                "repeat_id": repeat_id,
                "req_idx": int(req_idx),
                "token_ordinal": token_ordinal,
                "token_pos": token_pos,
                "layer_id": layer_id,
                "state": "ready",
                "resource": "gpu",
                "start_ms": float(cursor_ms),
                "end_ms": float(cursor_ms),
            }
        )

        service_model = str(row["service_model"])
        step_start_ms = float(cursor_ms)
        step_end_ms = float(step_start_ms)

        if service_model == "blocking_promotion_first":
            blocking_ms = float(row["weight_h2d_ms"])
            if blocking_ms > 0.0:
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "blocking_promotion",
                        "resource": "pcie",
                        "start_ms": step_start_ms,
                        "end_ms": step_start_ms + blocking_ms,
                    }
                )
                step_end_ms = max(step_end_ms, step_start_ms + blocking_ms)
        else:
            early_window_ms = float(row["early_overlap_window_ms"])
            late_window_ms = float(row["late_overlap_window_ms"])
            if early_window_ms > 0.0:
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "gpu_window_early",
                        "resource": "gpu",
                        "start_ms": step_start_ms,
                        "end_ms": step_start_ms + early_window_ms,
                    }
                )
                step_end_ms = max(step_end_ms, step_start_ms + early_window_ms)
            if late_window_ms > 0.0:
                late_anchor = step_start_ms + max(base_step_slot_ms * 0.50, early_window_ms * 0.50)
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "gpu_window_late",
                        "resource": "gpu",
                        "start_ms": late_anchor,
                        "end_ms": late_anchor + late_window_ms,
                    }
                )
                step_end_ms = max(step_end_ms, late_anchor + late_window_ms)

            early_fetch_ms = float(row["early_pack_ms"]) + float(row["early_d2h_ms"])
            early_cpu_ms = float(row["early_cpu_ms"])
            early_reinsert_ms = float(row["early_h2d_ms"]) + float(row["early_merge_ms"])
            if early_fetch_ms > 0.0:
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "cold_fetch_early",
                        "resource": "pcie",
                        "start_ms": step_start_ms,
                        "end_ms": step_start_ms + early_fetch_ms,
                    }
                )
                step_end_ms = max(step_end_ms, step_start_ms + early_fetch_ms)
            if early_cpu_ms > 0.0:
                early_cpu_start = step_start_ms + early_fetch_ms
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "cold_cpu_early",
                        "resource": "cpu",
                        "start_ms": early_cpu_start,
                        "end_ms": early_cpu_start + early_cpu_ms,
                    }
                )
                step_end_ms = max(step_end_ms, early_cpu_start + early_cpu_ms)
            if early_reinsert_ms > 0.0:
                early_reinsert_start = step_start_ms + early_fetch_ms + early_cpu_ms
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "reinsert_early",
                        "resource": "gpu",
                        "start_ms": early_reinsert_start,
                        "end_ms": early_reinsert_start + early_reinsert_ms,
                    }
                )
                step_end_ms = max(step_end_ms, early_reinsert_start + early_reinsert_ms)

            late_phase_start = step_start_ms + max(base_step_slot_ms * 0.50, early_window_ms * 0.50)
            late_fetch_ms = float(row["late_pack_ms"]) + float(row["late_d2h_ms"])
            late_cpu_ms = float(row["late_cpu_ms"])
            late_reinsert_ms = float(row["late_h2d_ms"]) + float(row["late_merge_ms"])
            if late_fetch_ms > 0.0:
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "cold_fetch_late",
                        "resource": "pcie",
                        "start_ms": late_phase_start,
                        "end_ms": late_phase_start + late_fetch_ms,
                    }
                )
                step_end_ms = max(step_end_ms, late_phase_start + late_fetch_ms)
            if late_cpu_ms > 0.0:
                late_cpu_start = late_phase_start + late_fetch_ms
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "cold_cpu_late",
                        "resource": "cpu",
                        "start_ms": late_cpu_start,
                        "end_ms": late_cpu_start + late_cpu_ms,
                    }
                )
                step_end_ms = max(step_end_ms, late_cpu_start + late_cpu_ms)
            if late_reinsert_ms > 0.0:
                late_reinsert_start = late_phase_start + late_fetch_ms + late_cpu_ms
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "reinsert_late",
                        "resource": "gpu",
                        "start_ms": late_reinsert_start,
                        "end_ms": late_reinsert_start + late_reinsert_ms,
                    }
                )
                step_end_ms = max(step_end_ms, late_reinsert_start + late_reinsert_ms)

            sync_promote_ms = float(row["weight_h2d_ms"])
            if sync_promote_ms > 0.0:
                sync_start = max(step_end_ms, step_start_ms + base_step_slot_ms)
                rows.append(
                    {
                        "variant": variant_id,
                        "cache_budget": int(cache_budget),
                        "repeat_id": repeat_id,
                        "req_idx": int(req_idx),
                        "token_ordinal": token_ordinal,
                        "token_pos": token_pos,
                        "layer_id": layer_id,
                        "state": "sync_promote",
                        "resource": "pcie",
                        "start_ms": sync_start,
                        "end_ms": sync_start + sync_promote_ms,
                    }
                )
                step_end_ms = max(step_end_ms, sync_start + sync_promote_ms)

        step_duration_ms = max(step_end_ms - step_start_ms, base_step_slot_ms + float(row["exposed_stall_ms"]), 1e-6)
        cursor_ms = step_start_ms + step_duration_ms
        rows.append(
            {
                "variant": variant_id,
                "cache_budget": int(cache_budget),
                "repeat_id": repeat_id,
                "req_idx": int(req_idx),
                "token_ordinal": token_ordinal,
                "token_pos": token_pos,
                "layer_id": layer_id,
                "state": "done",
                "resource": "gpu",
                "start_ms": float(cursor_ms),
                "end_ms": float(cursor_ms),
            }
        )
    return rows


def plot_timeline(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    if not rows:
        raise ValueError("timeline figure requires at least one row")
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["variant"]),
            RESOURCE_ORDER.index(str(row["resource"])) if str(row["resource"]) in RESOURCE_ORDER else len(RESOURCE_ORDER),
            float(row["start_ms"]),
            str(row["state"]),
        ),
    )
    lane_keys = []
    for row in ordered_rows:
        lane_key = (str(row["variant"]), str(row["resource"]))
        if lane_key not in lane_keys:
            lane_keys.append(lane_key)
    lane_y = {lane_key: float(index) for index, lane_key in enumerate(lane_keys)}

    fig_height = max(2.8, 0.65 * len(lane_keys) + 1.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))

    legend_handles = {}
    for row in ordered_rows:
        start_ms = float(row["start_ms"])
        end_ms = float(row["end_ms"])
        width_ms = max(end_ms - start_ms, 1e-6)
        lane_key = (str(row["variant"]), str(row["resource"]))
        state = str(row["state"])
        color = STATE_COLORS.get(state, "#999999")
        label = state.replace("_", " ")
        bar = ax.barh(
            y=lane_y[lane_key],
            width=width_ms,
            left=start_ms,
            height=0.65,
            color=color,
            edgecolor="#1F2933",
            linewidth=0.4,
            label=label,
        )
        legend_handles.setdefault(label, bar[0])

    lane_labels = []
    for variant_id, resource in lane_keys:
        variant_label = VARIANTS.get(variant_id).label if variant_id in VARIANTS else variant_id
        lane_labels.append(f"{variant_label} | {resource.upper()}")

    ax.set_yticks([lane_y[key] for key in lane_keys], labels=lane_labels)
    ax.set_xlabel("Approx. time (ms)")
    ax.set_ylabel("Variant / Resource")
    ax.set_title(title or "Ablation Timeline")
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.invert_yaxis()
    ax.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D7DCE0",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_timeline_rows(Path(args.timeline_csv))
    selected = filter_timeline_rows(
        rows,
        variants=[token.strip() for token in str(args.variants).split(",") if token.strip()] if args.variants else None,
        request_id=args.request_id,
    )
    plot_timeline(selected, Path(args.output_path), title=args.title)
    print(f"Wrote timeline figure to {args.output_path}")


if __name__ == "__main__":
    main()
