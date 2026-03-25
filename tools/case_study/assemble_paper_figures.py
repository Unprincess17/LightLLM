#!/usr/bin/env python3
"""Assemble B13 paper-facing figures, manifest, and storyline from B7-B10 artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (
        REPO_ROOT,
        artifact_root,
        ensure_parent_dir,
        load_global_config,
        resolve_run_id,
        stage_output_dir,
    )
else:
    from .common import (
        REPO_ROOT,
        artifact_root,
        ensure_parent_dir,
        load_global_config,
        resolve_run_id,
        stage_output_dir,
    )


PLOT_SCRIPT_PATH = Path("tools/case_study/assemble_paper_figures.py")
DEFAULT_STORYLINE_PATH = Path("notes/storyline_case_study.md")
FIGURE_STAGE = "figures"

CLAIM_FIG1 = "joint access is more fragmented than expert-only access"
CLAIM_FIG2 = "joint objects have worse temporal locality"
CLAIM_FIG3 = "joint modeling amplifies misses under the same budget"
CLAIM_FIG4 = "tail penalty rises quickly from the single-LoRA regime and then plateaus"
CLAIM_FIG5 = "tail requests touch more cold joint objects than average requests"

CACHE_CONDITION_ORDER = ("expert_only", "joint_indep", "joint_corr")
JOINT_CACHE_CONDITIONS = ("joint_indep", "joint_corr")

CONDITION_LABELS = {
    "expert_only": "B0 Expert-only",
    "joint_indep": "B1 Joint-indep",
    "joint_corr": "B2 Joint-corr",
    "B0": "B0 Expert-only",
    "B1": "B1 Joint-indep",
    "B2": "B2 Joint-corr",
}

CONDITION_SHORT_LABELS = {
    "expert_only": "B0",
    "joint_indep": "B1",
    "joint_corr": "B2",
    "B0": "B0",
    "B1": "B1",
    "B2": "B2",
}

CONDITION_COLORS = {
    "expert_only": "#355070",
    "joint_indep": "#C8553D",
    "joint_corr": "#2A9D8F",
    "B0": "#355070",
    "B1": "#C8553D",
    "B2": "#2A9D8F",
}

TAIL_CONDITION_PAIRS = {
    "joint_indep": "B1",
    "joint_corr": "B2",
}

REUSE_QUANTILES = (0.90, 0.95, 0.99)
REUSE_QUANTILE_LABELS = ("P90", "P95", "P99")
FIG4_PREFERRED_BUDGETS = (8192, 16384, 32768, 65536, 4096, 1024, 256)
FIG4_PLATEAU_START_LORAS = 4
FIGURE_SAVE_METADATA = {"Creator": str(PLOT_SCRIPT_PATH)}
PAPER_TWO_UP_FIGSIZE = (5.6, 3.6)
PAPER_TWO_UP_MULTI_PANEL_FIGSIZE = (6.1, 3.6)
PAPER_TWO_UP_COMPACT_FIGSIZE = (5.3, 3.5)


@dataclass(frozen=True)
class FigureRecord:
    output_path: Path
    input_paths: Sequence[Path]
    plotting_script: Path
    claim: str
    caption: str
    figure_slice: str


@dataclass(frozen=True)
class B13Paths:
    run_id: str
    run_root: Path
    sweeps_root: Path
    figures_dir: Path
    storyline_path: Path
    popularity_path: Path
    reuse_path: Path
    cache_curve_path: Path
    per_request_miss_path: Path
    tail_breakdown_path: Path
    num_loras_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble B13 paper-facing figures and story docs")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=None,
        help="Override output directory for figures and manifest",
    )
    parser.add_argument(
        "--sweeps_dir",
        type=str,
        default=None,
        help="Override directory containing B10 sweep artifacts (default: artifacts/case_study/<run_id>/sweeps)",
    )
    parser.add_argument(
        "--storyline_path",
        type=str,
        default=None,
        help="Override output path for notes/storyline_case_study.md",
    )
    return parser.parse_args()


def apply_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing required B13 inputs:\n{formatted}")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def condition_rows(rows: Sequence[Mapping[str, str]], condition: str) -> List[Mapping[str, str]]:
    return [row for row in rows if str(row["condition"]) == condition]


def single_row(rows: Sequence[Mapping[str, str]], **filters: str) -> Mapping[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in filters.items()):
            return row
    raise KeyError(f"row not found for filters={filters}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_parent_dir(path)
    fig.savefig(path, bbox_inches="tight", metadata=FIGURE_SAVE_METADATA)
    plt.close(fig)


def format_pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def format_ms(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f} ms"


def popularity_rank_coverage(rows: Sequence[Mapping[str, str]], condition: str, rank: int) -> float:
    subset = condition_rows(rows, condition)
    for row in subset:
        if int(row["rank"]) == rank:
            return float(row["cumulative_fraction"])
    if not subset:
        raise ValueError(f"no popularity rows for condition={condition}")
    max_row = max(subset, key=lambda row: int(row["rank"]))
    return float(max_row["cumulative_fraction"])


def weighted_quantiles(rows: Sequence[Mapping[str, str]], quantiles: Sequence[float]) -> Dict[float, int]:
    finite_rows = sorted(
        (
            (int(row["reuse_distance"]), int(row["count"]))
            for row in rows
            if int(row["reuse_distance"]) >= 0
        ),
        key=lambda item: item[0],
    )
    if not finite_rows:
        raise ValueError("reuse distance projection is empty after excluding cold touches")

    total = sum(count for _, count in finite_rows)
    results: Dict[float, int] = {}
    cumulative = 0
    next_index = 0
    sorted_quantiles = sorted(float(value) for value in quantiles)
    for quantile in sorted_quantiles:
        threshold = quantile * total
        while next_index < len(finite_rows):
            distance, count = finite_rows[next_index]
            cumulative += count
            next_index += 1
            if cumulative >= threshold:
                results[quantile] = distance
                break
        if quantile not in results:
            results[quantile] = finite_rows[-1][0]
    return results


def cold_touch_fraction(rows: Sequence[Mapping[str, str]]) -> float:
    total = sum(int(row["count"]) for row in rows)
    cold = sum(int(row["count"]) for row in rows if int(row["reuse_distance"]) < 0)
    return float(cold) / float(total) if total else 0.0


def mean_from_rows(rows: Sequence[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values:
        raise ValueError(f"cannot compute mean for empty row set: {field}")
    return float(sum(values) / len(values))


def select_fig4_budget(rows: Sequence[Mapping[str, str]]) -> int:
    budgets = sorted({int(row["cache_budget"]) for row in rows})
    if not budgets:
        raise ValueError("fig4 input CSV is empty")
    for candidate in FIG4_PREFERRED_BUDGETS:
        if candidate in budgets:
            return candidate
    return budgets[-1]


def select_fig4_plateau_start(num_loras: Sequence[int]) -> int:
    if not num_loras:
        raise ValueError("fig4 requires at least one num_loras point")
    if FIG4_PLATEAU_START_LORAS in num_loras:
        return FIG4_PLATEAU_START_LORAS
    higher = [value for value in num_loras if value > FIG4_PLATEAU_START_LORAS]
    if higher:
        return min(higher)
    if len(num_loras) >= 2:
        return num_loras[1]
    return num_loras[0]


def resolve_b13_paths(
    config_path: str | None = None,
    run_id: str | None = None,
    figures_dir: str | None = None,
    sweeps_dir: str | None = None,
    storyline_path: str | None = None,
) -> B13Paths:
    config = load_global_config(config_path)
    resolved_run_id = resolve_run_id(config, run_id)
    run_root = artifact_root(config) / "case_study" / resolved_run_id
    default_sweeps_root = run_root / "sweeps"
    legacy_sweeps_root = artifact_root(config) / "sweeps"
    if sweeps_dir:
        resolved_sweeps_root = Path(sweeps_dir)
    elif default_sweeps_root.exists() or not legacy_sweeps_root.exists():
        resolved_sweeps_root = default_sweeps_root
    else:
        resolved_sweeps_root = legacy_sweeps_root
    resolved_figures_dir = stage_output_dir(FIGURE_STAGE, config, resolved_run_id, figures_dir)
    resolved_storyline_path = Path(storyline_path) if storyline_path else REPO_ROOT / DEFAULT_STORYLINE_PATH

    return B13Paths(
        run_id=resolved_run_id,
        run_root=run_root,
        sweeps_root=resolved_sweeps_root,
        figures_dir=resolved_figures_dir,
        storyline_path=resolved_storyline_path,
        popularity_path=run_root / "replay" / "locality" / "popularity_rank.csv",
        reuse_path=run_root / "replay" / "locality" / "reuse_distance_cdf.csv",
        cache_curve_path=run_root / "replay" / "cache" / "cache_curve.csv",
        per_request_miss_path=run_root / "replay" / "cache" / "per_request_miss_count.csv",
        tail_breakdown_path=run_root / "replay" / "latency" / "tail_request_breakdown.csv",
        num_loras_path=resolved_sweeps_root / "num_loras_vs_p99.csv",
    )


def plot_fig1_popularity_rank(popularity_path: Path, output_path: Path) -> tuple[plt.Figure, FigureRecord]:
    rows = read_csv_rows(popularity_path)
    fig, ax = plt.subplots(figsize=PAPER_TWO_UP_FIGSIZE, constrained_layout=True)

    topk_lines = []
    for condition in CACHE_CONDITION_ORDER:
        subset = condition_rows(rows, condition)
        ranks = [int(row["rank"]) for row in subset]
        coverage = [float(row["cumulative_fraction"]) for row in subset]
        ax.plot(
            ranks,
            coverage,
            color=CONDITION_COLORS[condition],
            linewidth=2.2,
            label=CONDITION_LABELS[condition],
        )
        topk = popularity_rank_coverage(rows, condition, rank=1000)
        topk_lines.append(f"{CONDITION_SHORT_LABELS[condition]} top-1000: {format_pct(topk)}")

    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Object popularity rank")
    ax.set_ylabel("Cumulative access share")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Access Coverage by Popularity Rank")
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.03,
        0.72,
        "\n".join(topk_lines),
        transform=ax.transAxes,
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.3"},
    )

    coverage_b0 = popularity_rank_coverage(rows, "expert_only", rank=1000)
    coverage_b1 = popularity_rank_coverage(rows, "joint_indep", rank=1000)
    coverage_b2 = popularity_rank_coverage(rows, "joint_corr", rank=1000)
    caption = (
        "Cumulative access share versus object popularity rank. The expert-only stream concentrates "
        f"{format_pct(coverage_b0)} of accesses in the top 1000 objects, while the joint stream falls to "
        f"{format_pct(coverage_b1)} in B1 and {format_pct(coverage_b2)} in B2, indicating a much flatter and "
        "more fragmented popularity profile once adapters enter the access key."
    )
    return fig, FigureRecord(
        output_path=output_path,
        input_paths=[popularity_path],
        plotting_script=PLOT_SCRIPT_PATH,
        claim=CLAIM_FIG1,
        caption=caption,
        figure_slice="Full popularity-rank coverage curve with the top-1000 coverage highlighted from the same B7 CSV.",
    )


def build_fig1_popularity_rank(popularity_path: Path, output_path: Path) -> FigureRecord:
    fig, record = plot_fig1_popularity_rank(popularity_path, output_path)
    save_figure(fig, output_path)
    return record


def plot_fig2_reuse_distance(reuse_path: Path, output_path: Path) -> tuple[plt.Figure, FigureRecord]:
    rows = read_csv_rows(reuse_path)
    quantile_map = {
        condition: weighted_quantiles(condition_rows(rows, condition), REUSE_QUANTILES)
        for condition in CACHE_CONDITION_ORDER
    }
    cold_per_10k = {
        condition: cold_touch_fraction(condition_rows(rows, condition)) * 10000.0
        for condition in CACHE_CONDITION_ORDER
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=PAPER_TWO_UP_MULTI_PANEL_FIGSIZE,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3.15, 1.35]},
    )
    ax_left, ax_right = axes

    x = np.arange(len(REUSE_QUANTILES))
    width = 0.22
    for index, condition in enumerate(CACHE_CONDITION_ORDER):
        offsets = x + (index - 1) * width
        values = [quantile_map[condition][quantile] for quantile in REUSE_QUANTILES]
        ax_left.bar(
            offsets,
            values,
            width=width,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
    ax_left.set_xticks(x, REUSE_QUANTILE_LABELS)
    ax_left.set_ylabel("Finite reuse distance (events)")
    ax_left.set_title("Finite Reuse-Distance Quantiles")
    ax_left.legend(loc="upper left", frameon=False)

    cold_x = np.arange(len(CACHE_CONDITION_ORDER))
    cold_values = [cold_per_10k[condition] for condition in CACHE_CONDITION_ORDER]
    ax_right.bar(
        cold_x,
        cold_values,
        color=[CONDITION_COLORS[condition] for condition in CACHE_CONDITION_ORDER],
    )
    ax_right.set_xticks(cold_x, [CONDITION_SHORT_LABELS[condition] for condition in CACHE_CONDITION_ORDER])
    ax_right.set_ylabel("Cold first touches per 10k events")
    ax_right.set_title("Cold-Start Rate")

    q99_b0 = quantile_map["expert_only"][0.99]
    q99_b1 = quantile_map["joint_indep"][0.99]
    q99_b2 = quantile_map["joint_corr"][0.99]
    caption = (
        "Joint keys worsen temporal locality in both tail reuse distance and cold-start rate. The finite P99 reuse "
        f"distance rises from {q99_b0:,} events in B0 to {q99_b1:,} in B1 and {q99_b2:,} in B2, while cold first "
        f"touches rise from {cold_per_10k['expert_only']:.1f} to {cold_per_10k['joint_indep']:.1f} and "
        f"{cold_per_10k['joint_corr']:.1f} per 10k events."
    )
    return fig, FigureRecord(
        output_path=output_path,
        input_paths=[reuse_path],
        plotting_script=PLOT_SCRIPT_PATH,
        claim=CLAIM_FIG2,
        caption=caption,
        figure_slice="B7 reuse-distance projection summarized as finite P90/P95/P99 bars plus cold first touches per 10k events.",
    )


def build_fig2_reuse_distance(reuse_path: Path, output_path: Path) -> FigureRecord:
    fig, record = plot_fig2_reuse_distance(reuse_path, output_path)
    save_figure(fig, output_path)
    return record


def plot_fig3_cache_curve(cache_curve_path: Path, output_path: Path) -> tuple[plt.Figure, FigureRecord]:
    rows = read_csv_rows(cache_curve_path)
    fig, ax = plt.subplots(figsize=PAPER_TWO_UP_FIGSIZE, constrained_layout=True)

    for condition in CACHE_CONDITION_ORDER:
        subset = sorted(
            (
                row
                for row in condition_rows(rows, condition)
                if int(row["cache_budget"]) > 0
            ),
            key=lambda row: int(row["cache_budget"]),
        )
        budgets = [int(row["cache_budget"]) for row in subset]
        miss_rates = [float(row["miss_rate"]) for row in subset]
        ax.plot(
            budgets,
            miss_rates,
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Cache budget (objects)")
    ax.set_ylabel("Miss rate")
    ax.set_title("Miss-Rate Curve Under a Shared Cache Budget")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100.0 * value:g}%"))
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D7DCE0",
    )

    row_b0_2048 = single_row(rows, condition="expert_only", cache_budget="2048")
    row_b1_2048 = single_row(rows, condition="joint_indep", cache_budget="2048")
    row_b2_2048 = single_row(rows, condition="joint_corr", cache_budget="2048")
    ratio_b1_2048 = float(row_b1_2048["miss_rate"]) / float(row_b0_2048["miss_rate"])
    ratio_b2_2048 = float(row_b2_2048["miss_rate"]) / float(row_b0_2048["miss_rate"])
    caption = (
        "Cache replay under a shared LRU budget shows that joint modeling amplifies misses after the expert-only "
        f"condition reaches its locality scale. At 2048 objects, miss rate is {float(row_b0_2048['miss_rate']):.5f} "
        f"in B0 versus {float(row_b1_2048['miss_rate']):.5f} in B1 and {float(row_b2_2048['miss_rate']):.5f} in B2 "
        f"({ratio_b1_2048:.1f}x and {ratio_b2_2048:.1f}x worse than B0)."
    )
    return fig, FigureRecord(
        output_path=output_path,
        input_paths=[cache_curve_path],
        plotting_script=PLOT_SCRIPT_PATH,
        claim=CLAIM_FIG3,
        caption=caption,
        figure_slice="Positive-budget slice of the B8 cache curve on log-log axes, with the 2048-object comparison called out.",
    )


def build_fig3_cache_curve(cache_curve_path: Path, output_path: Path) -> FigureRecord:
    fig, record = plot_fig3_cache_curve(cache_curve_path, output_path)
    save_figure(fig, output_path)
    return record


def plot_fig4_num_loras_vs_p99(num_loras_path: Path, output_path: Path) -> tuple[plt.Figure, FigureRecord]:
    rows = read_csv_rows(num_loras_path)
    selected_budget = select_fig4_budget(rows)
    subset = [row for row in rows if int(row["cache_budget"]) == selected_budget]
    baseline = float(single_row(subset, condition="expert_only", cache_budget=str(selected_budget))["p99"])
    num_loras = sorted({int(row["num_loras"]) for row in subset})
    plateau_start = select_fig4_plateau_start(num_loras)
    plateau_indices = [index for index, value in enumerate(num_loras) if value >= plateau_start]
    plateau_loras = [num_loras[index] for index in plateau_indices]
    has_single_lora = 1 in num_loras
    onset_available = has_single_lora and len(plateau_loras) > 0 and plateau_start > 1

    gaps_by_condition: Dict[str, List[float]] = {}
    for condition in JOINT_CACHE_CONDITIONS:
        gaps_by_condition[condition] = [
            float(single_row(subset, condition=condition, num_loras=str(num), cache_budget=str(selected_budget))["p99"])
            - baseline
            for num in num_loras
        ]

    mean_gap = [
        (gaps_by_condition["joint_indep"][index] + gaps_by_condition["joint_corr"][index]) / 2.0
        for index in range(len(num_loras))
    ]
    plateau_mean_joint_gap = float(sum(mean_gap[index] for index in plateau_indices) / len(plateau_indices))
    plateau_mean_by_condition = {
        condition: float(sum(gaps_by_condition[condition][index] for index in plateau_indices) / len(plateau_indices))
        for condition in JOINT_CACHE_CONDITIONS
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axvspan(plateau_start, max(num_loras), color="#F5EBDD", alpha=0.55, zorder=0)
    ax.axvline(plateau_start, color="#8D3B2C", linewidth=1.2, linestyle=":")
    ax.plot(
        num_loras,
        gaps_by_condition["joint_indep"],
        color=CONDITION_COLORS["joint_indep"],
        linewidth=2.0,
        marker="s",
        markersize=5.2,
        label=CONDITION_LABELS["joint_indep"],
        zorder=3,
    )
    ax.plot(
        num_loras,
        gaps_by_condition["joint_corr"],
        color=CONDITION_COLORS["joint_corr"],
        linewidth=2.0,
        marker="^",
        markersize=5.8,
        label=CONDITION_LABELS["joint_corr"],
        zorder=3,
    )
    ax.plot(
        num_loras,
        mean_gap,
        color="#8D3B2C",
        linewidth=2.4,
        marker="o",
        markersize=4.8,
        label="Mean joint gap",
    )
    ax.axhline(plateau_mean_joint_gap, color="#8D3B2C", linewidth=1.2, linestyle="--", alpha=0.9)
    ax.axhline(0.0, color="#7A7F85", linewidth=1.0, linestyle="--")
    if max(num_loras) / max(min(num_loras), 1) >= 8:
        ax.set_xscale("log", base=2)
    ax.set_xticks(num_loras, [str(value) for value in num_loras])
    ax.set_xlabel("Number of modeled LoRAs")
    ax.set_ylabel("P99 latency gap vs. B0 (ms)")
    ax.set_title(f"Tail Penalty Onset and Plateau ({selected_budget} objects)")
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D7DCE0",
    )
    plateau_text = (
        f"Plateau zone: >= {plateau_start} LoRAs\n"
        f"Mean plateau gap: {format_ms(plateau_mean_joint_gap)}"
    )
    if onset_available:
        single_index = num_loras.index(1)
        onset_gain = mean_gap[plateau_indices[0]] - mean_gap[single_index]
        plateau_text += f"\n1 -> {plateau_start} LoRAs: +{format_ms(onset_gain)}"
    ax.text(
        0.98,
        0.04,
        plateau_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.3"},
    )

    if onset_available:
        single_index = num_loras.index(1)
        plateau_entry_index = plateau_indices[0]
        caption = (
            f"At the {selected_budget}-object slice, the joint P99 penalty rises from "
            f"{format_ms(mean_gap[single_index])} at 1 LoRA to {format_ms(mean_gap[plateau_entry_index])} at "
            f"{plateau_start} LoRAs, then stays on a broad plateau around {format_ms(plateau_mean_joint_gap)} for "
            f">= {plateau_start} LoRAs. B1 and B2 plateau near {format_ms(plateau_mean_by_condition['joint_indep'])} "
            f"and {format_ms(plateau_mean_by_condition['joint_corr'])}, respectively."
        )
        figure_slice = (
            f"B10 num_loras sweep at cache_budget={selected_budget}, emphasizing the 1-to-{plateau_start} LoRA onset "
            "and the plateau beyond it."
        )
    else:
        start_gap = mean_gap[0]
        end_gap = mean_gap[-1]
        caption = (
            f"At the {selected_budget}-object slice, the available LoRA-count points already sit on the plateau regime: "
            f"the mean joint P99 gap varies from {format_ms(start_gap)} at {num_loras[0]} LoRAs to {format_ms(end_gap)} "
            f"at {num_loras[-1]} LoRAs around a plateau level of {format_ms(plateau_mean_joint_gap)}."
        )
        figure_slice = (
            f"B10 num_loras sweep at cache_budget={selected_budget}; available points begin in the plateau regime, "
            "so the figure emphasizes its flat high tail-penalty band."
        )
    return fig, FigureRecord(
        output_path=output_path,
        input_paths=[num_loras_path],
        plotting_script=PLOT_SCRIPT_PATH,
        claim=CLAIM_FIG4,
        caption=caption,
        figure_slice=figure_slice,
    )


def build_fig4_num_loras_vs_p99(num_loras_path: Path, output_path: Path) -> FigureRecord:
    fig, record = plot_fig4_num_loras_vs_p99(num_loras_path, output_path)
    save_figure(fig, output_path)
    return record


def plot_fig5_tail_breakdown(
    cache_curve_path: Path,
    per_request_miss_path: Path,
    tail_breakdown_path: Path,
    output_path: Path,
) -> tuple[plt.Figure, FigureRecord]:
    cache_rows = read_csv_rows(cache_curve_path)
    per_request_rows = read_csv_rows(per_request_miss_path)
    tail_rows = read_csv_rows(tail_breakdown_path)

    selected_budget = 65536
    for cache_condition in JOINT_CACHE_CONDITIONS:
        budget_row = single_row(cache_rows, condition=cache_condition, cache_budget=str(selected_budget))
        miss_rate = float(budget_row["miss_rate"])
        cold_rate = float(budget_row["cold_miss_rate"])
        capacity_rate = float(budget_row["capacity_miss_rate"])
        if abs(miss_rate - cold_rate) > 1e-12 or abs(capacity_rate) > 1e-12:
            raise ValueError(
                f"fig5 expects the {selected_budget}-object slice to be the cold-miss floor for {cache_condition}, "
                f"but saw miss_rate={miss_rate}, cold_miss_rate={cold_rate}, capacity_miss_rate={capacity_rate}"
            )

    average_request_misses = []
    average_tail_cold_misses = []
    ratio_strings = []
    x_labels = []
    tail_bar_colors = []
    for cache_condition in JOINT_CACHE_CONDITIONS:
        latency_condition = TAIL_CONDITION_PAIRS[cache_condition]
        all_rows = [
            row
            for row in per_request_rows
            if row["condition"] == cache_condition and int(row["cache_budget"]) == selected_budget
        ]
        tail_subset = [
            row
            for row in tail_rows
            if row["condition"] == latency_condition and int(row["cache_budget"]) == selected_budget
        ]
        average_all = mean_from_rows(all_rows, "miss_count")
        average_tail = mean_from_rows(tail_subset, "cold_miss_count")
        ratio_strings.append(f"{average_tail / average_all:.1f}x")
        average_request_misses.append(average_all)
        average_tail_cold_misses.append(average_tail)
        x_labels.append(CONDITION_SHORT_LABELS[latency_condition])
        tail_bar_colors.append(CONDITION_COLORS[latency_condition])

    fig, ax = plt.subplots(figsize=PAPER_TWO_UP_COMPACT_FIGSIZE, constrained_layout=True)
    x = np.arange(len(JOINT_CACHE_CONDITIONS))
    width = 0.32
    ax.bar(
        x - width / 2.0,
        average_request_misses,
        width=width,
        color="#C9CED6",
        label="Average request",
    )
    ax.bar(
        x + width / 2.0,
        average_tail_cold_misses,
        width=width,
        color=tail_bar_colors,
        label="P95+ tail request",
    )
    for index, ratio in enumerate(ratio_strings):
        ax.text(
            x[index] + width / 2.0,
            average_tail_cold_misses[index] + max(average_tail_cold_misses) * 0.025,
            ratio,
            ha="center",
            va="bottom",
            fontsize=13,
        )
    ax.set_xticks(x, x_labels)
    ax.set_ylabel("Cold joint-object touches per request")
    ax.set_title(f"Tail Requests Touch More Cold Objects ({selected_budget} objects)")
    ax.legend(loc="upper left", frameon=False)

    caption = (
        f"At the {selected_budget}-object cold-miss floor, an average request touches "
        f"{average_request_misses[0]:.1f} cold joint objects in B1 and {average_request_misses[1]:.1f} in B2, "
        f"but a P95+ tail request touches {average_tail_cold_misses[0]:.1f} and {average_tail_cold_misses[1]:.1f}, "
        f"roughly {ratio_strings[0]} and {ratio_strings[1]} more than average."
    )
    return fig, FigureRecord(
        output_path=output_path,
        input_paths=[cache_curve_path, per_request_miss_path, tail_breakdown_path],
        plotting_script=PLOT_SCRIPT_PATH,
        claim=CLAIM_FIG5,
        caption=caption,
        figure_slice="B8/B9 request-level slice at cache_budget=65536, where miss_count equals cold-object touches for joint conditions.",
    )


def build_fig5_tail_breakdown(
    cache_curve_path: Path,
    per_request_miss_path: Path,
    tail_breakdown_path: Path,
    output_path: Path,
) -> FigureRecord:
    fig, record = plot_fig5_tail_breakdown(
        cache_curve_path,
        per_request_miss_path,
        tail_breakdown_path,
        output_path,
    )
    save_figure(fig, output_path)
    return record


def write_manifest(path: Path, run_id: str, figure_records: Sequence[FigureRecord]) -> None:
    lines = [
        "# Figure Manifest",
        "",
        f"Autogenerated by `{PLOT_SCRIPT_PATH}` for run `{run_id}`.",
        "",
    ]
    for record in figure_records:
        lines.extend(
            [
                f"## {record.output_path.name}",
                "",
                f"- Output file: `{relative_path(record.output_path)}`",
                "- Input files used:",
            ]
        )
        lines.extend(f"  - `{relative_path(input_path)}`" for input_path in record.input_paths)
        lines.extend(
            [
                f"- Plotting script used: `{record.plotting_script}`",
                f"- Exact claim supported: {record.claim}",
                f"- Figure slice: {record.figure_slice}",
                f"- Draft caption: {record.caption}",
                "",
            ]
        )
    ensure_parent_dir(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_storyline(path: Path, run_id: str, figure_records: Sequence[FigureRecord]) -> None:
    fig1 = next(record for record in figure_records if record.output_path.name == "fig1_popularity_rank.pdf")
    fig2 = next(record for record in figure_records if record.output_path.name == "fig2_reuse_distance.pdf")
    fig3 = next(record for record in figure_records if record.output_path.name == "fig3_cache_curve.pdf")
    fig4 = next(record for record in figure_records if record.output_path.name == "fig4_num_loras_vs_p99.pdf")
    fig5 = next(record for record in figure_records if record.output_path.name == "fig5_tail_breakdown.pdf")

    lines = [
        "# Case Study Storyline",
        "",
        f"Autogenerated by `{PLOT_SCRIPT_PATH}` for run `{run_id}`.",
        "",
        "## Setup",
        "",
        "We replay one aligned trace-driven request stream under three object definitions: B0 uses expert-only objects "
        "`(layer_id, expert_id)`, B1 uses joint objects `(layer_id, expert_id, adapter_id)` under the independent "
        "mapping, and B2 uses the same joint key under the correlated mapping. Figures 1-3 are direct measurements "
        "from the real joined trace in B7-B9, while Figure 4 uses a richer B10 synthetic num_loras sweep to show "
        "how the tail turns on once the system leaves the single-LoRA regime and then saturates.",
        "",
        "## Core Findings",
        "",
        "The storyline should move from access structure to system impact. First show that adding adapters to the "
        "access key flattens popularity and worsens reuse locality; then show that the same locality loss drives much "
        "higher miss rates under a shared cache budget; then show that a small multi-LoRA regime is enough to trigger "
        "most of the eventual tail penalty; then close the loop at the request level by showing that those tail "
        "requests are exactly the ones that touch many more cold joint objects than an average request.",
        "",
        "## Figure Order",
        "",
        "1. Introduce `Fig. 1` as the access-distribution setup figure. Lead with the fact that the top of the "
        "expert-only working set absorbs most accesses, while the joint working set spreads those accesses over far "
        f"more objects. Use the caption directly: {fig1.caption}",
        "2. Follow with `Fig. 2` to convert fragmentation into a temporal-locality claim. The paper text should say "
        "that the median request structure is unchanged, but the joint stream has a much heavier far tail of reuse "
        f"distance and a higher cold-start rate. Use the caption directly: {fig2.caption}",
        "3. Use `Fig. 3` as the bridge from locality to cache behavior. Introduce it with the phrase \"under the same "
        "budget, the joint key expansion amplifies misses once the expert-only cache is large enough to exploit its "
        f"locality.\" Use the caption directly: {fig3.caption}",
        "4. Bring in `Fig. 4` after the main trace-driven result to show that the tail penalty turns on quickly once "
        "the system leaves the single-LoRA regime and then remains on a broad plateau. Frame it as a controlled "
        "perturbation rather than the primary evidence. The paper text should emphasize the threshold-plus-plateau "
        "shape instead of claiming a strong monotonic rise across the whole x-axis. Use the caption directly: "
        f"{fig4.caption}",
        "5. Close with `Fig. 5` as the request-level mechanism figure: the tail is cold-object-heavy, not just "
        "slightly slower on average. That gives the paper a concrete explanation for why the mean can nearly "
        f"converge while the P99 stays elevated. Use the caption directly: {fig5.caption}",
        "",
        "The concise paper claim sequence is: fragmented joint popularity -> worse temporal locality -> higher miss "
        "rates under the same budget -> a small multi-LoRA regime already triggers most of the eventual tail penalty "
        "-> tail requests dominated by cold joint-object touches.",
        "",
    ]
    ensure_parent_dir(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def assemble_paper_figures(
    config_path: str | None = None,
    run_id: str | None = None,
    figures_dir: str | None = None,
    sweeps_dir: str | None = None,
    storyline_path: str | None = None,
) -> tuple[list[FigureRecord], Path, Path]:
    apply_matplotlib_style()
    paths = resolve_b13_paths(
        config_path=config_path,
        run_id=run_id,
        figures_dir=figures_dir,
        sweeps_dir=sweeps_dir,
        storyline_path=storyline_path,
    )
    ensure_paths_exist(
        [
            paths.popularity_path,
            paths.reuse_path,
            paths.cache_curve_path,
            paths.per_request_miss_path,
            paths.tail_breakdown_path,
            paths.num_loras_path,
        ]
    )

    figure_records = [
        build_fig1_popularity_rank(paths.popularity_path, paths.figures_dir / "fig1_popularity_rank.pdf"),
        build_fig2_reuse_distance(paths.reuse_path, paths.figures_dir / "fig2_reuse_distance.pdf"),
        build_fig3_cache_curve(paths.cache_curve_path, paths.figures_dir / "fig3_cache_curve.pdf"),
        build_fig4_num_loras_vs_p99(paths.num_loras_path, paths.figures_dir / "fig4_num_loras_vs_p99.pdf"),
        build_fig5_tail_breakdown(
            paths.cache_curve_path,
            paths.per_request_miss_path,
            paths.tail_breakdown_path,
            paths.figures_dir / "fig5_tail_breakdown.pdf",
        ),
    ]

    manifest_path = paths.figures_dir / "figure_manifest.md"
    write_manifest(manifest_path, paths.run_id, figure_records)
    write_storyline(paths.storyline_path, paths.run_id, figure_records)
    return figure_records, manifest_path, paths.storyline_path


def main() -> None:
    args = parse_args()
    figure_records, manifest_path, storyline_path = assemble_paper_figures(
        config_path=args.config,
        run_id=args.run_id,
        figures_dir=args.figures_dir,
        sweeps_dir=args.sweeps_dir,
        storyline_path=args.storyline_path,
    )
    figures_dir = manifest_path.parent
    print(f"Wrote {len(figure_records)} figures to {figures_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote storyline to {storyline_path}")


if __name__ == "__main__":
    main()
