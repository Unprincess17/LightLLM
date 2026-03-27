#!/usr/bin/env python3
"""Assemble motivation panels and a compact 3x1 figure from locality and TPOT artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import artifact_root, ensure_parent_dir, load_global_config, resolve_run_id
else:
    from .common import artifact_root, ensure_parent_dir, load_global_config, resolve_run_id


PLOT_SCRIPT_PATH = Path("tools/case_study/assemble_motivation_figures.py")
CONDITION_ORDER = ("expert_only", "joint_indep", "joint_corr")
FIRST_CONDITION_LABELS = {
    "expert_only": "C0: (Expert-only)",
    "joint_indep": "C1: (Joint-Indep.)",
    "joint_corr": "C2: (Joint-Corr.)",
}
CONDITION_LABELS = {
    "expert_only": "C0 (Oracle)",
    "joint_indep": "C1 (Indep)",
    "joint_corr": "C2 (Corr)",
}
CONDITION_COLORS = {
    "expert_only": "#355070",
    "joint_indep": "#C8553D",
    "joint_corr": "#2A9D8F",
}
CONDITION_LINEWIDTHS = {
    "expert_only": 1.5,
    "joint_indep": 2.2,
    "joint_corr": 2.2,
}
PRACTICAL_CACHE_LIMIT = 1024
PRACTICAL_CACHE_LABEL = "Cache Limit\n(1K objs)"
METAL_FLOOR_MS = 1.2
FLOOR_TOLERANCE_MS = 0.01
DEFAULT_TAIL_BUDGET = 4096
DEFAULT_SYSTEM_TPOT_STAGE = "replay/system_tpot_stressed"
TAIL_PERCENTILE = 0.999
TAIL_PERCENTILE_LABEL = "P99.9"
TAIL_CURVE_PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, TAIL_PERCENTILE)
TAIL_CURVE_TICK_PERCENTILES = (0.50, 0.90, 0.95, 0.99, 0.995, TAIL_PERCENTILE)
TAIL_CURVE_TICK_LABELS = ("P50", "P90", "P95", "P99", "P99.5", TAIL_PERCENTILE_LABEL)
CAPACITY_TICK_BUDGETS = (128, 512, 2048, 8192, 32768, 65536)
CAPACITY_TICK_LABELS = ("128", "512", "2K", "8K", "32K", "65K")
# Stretch the "nines" so the 99th+ percentile tail does not collapse into the plot edge.
TAIL_CURVE_AXIS_BASE = math.log10(1.0 / (1.0 - TAIL_CURVE_TICK_PERCENTILES[0]))
SINGLE_COLUMN_WIDTH_IN = 3.35
STANDALONE_PANEL_SIZE = (SINGLE_COLUMN_WIDTH_IN, 2.5)
COMBINED_VERTICAL_SIZE = (SINGLE_COLUMN_WIDTH_IN, 6.55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble motivation figure panels from B7/B9 artifacts")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=None,
        help="Override output directory for motivation figures",
    )
    parser.add_argument(
        "--system_tpot_dir",
        type=str,
        default=None,
        help="Override replay/system_tpot directory (default: replay/system_tpot_stressed)",
    )
    parser.add_argument(
        "--tail_budget",
        type=int,
        default=DEFAULT_TAIL_BUDGET,
        help="Cache budget used for the token-TPOT CDF panel",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default="motivation",
        help="Filename stem used for generated outputs",
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing required motivation inputs:\n{formatted}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_parent_dir(path)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": str(PLOT_SCRIPT_PATH)})
    plt.close(fig)


def condition_rows(rows: Sequence[Mapping[str, str]], condition: str) -> List[Mapping[str, str]]:
    return [row for row in rows if str(row["condition"]) == condition]


def single_row(rows: Sequence[Mapping[str, str]], **filters: object) -> Mapping[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in filters.items()):
            return row
    raise KeyError(f"row not found for filters={filters}")


def condition_floor_budget(rows: Sequence[Mapping[str, str]], condition: str) -> int:
    subset = sorted(
        (
            row
            for row in condition_rows(rows, condition)
            if int(row["cache_budget"]) > 0
        ),
        key=lambda row: int(row["cache_budget"]),
    )
    for row in subset:
        if float(row["p99"]) <= METAL_FLOOR_MS + FLOOR_TOLERANCE_MS:
            return int(row["cache_budget"])
    return int(subset[-1]["cache_budget"])


def topk_coverage(popularity_rows: Sequence[Mapping[str, str]], condition: str, rank: int) -> float:
    subset = condition_rows(popularity_rows, condition)
    for row in subset:
        if int(row["rank"]) == rank:
            return float(row["cumulative_fraction"])
    raise KeyError(f"rank {rank} not found for condition={condition}")


def quantile_from_sorted(values: Sequence[float], quantile: float) -> float:
    if len(values) == 0:
        raise ValueError("cannot compute quantile from empty sequence")
    if quantile <= 0.0:
        return float(values[0])
    if quantile >= 1.0:
        return float(values[-1])
    position = (len(values) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def load_token_tpot_budget_slice(path: Path, budget: int) -> dict[str, List[float]]:
    series = {condition: [] for condition in CONDITION_ORDER}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["cache_budget"]) != int(budget):
                continue
            condition = str(row["condition"])
            if condition in series:
                series[condition].append(float(row["tpot_ms"]))
    for condition, values in series.items():
        if not values:
            raise ValueError(f"token_tpot.csv does not contain cache_budget={budget} for condition={condition}")
        values.sort()
    return series


def tail_percentile_axis(percentile: float) -> float:
    if not 0.0 < percentile < 1.0:
        raise ValueError(f"percentile must be between 0 and 1, got {percentile}")
    return math.log10(1.0 / (1.0 - percentile)) - TAIL_CURVE_AXIS_BASE


def tail_curve_latencies(values: Sequence[float], percentiles: Sequence[float]) -> np.ndarray:
    return np.asarray([quantile_from_sorted(values, percentile) for percentile in percentiles], dtype=float)


def plot_root_cause(ax: plt.Axes, popularity_rows: Sequence[Mapping[str, str]]) -> None:
    limit_coverages: dict[str, float] = {}
    for condition in CONDITION_ORDER:
        subset = condition_rows(popularity_rows, condition)
        ranks = [int(row["rank"]) for row in subset]
        coverage = [float(row["cumulative_fraction"]) for row in subset]
        ax.plot(
            ranks,
            coverage,
            color=CONDITION_COLORS[condition],
            linewidth=2.0,
            label=FIRST_CONDITION_LABELS[condition],
        )
        limit_coverages[condition] = topk_coverage(popularity_rows, condition, PRACTICAL_CACHE_LIMIT)

    ax.axvline(
        PRACTICAL_CACHE_LIMIT,
        color="#7A7F85",
        linewidth=1.5,
        linestyle="--",
        alpha=0.9,
        zorder=1,
    )
    ax.annotate(
        PRACTICAL_CACHE_LABEL,
        xy=(PRACTICAL_CACHE_LIMIT, 0.985),
        xycoords=ax.get_xaxis_transform(),
        xytext=(-5, -1),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=10,
        color="#4A4F55",
        fontweight="bold",
    )

    label_offsets = {
        "expert_only": (6, 8),
        "joint_indep": (6, 2),
        "joint_corr": (6, -8),
    }
    for condition in CONDITION_ORDER:
        coverage = limit_coverages[condition]
        ax.plot(
            PRACTICAL_CACHE_LIMIT,
            coverage,
            marker="o",
            markersize=5.8,
            color=CONDITION_COLORS[condition],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{100.0 * coverage:.1f}%",
            xy=(PRACTICAL_CACHE_LIMIT, coverage),
            xytext=label_offsets[condition],
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color=CONDITION_COLORS[condition],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12, "alpha": 0.9},
        )

    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Expert-LoRA Rank")
    ax.set_ylabel("Cumulative Access Fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.85,
        fontsize=8.5,
        facecolor="white",
        edgecolor="#D7DCE0",
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.5,
        handletextpad=0.4,
    )


def plot_capacity_illusion(ax: plt.Axes, quantile_rows: Sequence[Mapping[str, str]]) -> None:
    for condition in CONDITION_ORDER:
        subset = sorted(
            (
                row
                for row in condition_rows(quantile_rows, condition)
                if 128 <= int(row["cache_budget"]) <= 65536
            ),
            key=lambda row: int(row["cache_budget"]),
        )
        budgets = [int(row["cache_budget"]) for row in subset]
        p99s = [float(row["p99"]) for row in subset]
        ax.plot(
            budgets,
            p99s,
            marker="o",
            markersize=3.8,
            linewidth=2.0,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )

    floor_b0 = condition_floor_budget(quantile_rows, "expert_only")
    floor_b2 = condition_floor_budget(quantile_rows, "joint_corr")
    gap_ratio = floor_b2 / floor_b0
    gap_label = (
        f"{int(gap_ratio)}x Capacity Gap"
        if float(gap_ratio).is_integer()
        else f"{gap_ratio:.1f}x Capacity Gap"
    )
    gap_curve_ceiling = max(
        float(row["p99"])
        for row in quantile_rows
        if floor_b0 <= int(row["cache_budget"]) <= floor_b2
    )
    gap_arrow_y = gap_curve_ceiling + 0.18
    gap_label_x = math.sqrt(floor_b0 * floor_b2)

    ax.axhline(METAL_FLOOR_MS, color="#7A7F85", linewidth=1.1, linestyle="--", alpha=0.9)
    ax.text(
        150, 
        1.25,
        "C0 P99 Target",
        fontsize=10
    )
    ax.axvline(floor_b0, color="#AEB6BF", linewidth=0.95, linestyle=":", alpha=0.6)
    ax.axvline(floor_b2, color="#AEB6BF", linewidth=0.95, linestyle=":", alpha=0.6)
    # Keep the gap annotation above the descending curves so it reads as a bracket, not another data series.
    ax.annotate(
        "",
        xy=(floor_b2, gap_arrow_y),
        xytext=(floor_b0, gap_arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "#3A3F44", "lw": 1.35, "shrinkA": 0, "shrinkB": 0},
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(128, 65536)
    ax.set_ylim(1.0, max(7.35, gap_arrow_y + 0.55))
    ax.set_xticks(CAPACITY_TICK_BUDGETS)
    ax.set_xticklabels(CAPACITY_TICK_LABELS)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.text(
        gap_label_x,
        gap_arrow_y + 0.07,
        gap_label,
        fontsize=10,
        fontweight="semibold",
        ha="center",
        va="bottom",
        color="#3A3F44",
    )
    ax.set_xlabel("Cache Budget (#objects)")
    ax.set_ylabel("P99 TPOT (ms)")
    # ax.legend(
    #     loc="upper right",
    #     frameon=True,
    #     facecolor="white",
    #     edgecolor="#D7DCE0",
    #     borderpad=0.35,
    #     labelspacing=0.25,
    #     handlelength=2.0,
    # )


def plot_tail_blowout(ax: plt.Axes, token_tpot_by_condition: Mapping[str, Sequence[float]], _budget: int) -> None:
    curve_x = np.asarray([tail_percentile_axis(percentile) for percentile in TAIL_CURVE_PERCENTILES], dtype=float)
    tick_x = [tail_percentile_axis(percentile) for percentile in TAIL_CURVE_TICK_PERCENTILES]
    tail_values: dict[str, np.ndarray] = {}

    for condition in CONDITION_ORDER:
        latencies = tail_curve_latencies(token_tpot_by_condition[condition], TAIL_CURVE_PERCENTILES)
        tail_values[condition] = latencies
        ax.plot(
            curve_x,
            latencies,
            linewidth=CONDITION_LINEWIDTHS[condition],
            alpha=0.85,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )

    c0_tail = tail_values["expert_only"][-1]
    c1_tail = tail_values["joint_indep"][-1]
    c2_tail = tail_values["joint_corr"][-1]
    tail_tip_x = tick_x[-1]
    flat_tip_x = tick_x[0]
    max_latency = max(float(np.max(latencies)) for latencies in tail_values.values())

    ax.axhline(METAL_FLOOR_MS, color="#7A7F85", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.set_xlim(curve_x[0] - 0.06, curve_x[-1] + 0.18)
    ax.set_ylim(1.0, max_latency + 0.35)
    ax.set_xticks(tick_x)
    ax.set_xticklabels(TAIL_CURVE_TICK_LABELS)
    ax.set_xlabel("Token Percentile")
    ax.set_ylabel("TPOT (ms)")
    # ax.legend(
    #     loc="lower left",
    #     bbox_to_anchor=(0.0, 0.03),
    #     frameon=True,
    #     facecolor="white",
    #     edgecolor="#D7DCE0",
    #     borderpad=0.35,
    #     labelspacing=0.25,
    #     handlelength=2.0,
    # )
    ax.annotate(
        f"C1/C2 {TAIL_PERCENTILE_LABEL} = {max(c1_tail, c2_tail):.2f} ms\n≈ 6× slowdown",
        xy=(tail_tip_x, max(c1_tail, c2_tail)),
        xytext=(tail_tip_x - 2, max(c1_tail, c2_tail) - 0.72),
        arrowprops={"arrowstyle": "->", "color": CONDITION_COLORS["joint_corr"], "lw": 1.15},
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.22"},
    )
    ax.annotate(
        f"Median (P50) ≈ {c0_tail:.2f} ms",
        xy=(flat_tip_x, c0_tail),
        xytext=(flat_tip_x, c0_tail + 0.8),
        arrowprops={"arrowstyle": "->", "color": CONDITION_COLORS["expert_only"], "lw": 1.1},
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.22"},
    )


def build_standalone_panel(
    plot_fn,
    output_path: Path,
    *plot_args,
    figsize: tuple[float, float] = STANDALONE_PANEL_SIZE,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_fn(ax, *plot_args)
    save_figure(fig, output_path)


def resolve_paths(
    config_path: str | None,
    run_id: str | None,
    figures_dir: str | None,
    system_tpot_dir: str | None,
) -> dict[str, Path]:
    config = load_global_config(config_path)
    resolved_run_id = resolve_run_id(config, run_id)
    run_root = artifact_root(config) / "case_study" / resolved_run_id
    resolved_system_tpot_dir = Path(system_tpot_dir) if system_tpot_dir else run_root / DEFAULT_SYSTEM_TPOT_STAGE
    resolved_figures_dir = Path(figures_dir) if figures_dir else run_root / "figures" / "motivation"
    return {
        "run_root": run_root,
        "figures_dir": resolved_figures_dir,
        "popularity_path": run_root / "replay" / "locality" / "popularity_rank.csv",
        "quantiles_path": resolved_system_tpot_dir / "tpot_quantiles.csv",
        "token_tpot_path": resolved_system_tpot_dir / "token_tpot.csv",
    }


def main() -> None:
    args = parse_args()
    apply_style()

    paths = resolve_paths(
        config_path=args.config,
        run_id=args.run_id,
        figures_dir=args.figures_dir,
        system_tpot_dir=args.system_tpot_dir,
    )
    ensure_paths_exist(
        [
            paths["popularity_path"],
            paths["quantiles_path"],
            paths["token_tpot_path"],
        ]
    )

    popularity_rows = read_csv_rows(paths["popularity_path"])
    quantile_rows = read_csv_rows(paths["quantiles_path"])
    token_tpot_by_condition = load_token_tpot_budget_slice(paths["token_tpot_path"], args.tail_budget)

    output_stem = args.output_stem
    figures_dir = paths["figures_dir"]
    panel_a_pdf = figures_dir / f"{output_stem}_panel_a_root_cause.pdf"
    panel_b_pdf = figures_dir / f"{output_stem}_panel_b_capacity_illusion.pdf"
    panel_c_pdf = figures_dir / f"{output_stem}_panel_c_tail_blowout_budget_{args.tail_budget}.pdf"
    combined_pdf = figures_dir / f"{output_stem}_figure.pdf"
    panel_a_png = panel_a_pdf.with_suffix(".png")
    panel_b_png = panel_b_pdf.with_suffix(".png")
    panel_c_png = panel_c_pdf.with_suffix(".png")
    combined_png = combined_pdf.with_suffix(".png")

    build_standalone_panel(plot_root_cause, panel_a_pdf, popularity_rows)
    build_standalone_panel(plot_root_cause, panel_a_png, popularity_rows)
    build_standalone_panel(plot_capacity_illusion, panel_b_pdf, quantile_rows)
    build_standalone_panel(plot_capacity_illusion, panel_b_png, quantile_rows)
    build_standalone_panel(plot_tail_blowout, panel_c_pdf, token_tpot_by_condition, args.tail_budget)
    build_standalone_panel(plot_tail_blowout, panel_c_png, token_tpot_by_condition, args.tail_budget)

    fig, axes = plt.subplots(3, 1, figsize=COMBINED_VERTICAL_SIZE, constrained_layout=True)
    plot_root_cause(axes[0], popularity_rows)
    plot_capacity_illusion(axes[1], quantile_rows)
    plot_tail_blowout(axes[2], token_tpot_by_condition, args.tail_budget)
    save_figure(fig, combined_pdf)

    fig, axes = plt.subplots(3, 1, figsize=COMBINED_VERTICAL_SIZE, constrained_layout=True)
    plot_root_cause(axes[0], popularity_rows)
    plot_capacity_illusion(axes[1], quantile_rows)
    plot_tail_blowout(axes[2], token_tpot_by_condition, args.tail_budget)
    save_figure(fig, combined_png)

    floor_b0 = condition_floor_budget(quantile_rows, "expert_only")
    floor_b2 = condition_floor_budget(quantile_rows, "joint_corr")
    c0_practical = topk_coverage(popularity_rows, "expert_only", PRACTICAL_CACHE_LIMIT)
    c1_practical = topk_coverage(popularity_rows, "joint_indep", PRACTICAL_CACHE_LIMIT)
    c2_practical = topk_coverage(popularity_rows, "joint_corr", PRACTICAL_CACHE_LIMIT)
    c0_p99 = quantile_from_sorted(token_tpot_by_condition["expert_only"], TAIL_PERCENTILE)
    c1_p99 = quantile_from_sorted(token_tpot_by_condition["joint_indep"], TAIL_PERCENTILE)
    c2_p99 = quantile_from_sorted(token_tpot_by_condition["joint_corr"], TAIL_PERCENTILE)

    print(f"Wrote panel A to {panel_a_pdf} and {panel_a_png}")
    print(f"Wrote panel B to {panel_b_pdf} and {panel_b_png}")
    print(f"Wrote panel C to {panel_c_pdf} and {panel_c_png}")
    print(f"Wrote combined figure to {combined_pdf} and {combined_png}")
    print(
        f"Panel A practical-limit coverage at rank={PRACTICAL_CACHE_LIMIT}: "
        f"C0={100.0 * c0_practical:.1f}%, C1={100.0 * c1_practical:.1f}%, C2={100.0 * c2_practical:.1f}%"
    )
    print(
        "Panel B floor budgets: "
        f"C0={floor_b0}, C2={floor_b2}, "
        f"metal_floor_ms={METAL_FLOOR_MS:.2f}, tolerance_ms={FLOOR_TOLERANCE_MS:.2f}"
    )
    print(
        f"Panel C budget={args.tail_budget} {TAIL_PERCENTILE_LABEL} TPOT: "
        f"C0={c0_p99:.3f} ms, C1={c1_p99:.3f} ms, C2={c2_p99:.3f} ms"
    )


if __name__ == "__main__":
    main()
