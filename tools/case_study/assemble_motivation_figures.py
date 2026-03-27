#!/usr/bin/env python3
"""Assemble motivation panels and a combined 1x3 figure from locality and TPOT artifacts."""

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
from matplotlib.ticker import FuncFormatter, PercentFormatter

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import artifact_root, ensure_parent_dir, load_global_config, resolve_run_id
else:
    from .common import artifact_root, ensure_parent_dir, load_global_config, resolve_run_id


PLOT_SCRIPT_PATH = Path("tools/case_study/assemble_motivation_figures.py")
CONDITION_ORDER = ("expert_only", "joint_indep", "joint_corr")
CONDITION_LABELS = {
    "expert_only": "C0",
    "joint_indep": "C1",
    "joint_corr": "C2",
}
CONDITION_COLORS = {
    "expert_only": "#355070",
    "joint_indep": "#C8553D",
    "joint_corr": "#2A9D8F",
}
METAL_FLOOR_MS = 1.2
FLOOR_TOLERANCE_MS = 0.01
DEFAULT_TAIL_BUDGET = 4096
DEFAULT_SYSTEM_TPOT_STAGE = "replay/system_tpot_stressed"


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
            "grid.linewidth": 0.7,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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


def build_ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.asarray(values, dtype=float)
    probs = np.arange(1, len(ordered) + 1, dtype=float) / float(len(ordered))
    return ordered, probs


def plot_root_cause(ax: plt.Axes, popularity_rows: Sequence[Mapping[str, str]]) -> None:
    top1k_lines = []
    for condition in CONDITION_ORDER:
        subset = condition_rows(popularity_rows, condition)
        ranks = [int(row["rank"]) for row in subset]
        coverage = [float(row["cumulative_fraction"]) for row in subset]
        ax.plot(
            ranks,
            coverage,
            color=CONDITION_COLORS[condition],
            linewidth=2.4,
            label=CONDITION_LABELS[condition],
        )
        top1k_lines.append(
            f"{CONDITION_LABELS[condition]} top-1k: {100.0 * topk_coverage(popularity_rows, condition, 1000):.1f}%"
        )

    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Rank of Expert-LoRA Object")
    ax.set_ylabel("Cumulative Access Fraction")
    ax.set_title("A. Root Cause: Algorithmic Fragmentation")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D7DCE0")
    ax.text(
        0.03,
        0.72,
        "\n".join(top1k_lines),
        transform=ax.transAxes,
        fontsize=9.8,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.3"},
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
            markersize=4.8,
            linewidth=2.3,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )

    floor_b0 = condition_floor_budget(quantile_rows, "expert_only")
    floor_b2 = condition_floor_budget(quantile_rows, "joint_corr")
    b0_floor_row = single_row(quantile_rows, condition="expert_only", cache_budget=floor_b0)
    b2_floor_row = single_row(quantile_rows, condition="joint_corr", cache_budget=floor_b2)
    b2_small_row = single_row(quantile_rows, condition="joint_corr", cache_budget=floor_b0)

    ax.axhline(METAL_FLOOR_MS, color="#7A7F85", linewidth=1.1, linestyle="--", alpha=0.9)
    ax.axvline(floor_b0, color=CONDITION_COLORS["expert_only"], linewidth=1.0, linestyle=":", alpha=0.75)
    ax.axvline(floor_b2, color=CONDITION_COLORS["joint_corr"], linewidth=1.0, linestyle=":", alpha=0.75)
    ax.annotate(
        f"C0 reaches {float(b0_floor_row['p99']):.2f} ms\nat budget={floor_b0}",
        xy=(floor_b0, float(b0_floor_row["p99"])),
        xytext=(0.06, 0.22),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": CONDITION_COLORS["expert_only"], "lw": 1.1},
        fontsize=9.4,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.25"},
    )
    ax.annotate(
        f"C2 is still {float(b2_small_row['p99']):.2f} ms\nat {floor_b0}, needs {floor_b2}",
        xy=(floor_b2, float(b2_floor_row["p99"])),
        xytext=(0.51, 0.78),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": CONDITION_COLORS["joint_corr"], "lw": 1.1},
        fontsize=9.4,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.25"},
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(128, 65536)
    ax.set_ylim(1.0, 7.35)
    ax.set_xlabel("Cache Budget")
    ax.set_ylabel("P99 TPOT (ms)")
    ax.set_title("B. Capacity Illusion")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D7DCE0")


def plot_tail_blowout(ax: plt.Axes, token_tpot_by_condition: Mapping[str, Sequence[float]], budget: int) -> None:
    summary_lines = []
    for condition in CONDITION_ORDER:
        x, y = build_ecdf(token_tpot_by_condition[condition])
        ax.plot(
            x,
            y,
            linewidth=2.3,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
        summary_lines.append(
            f"{CONDITION_LABELS[condition]} P99.9: {quantile_from_sorted(x, 0.999):.2f} ms"
        )

    ax.axvline(METAL_FLOOR_MS, color="#7A7F85", linewidth=1.1, linestyle="--", alpha=0.9)
    ax.set_xlim(1.1, 7.25)
    ax.set_ylim(0.50, 0.999)
    ax.set_xlabel("Token TPOT (ms)")
    ax.set_ylabel("Token Percentile")
    ax.set_title(f"C. Tail Blowout at Budget={budget}")
    ax.set_yticks([0.50, 0.90, 0.95, 0.99, 0.999])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100.0 * value:.1f}%"))
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D7DCE0")
    ax.text(
        0.03,
        0.70,
        "P50 all conditions: 1.20 ms\n" + "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9.7,
        bbox={"facecolor": "white", "edgecolor": "#D7DCE0", "boxstyle": "round,pad=0.3"},
    )


def build_standalone_panel(
    plot_fn,
    output_path: Path,
    *plot_args,
    figsize: tuple[float, float] = (5.3, 3.8),
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

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.25), constrained_layout=True)
    plot_root_cause(axes[0], popularity_rows)
    plot_capacity_illusion(axes[1], quantile_rows)
    plot_tail_blowout(axes[2], token_tpot_by_condition, args.tail_budget)
    save_figure(fig, combined_pdf)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.25), constrained_layout=True)
    plot_root_cause(axes[0], popularity_rows)
    plot_capacity_illusion(axes[1], quantile_rows)
    plot_tail_blowout(axes[2], token_tpot_by_condition, args.tail_budget)
    save_figure(fig, combined_png)

    floor_b0 = condition_floor_budget(quantile_rows, "expert_only")
    floor_b2 = condition_floor_budget(quantile_rows, "joint_corr")
    c0_top1k = topk_coverage(popularity_rows, "expert_only", 1000)
    c1_top1k = topk_coverage(popularity_rows, "joint_indep", 1000)
    c2_top1k = topk_coverage(popularity_rows, "joint_corr", 1000)
    c0_p999 = quantile_from_sorted(token_tpot_by_condition["expert_only"], 0.999)
    c1_p999 = quantile_from_sorted(token_tpot_by_condition["joint_indep"], 0.999)
    c2_p999 = quantile_from_sorted(token_tpot_by_condition["joint_corr"], 0.999)

    print(f"Wrote panel A to {panel_a_pdf} and {panel_a_png}")
    print(f"Wrote panel B to {panel_b_pdf} and {panel_b_png}")
    print(f"Wrote panel C to {panel_c_pdf} and {panel_c_png}")
    print(f"Wrote combined figure to {combined_pdf} and {combined_png}")
    print(
        "Panel A top-1k coverage: "
        f"C0={100.0 * c0_top1k:.1f}%, C1={100.0 * c1_top1k:.1f}%, C2={100.0 * c2_top1k:.1f}%"
    )
    print(
        "Panel B floor budgets: "
        f"C0={floor_b0}, C2={floor_b2}, "
        f"metal_floor_ms={METAL_FLOOR_MS:.2f}, tolerance_ms={FLOOR_TOLERANCE_MS:.2f}"
    )
    print(
        f"Panel C budget={args.tail_budget} P99.9 TPOT: "
        f"C0={c0_p999:.3f} ms, C1={c1_p999:.3f} ms, C2={c2_p999:.3f} ms"
    )


if __name__ == "__main__":
    main()
