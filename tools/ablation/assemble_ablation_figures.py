#!/usr/bin/env python3
"""Assemble paper-facing figures from ablation outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from ablation_utils import read_csv_rows, resolve_ablation_output_paths, resolve_config_and_run_id
from export_timeline import plot_timeline, read_timeline_rows
from variants import VARIANTS


VARIANT_COLORS = {
    "colora_full": "#355070",
    "no_cpu_path": "#C8553D",
    "no_overlap": "#2A9D8F",
    "no_deferred_sync": "#A44A3F",
    "no_deferred_never": "#8D99AE",
    "no_prefetch": "#E9C46A",
    "expert_only": "#6D597A",
}
BREAKDOWN_COMPONENTS = (
    ("gpu_wait_ms", "GPU Wait"),
    ("cpu_compute_ms", "CPU Compute"),
    ("d2h_h2d_ms", "D2H/H2D"),
    ("merge_ms", "Merge"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble figures from ablation metrics")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Run id")
    parser.add_argument("--suite_id", type=str, default="core", help="Suite id")
    parser.add_argument("--suite_dir", type=str, default=None, help="Explicit suite output root")
    parser.add_argument("--budget", type=int, default=2048, help="Budget used for the breakdown and timeline figures")
    return parser.parse_args()


def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = float(sum(values) / len(values))
    std_value = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean_value, std_value


def plot_p99_vs_budget(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    variants = []
    for row in rows:
        variant = str(row["variant"])
        if variant not in variants:
            variants.append(variant)
    for variant in variants:
        by_budget: Dict[int, List[float]] = {}
        for row in rows:
            if str(row["variant"]) != variant:
                continue
            by_budget.setdefault(int(row["cache_budget"]), []).append(float(row["p99"]))
        budgets = sorted(by_budget)
        means = [mean_and_std(by_budget[budget])[0] for budget in budgets]
        stds = [mean_and_std(by_budget[budget])[1] for budget in budgets]
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=VARIANT_COLORS.get(variant, "#444444"),
            label=VARIANTS.get(variant).label if variant in VARIANTS else variant,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cache Budget (objects)")
    ax.set_ylabel("Token P99 TPOT (ms)")
    ax.set_title("Ablation P99 TPOT vs. Cache Budget")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#D7DCE0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(rows: Sequence[Mapping[str, str]], budget: int, output_path: Path) -> None:
    filtered = [row for row in rows if int(row["cache_budget"]) == int(budget)]
    variants = []
    for row in filtered:
        variant = str(row["variant"])
        if variant not in variants:
            variants.append(variant)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x_positions = range(len(variants))
    stacked_bottom = [0.0 for _ in variants]
    for component_key, component_label in BREAKDOWN_COMPONENTS:
        component_means = []
        for variant in variants:
            values = [float(row[component_key]) for row in filtered if str(row["variant"]) == variant]
            component_means.append(mean_and_std(values)[0])
        ax.bar(
            list(x_positions),
            component_means,
            bottom=stacked_bottom,
            label=component_label,
            edgecolor="#1F2933",
            linewidth=0.5,
        )
        stacked_bottom = [bottom + value for bottom, value in zip(stacked_bottom, component_means)]
    overlap_means = []
    for variant in variants:
        values = [float(row["overlap_hidden_ms"]) for row in filtered if str(row["variant"]) == variant]
        overlap_means.append(mean_and_std(values)[0])
    ax.plot(list(x_positions), overlap_means, marker="D", color="#C1121F", linewidth=1.6, label="Hidden Overlap")
    ax.set_xticks(list(x_positions), [VARIANTS.get(variant).label if variant in VARIANTS else variant for variant in variants], rotation=12)
    ax.set_ylabel("Mean Per-Token Service (ms)")
    ax.set_title(f"Ablation Breakdown at Budget={budget}")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#D7DCE0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config, run_id = resolve_config_and_run_id(args.config, args.run_id)
    suite_root = Path(args.suite_dir) if args.suite_dir else resolve_ablation_output_paths(config, run_id, str(args.suite_id)).root_dir
    metrics_dir = suite_root / "metrics"
    figures_dir = suite_root / "figures"
    timelines_dir = suite_root / "timelines"
    quantile_rows = read_csv_rows(metrics_dir / "ablation_quantiles.csv")
    breakdown_rows = read_csv_rows(metrics_dir / "ablation_breakdown.csv")
    timeline_rows = read_timeline_rows(timelines_dir / "ablation_timeline.csv")

    fig_p99 = figures_dir / "fig_ablation_p99_vs_budget.pdf"
    fig_breakdown = figures_dir / f"fig_ablation_breakdown_budget_{int(args.budget)}.pdf"
    fig_timeline = figures_dir / f"fig_ablation_timeline_budget_{int(args.budget)}.pdf"

    plot_p99_vs_budget(quantile_rows, fig_p99)
    plot_breakdown(breakdown_rows, int(args.budget), fig_breakdown)
    if timeline_rows:
        plot_timeline(timeline_rows, fig_timeline, title=f"Ablation Timeline at Budget={int(args.budget)}")

    write_manifest(
        figures_dir / "ablation_figure_manifest.md",
        [
            "# Ablation Figure Manifest",
            "",
            f"- Suite root: `{suite_root}`",
            f"- Quantiles source: `{metrics_dir / 'ablation_quantiles.csv'}`",
            f"- Breakdown source: `{metrics_dir / 'ablation_breakdown.csv'}`",
            f"- Timeline source: `{timelines_dir / 'ablation_timeline.csv'}`",
            f"- Figure: `{fig_p99}`",
            f"- Figure: `{fig_breakdown}`",
            f"- Figure: `{fig_timeline}`",
        ],
    )
    print(f"Wrote ablation figures to {figures_dir}")


if __name__ == "__main__":
    main()
