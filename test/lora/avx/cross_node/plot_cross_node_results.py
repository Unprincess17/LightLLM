#!/usr/bin/env python3
"""
Plot cross-node benchmark results.

Reads: results/cross_node_benchmark/cross_node_results.csv
Outputs:
  - fig3a_strategy_comparison.png/pdf
  - fig3b_latency_breakdown.png/pdf
  - fig3c_degradation_ratio.png/pdf
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> List[dict]:
    """Load and type-coerce the benchmark CSV."""
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["rank"] = int(row["rank"])
            row["num_miss"] = int(row["num_miss"])
            row["strategy"] = int(row["strategy"])
            row["ep_bw_pct"] = int(row["ep_bw_pct"])
            timing_fields = [
                "rdma_weight_ms", "d2h_ms", "rdma_activation_ms",
                "cpu_compute_ms", "rdma_result_ms", "h2d_ms",
                "gpu_compute_ms", "total_ms",
            ]
            for field in timing_fields:
                val = row.get(field, "")
                row[field] = float(val) if val.strip() else None
            rows.append(row)
    return rows


def validate_csv(rows: List[dict]) -> None:
    """Basic sanity checks on loaded data."""
    if not rows:
        raise ValueError("CSV is empty")
    required = {"rank", "num_miss", "strategy", "ep_bw_pct", "total_ms"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    print(f"[validate] {len(rows)} rows loaded, "
          f"ranks={sorted(set(r['rank'] for r in rows))}, "
          f"num_miss={sorted(set(r['num_miss'] for r in rows))}, "
          f"strategies={sorted(set(r['strategy'] for r in rows))}, "
          f"EP levels={sorted(set(r['ep_bw_pct'] for r in rows))}")


# ---------------------------------------------------------------------------
# Data grouping helpers
# ---------------------------------------------------------------------------

def group_by_config(rows: List[dict]) -> Dict[Tuple, List[dict]]:
    """Group rows by (rank, num_miss, strategy)."""
    out: Dict[Tuple, List[dict]] = defaultdict(list)
    for row in rows:
        key = (row["rank"], row["num_miss"], row["strategy"])
        out[key].append(row)
    return out


def group_by_ep(rows: List[dict]) -> Dict[int, List[dict]]:
    """Group rows by EP background traffic level."""
    out: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        out[row["ep_bw_pct"]].append(row)
    return out


# ---------------------------------------------------------------------------
# Color / style helpers
# ---------------------------------------------------------------------------

EP_LEVELS = [0, 25, 50, 75, 90]
EP_LABELS = [f"{v}%" for v in EP_LEVELS]

# Distinct colors for (rank, num_miss) combos
COMBO_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def combo_label(rank: int, num_miss: int) -> str:
    return f"R={rank} NM={num_miss}"


def combo_color(idx: int) -> str:
    return COMBO_PALETTE[idx % len(COMBO_PALETTE)]


# ---------------------------------------------------------------------------
# Figure 3a – Strategy Comparison (EP traffic on X, latency on Y)
# ---------------------------------------------------------------------------

def plot_figure3a(
    rows: List[dict],
    output_dir: Path,
    formats: List[str] = ["png"],
) -> None:
    """
    Multi-panel plot: one subplot per (rank, num_miss) combo.
    Each subplot: X = EP traffic level, Y = total latency (ms),
    two lines (solid = Strategy 1, dashed = Strategy 2).
    """
    # Collect unique combos sorted for stable ordering
    combos = sorted(set((r["rank"], r["num_miss"]) for r in rows))
    n_combos = len(combos)

    ncols = min(3, n_combos)
    nrows = (n_combos + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )

    for ax_idx, (rank, num_miss) in enumerate(combos):
        ax = axes[ax_idx // ncols][ax_idx % ncols]

        for strategy in [1, 2, 3]:
            subset = [
                r for r in rows
                if r["rank"] == rank
                and r["num_miss"] == num_miss
                and r["strategy"] == strategy
            ]
            if not subset:
                continue

            # Sort by EP level
            subset = sorted(subset, key=lambda r: r["ep_bw_pct"])
            ep_vals = [r["ep_bw_pct"] for r in subset]
            lat_vals = [r["total_ms"] for r in subset]

            ls_map = {1: "-", 2: "--", 3: "-."}
            lw_map = {1: 2.0, 2: 1.5, 3: 1.5}
            color_map = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50"}
            ls = ls_map[strategy]
            lw = lw_map[strategy]
            label = f"Strategy {strategy}"
            color = color_map[strategy]
            ax.plot(
                ep_vals, lat_vals,
                marker="o", linestyle=ls, linewidth=lw,
                color=color, label=label,
                markersize=5,
            )

        ax.set_xlabel("EP Background Traffic (%)")
        ax.set_ylabel("Total Latency (ms)")
        ax.set_title(f"R={rank}, NumMiss={num_miss}")
        ax.set_xticks(EP_LEVELS)
        ax.set_xticklabels([f"{v}" for v in EP_LEVELS])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for ax_idx in range(len(combos), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.suptitle("Figure 3a: Latency vs EP Background Traffic — All 3 Strategies", fontsize=14, y=1.0)
    fig.tight_layout()

    for fmt in formats:
        out_path = output_dir / f"fig3a_strategy_comparison.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3b – Latency Breakdown (stacked bar)
# ---------------------------------------------------------------------------

def _breakdown_segments(row: dict, strategy: int) -> List[Tuple[str, Optional[float]]]:
    """Return ordered (label, duration_ms) segments for a row."""
    if strategy == 1:
        return [
            ("RDMA Weight Transfer", row.get("rdma_weight_ms")),
            ("GPU Compute",          row.get("gpu_compute_ms")),
        ]
    elif strategy == 2:
        return [
            ("D2H Transfer",          row.get("d2h_ms")),
            ("RDMA Activation",       row.get("rdma_activation_ms")),
            ("CPU Compute",           row.get("cpu_compute_ms")),
            ("RDMA Result",           row.get("rdma_result_ms")),
            ("H2D Transfer",          row.get("h2d_ms")),
        ]
    else:  # strategy == 3
        return [
            ("D2H Transfer",          row.get("d2h_ms")),
            ("RDMA Act Relay",        row.get("rdma_activation_ms")),
            ("GPU Compute",           row.get("gpu_compute_ms")),
            ("RDMA Result",           row.get("rdma_result_ms")),
        ]


# Segment color palette
SEGMENT_COLORS = {
    "RDMA Weight Transfer": "#1976D2",
    "GPU Compute":          "#D32F2F",
    "D2H Transfer":         "#81D4FA",
    "RDMA Activation":      "#0288D1",
    "RDMA Act Relay":       "#0288D1",
    "CPU Compute":          "#388E3C",
    "RDMA Result":          "#7B1FA2",
    "H2D Transfer":         "#F57C00",
}


def plot_figure3b(
    rows: List[dict],
    output_dir: Path,
    formats: List[str] = ["png"],
) -> None:
    """
    Grouped stacked-bar chart comparing Strategy 1 vs Strategy 2
    for representative configs at EP=0% and EP=75%.
    """
    # Select representative configs
    target_combos = [(32, 2)]
    ep_levels = [0, 75]

    n_groups = len(target_combos) * len(ep_levels)
    x_labels: List[str] = []
    strategies_data: List[Tuple[int, List[Tuple[str, Optional[float]]]]] = []

    for rank, num_miss in target_combos:
        for ep in ep_levels:
            label = f"R={rank} NM={num_miss}\nEP={ep}%"
            x_labels.append(label)
            for strat in [1, 2]:
                subset = [
                    r for r in rows
                    if r["rank"] == rank
                    and r["num_miss"] == num_miss
                    and r["ep_bw_pct"] == ep
                    and r["strategy"] == strat
                ]
                if subset:
                    row = subset[0]
                    strategies_data.append((strat, _breakdown_segments(row, strat)))
                else:
                    strategies_data.append((strat, []))

    n_bars = len(x_labels) * 2  # 2 strategies per group
    bar_width = 0.35
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(max(8, n_bars * 0.9), 6))

    strat1_bars = []  # list of bar containers for strategy 1
    strat2_bars = []  # list of bar containers for strategy 2

    offset_base = -bar_width / 2

    # Build bottom arrays for stacking
    s1_bottom = np.zeros(len(x_labels))
    s2_bottom = np.zeros(len(x_labels))

    # Collect all segment names per strategy
    s1_segments: List[str] = []
    s2_segments: List[str] = []

    # First pass: collect segment names in order
    for (strat, segs) in strategies_data:
        if strat == 1:
            for label, _ in segs:
                if label not in s1_segments:
                    s1_segments.append(label)
        else:
            for label, _ in segs:
                if label not in s2_segments:
                    s2_segments.append(label)

    # Plot strategy 1
    for seg_label in s1_segments:
        heights = []
        for i, (strat, segs) in enumerate(strategies_data):
            if strat != 1:
                heights.append(0.0)
                continue
            val = next((v for l, v in segs if l == seg_label), 0.0)
            heights.append(val if val is not None else 0.0)
        color = SEGMENT_COLORS.get(seg_label, "#999999")
        bars = ax.bar(
            x + offset_base, heights, bar_width,
            bottom=s1_bottom, label=f"S1: {seg_label}",
            color=color, edgecolor="white", linewidth=0.5,
        )
        s1_bottom += np.array(heights)

    # Plot strategy 2
    for seg_label in s2_segments:
        heights = []
        for i, (strat, segs) in enumerate(strategies_data):
            if strat != 2:
                heights.append(0.0)
                continue
            val = next((v for l, v in segs if l == seg_label), 0.0)
            heights.append(val if val is not None else 0.0)
        color = SEGMENT_COLORS.get(seg_label, "#999999")
        bars = ax.bar(
            x - offset_base, heights, bar_width,
            bottom=s2_bottom, label=f"S2: {seg_label}",
            color=color, edgecolor="white", linewidth=0.5,
            hatch="//",
        )
        s2_bottom += np.array(heights)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_title("Figure 3b: Latency Breakdown – Strategy 1 vs Strategy 2")
    ax.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=7,
        ncol=1,
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate total latency on top of each bar pair
    for xi, xlabel in enumerate(x_labels):
        s1_total = s1_bottom[xi]
        s2_total = s2_bottom[xi]
        ax.annotate(
            f"S1\n{s1_total:.1f}ms",
            xy=(x[xi] + offset_base, s1_total),
            ha="center", va="bottom", fontsize=7,
            color=SEGMENT_COLORS["GPU Compute"],
        )
        ax.annotate(
            f"S2\n{s2_total:.1f}ms",
            xy=(x[xi] - offset_base, s2_total),
            ha="center", va="bottom", fontsize=7,
            color=SEGMENT_COLORS["CPU Compute"],
        )

    fig.tight_layout()

    for fmt in formats:
        out_path = output_dir / f"fig3b_latency_breakdown.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3c – Degradation Ratio
# ---------------------------------------------------------------------------

def plot_figure3c(
    rows: List[dict],
    output_dir: Path,
    formats: List[str] = ["png"],
) -> None:
    """
    X = EP background traffic level, Y = degradation ratio
    (latency at EP / latency at EP=0).
    One line per strategy (averaged across all configs).
    Dashed horizontal line at y=1.0.
    """
    # Collect (strategy, ep_bw_pct) -> list of ratios
    ratio_map: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    # Group by (rank, num_miss, strategy) first
    by_config = group_by_config(rows)

    for (rank, num_miss, strat), config_rows in by_config.items():
        by_ep = {r["ep_bw_pct"]: r for r in config_rows}

        base_latency = by_ep.get(0, {}).get("total_ms")
        if base_latency is None or base_latency <= 0:
            continue

        for ep in EP_LEVELS:
            if ep == 0:
                ratio_map[(strat, ep)].append(1.0)
                continue
            ep_row = by_ep.get(ep)
            if ep_row is None:
                continue
            ep_latency = ep_row.get("total_ms")
            if ep_latency is None:
                continue
            ratio = ep_latency / base_latency
            ratio_map[(strat, ep)].append(ratio)

    # Aggregate: mean ratio per (strategy, ep)
    strat_points: Dict[int, Tuple[List[int], List[float]]] = {1: ([], []), 2: ([], []), 3: ([], [])}
    for (strat, ep), ratios in sorted(ratio_map.items()):
        if ratios:
            strat_points[strat][0].append(ep)
            strat_points[strat][1].append(np.mean(ratios))

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {1: "#1976D2", 2: "#D32F2F", 3: "#4CAF50"}
    styles = {1: "-", 2: "--", 3: "-."}

    for strat in [1, 2, 3]:
        eps, means = strat_points[strat]
        if not eps:
            continue
        # Sort by EP
        sorted_pairs = sorted(zip(eps, means))
        eps_s, means_s = zip(*sorted_pairs)
        ax.plot(
            eps_s, means_s,
            marker="o", linestyle=styles[strat],
            color=colors[strat], linewidth=2,
            label=f"Strategy {strat}", markersize=7,
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="No degradation")
    ax.set_xlabel("EP Background Traffic (%)")
    ax.set_ylabel("Degradation Ratio (latency / latency@EP=0%)")
    ax.set_title("Figure 3c: Latency Degradation vs EP Background Traffic — All 3 Strategies")
    ax.set_xticks(EP_LEVELS)
    ax.set_xticklabels([f"{v}%" for v in EP_LEVELS])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate final points
    for strat in [1, 2, 3]:
        eps, means = strat_points[strat]
        if eps:
            ax.annotate(
                f"S{strat}: {means[-1]:.2f}x",
                xy=(eps[-1], means[-1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=colors[strat],
            )

    fig.tight_layout()

    for fmt in formats:
        out_path = output_dir / f"fig3c_degradation_ratio.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cross-node benchmark results.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to cross_node_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: same as CSV parent)",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=[],
        dest="formats",
        choices=["png", "pdf", "svg", "eps"],
        help="Output format(s); can be repeated (default: png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only validate CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        output_dir = args.csv.parent
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.csv)
    validate_csv(rows)

    if args.no_plot:
        print("[no-plot] CSV validated successfully.")
        return

    formats = args.formats if args.formats else ["png"]

    print("[plot] Generating Figure 3a …")
    plot_figure3a(rows, output_dir, formats)

    print("[plot] Generating Figure 3b …")
    plot_figure3b(rows, output_dir, formats)

    print("[plot] Generating Figure 3c …")
    plot_figure3c(rows, output_dir, formats)

    print(f"[done] All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
