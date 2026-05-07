#!/usr/bin/env python3
"""P2 Cold Recovery Microbenchmark plotting - single column layout.

Produces a two-panel one-column figure:
  - (a) Stacked component breakdown for one representative config
  - (b) Compact speedup heatmap across all (rank x batch) configs

Final computed data is embedded directly - no intermediate CSV processing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Final computed data (pre-computed from benchmark results)
# ---------------------------------------------------------------------------

# Speedup ratios = promotion-first / execution-first
# Rows: batch = [1, 2, 4, 8] (index 0 = batch=1, index 3 = batch=8)
# Columns: rank = [8, 16, 32, 64] (index 0 = rank=8, index 3 = rank=64)
SPEEDUP_MATRIX = np.array([
    [1.222, 2.122, 2.669, 5.240],   # batch=1
    [1.738, 2.622, 2.838, 4.620],   # batch=2
    [2.004, 2.175, 3.174, 2.196],   # batch=4
    [2.604, 2.388, 3.714, 1.325],   # batch=8
])

# Axes labels for heatmap
RANKS = [8, 16, 32, 64]
BATCHES = [1, 2, 4, 8]

# Stacked breakdown latencies (μs) for rank=16, batch=8 (representative config)
BREAKDOWN_DATA = {
    "execution_first": {
        "cpu_compute": 75.978,
        "data_transfer": 64.889,  # TD2H_activation + TH2D_residual
        "other": 20.538,          # Tpack + Tmerge
    },
    "promotion_first": {
        "h2d_weights": 75.04,
        "gpu_compute": 235.675,
        "other": 10.0,             # Tadmit
    },
}

# Alternative: stress case (rank=64, batch=8)
BREAKDOWN_STRESS = {
    "execution_first": {
        "cpu_compute": 2375.479,
        "data_transfer": 131.210,
        "other": 41.424,
    },
    "promotion_first": {
        "h2d_weights": 6009.921,
        "gpu_compute": 2862.005,
        "other": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

POLICY_LABELS = {
    "cpu_first": "Execution-first",
    "load_then_run": "Promotion-first",
}

# Grouped component colors for simplified legend
# Execution-first (CPU path): blue/green family
EXEC_GROUP_COLORS = {
    "cpu_compute": "#2A9D8F",      # Teal/green for compute
    "data_transfer": "#219EBC",    # Blue for transfer
    "other": "#023047",            # Dark blue for overhead
}
EXEC_GROUP_LABELS = {
    "cpu_compute": "CPU LoRA compute",
    "data_transfer": "Activation/Residual transfer",
    "other": "Pack/Merge",
}

# Promotion-first (GPU path): orange/red family
PROM_GROUP_COLORS = {
    "h2d_weights": "#E76F51",      # Red for transfer
    "gpu_compute": "#F4A261",      # Orange for compute
    "other": "#6D6875",            # Gray for overhead
}
PROM_GROUP_LABELS = {
    "h2d_weights": "H2D weights",
    "gpu_compute": "GPU LoRA compute",
    "other": "Admission",
}

FIG_DPI = 150
FIG_WIDTH = 3.5   # Single column width
FIG_HEIGHT = 5.0  # Two panels stacked vertically


# ---------------------------------------------------------------------------
# Panel (a): Single stacked breakdown
# ---------------------------------------------------------------------------

def plot_single_breakdown(
    breakdown_data: dict,
    ax: plt.Axes,
    target_rank: int = 16,
    target_batch: int = 8,
) -> None:
    """Panel (a): Stacked component breakdown for one representative config."""
    exec_data = breakdown_data["execution_first"]
    prom_data = breakdown_data["promotion_first"]

    exec_groups = [
        ("cpu_compute", exec_data["cpu_compute"]),
        ("data_transfer", exec_data["data_transfer"]),
        ("other", exec_data["other"]),
    ]
    prom_groups = [
        ("h2d_weights", prom_data["h2d_weights"]),
        ("gpu_compute", prom_data["gpu_compute"]),
        ("other", prom_data["other"]),
    ]

    x = np.array([0, 1])
    bar_width = 0.6

    # Execution-first stacked bar
    legend_added = set()
    bottom = 0.0
    for key, val in exec_groups:
        label = EXEC_GROUP_LABELS[key] if key not in legend_added else ""
        ax.bar(x[0], max(val, 0.0), bar_width, bottom=bottom,
               color=EXEC_GROUP_COLORS[key],
               label=label,
               edgecolor="white", linewidth=0.5)
        legend_added.add(key)
        bottom += val

    # Promotion-first stacked bar
    bottom = 0.0
    for key, val in prom_groups:
        label = PROM_GROUP_LABELS[key] if key not in legend_added else ""
        ax.bar(x[1], max(val, 0.0), bar_width, bottom=bottom,
               color=PROM_GROUP_COLORS[key],
               label=label,
               edgecolor="white", linewidth=0.5)
        legend_added.add(key)
        bottom += val

    # Total time annotations on top of each bar
    exec_total = sum(exec_data.values())
    prom_total = sum(prom_data.values())
    speedup = prom_total / exec_total

    # Dynamic ylim - leave 25% extra space for annotations
    y_max = max(exec_total, prom_total) * 1.25

    ax.set_xticks(x)
    ax.set_xticklabels([
        POLICY_LABELS["cpu_first"],
        POLICY_LABELS["load_then_run"],
    ], fontsize=9)
    ax.set_ylabel("Recovery service time (μs)", fontsize=9)
    ax.set_ylim(0, y_max)
    ax.set_title(f"(a) breakdown (rank={target_rank}, batch={target_batch})",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)

    # Bar height annotation offset
    offset = y_max * 0.02

    ax.text(x[0], exec_total + offset, f"{exec_total:.0f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(x[1], prom_total + offset, f"{prom_total:.0f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Speedup annotation with arrow
    anno_y = max(exec_total, prom_total) + y_max * 0.08
    ax.annotate(f"{speedup:.1f}×",
                xy=(x[1], anno_y),
                xytext=(x[0], anno_y),
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))

    # Legend - compact layout
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, loc="upper left", framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel (b): Speedup heatmap
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(
    speedup_matrix: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Panel (b): Compact heatmap of speedup = promotion-first / execution-first."""
    # Compute geometric mean
    geomean = np.exp(np.mean(np.log(speedup_matrix.flatten())))

    # Plot heatmap - sequential colormap (darker = better speedup)
    im = ax.imshow(speedup_matrix, cmap="YlGn", vmin=1.0, vmax=5.5,
                   aspect="auto", origin="upper")  # origin=upper: batch=1 at top

    # Set ticks and labels
    ax.set_xticks(np.arange(len(RANKS)))
    ax.set_xticklabels([str(r) for r in RANKS], fontsize=8)
    ax.set_yticks(np.arange(len(BATCHES)))
    ax.set_yticklabels([str(b) for b in BATCHES], fontsize=8)
    ax.set_xlabel("LoRA rank", fontsize=9)
    ax.set_ylabel("Batch size", fontsize=9)
    ax.set_title(f"(b) Full sweep speedup (geomean {geomean:.2f}×)",
                 fontsize=10, loc="left")

    # Annotate cells - always black text for readability in print
    for i in range(len(BATCHES)):
        for j in range(len(RANKS)):
            val = speedup_matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}×", ha="center", va="center",
                        fontsize=7, color="black")

    # Colorbar - simplified label
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Speedup (×)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def plot_singlecol_figure(
    output_path: Path,
    use_stress: bool = False,
) -> None:
    """Build the full two-panel one-column figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Select breakdown config
    if use_stress:
        breakdown_data = BREAKDOWN_STRESS
        target_rank = 64
        target_batch = 8
    else:
        breakdown_data = BREAKDOWN_DATA
        target_rank = 16
        target_batch = 8

    # Panel (a): stacked breakdown
    plot_single_breakdown(breakdown_data, ax1, target_rank, target_batch)

    # Panel (b): speedup heatmap
    plot_speedup_heatmap(SPEEDUP_MATRIX, ax2)

    # Overall figure title
    # fig.suptitle("Single-miss recovery service time", fontsize=11, y=0.99)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot single-column cold recovery figure"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/microbench_cold_recovery"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Use stress case (rank=64, batch=8) instead of representative",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir

    plot_singlecol_figure(
        output_dir / "fig_cold_recovery_singlecol.pdf",
        use_stress=args.stress,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
