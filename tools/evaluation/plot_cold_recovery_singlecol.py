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
from matplotlib.patches import Rectangle


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
    [2.604, 1.987, 3.714, 1.325],   # batch=8
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
FIG_HEIGHT = 4.2  # More vertical space for panel (a) + heatmap


# ---------------------------------------------------------------------------
# Panel (a): Single stacked breakdown
# ---------------------------------------------------------------------------

def plot_single_breakdown(
    breakdown_data: dict,
    ax: plt.Axes,
    target_rank: int = 16,
    target_batch: int = 8,
) -> None:
    """Panel (a): Horizontal stacked-bar breakdown for one representative config."""
    exec_data = breakdown_data["execution_first"]
    prom_data = breakdown_data["promotion_first"]

    exec_segments = [
        ("other", exec_data["other"]),
        ("data_transfer", exec_data["data_transfer"]),
        ("cpu_compute", exec_data["cpu_compute"]),
    ]
    prom_segments = [
        ("other", prom_data["other"]),
        ("h2d_weights", prom_data["h2d_weights"]),
        ("gpu_compute", prom_data["gpu_compute"]),
    ]

    y_positions = [1, 0]
    bar_height = 0.5

    # Execution-first (bottom bar, y=0)
    left = 0.0
    for key, val in exec_segments:
        ax.barh(y_positions[1], max(val, 0.1), bar_height, left=left,
                color=EXEC_GROUP_COLORS[key],
                edgecolor="white", linewidth=0.5)
        left += val
    exec_total = sum(exec_data.values())

    # Promotion-first (top bar, y=1)
    left = 0.0
    for key, val in prom_segments:
        ax.barh(y_positions[0], max(val, 0.1), bar_height, left=left,
                color=PROM_GROUP_COLORS[key],
                edgecolor="white", linewidth=0.5)
        left += val
    prom_total = sum(prom_data.values())

    x_max = max(exec_total, prom_total) * 1.18  # More right space for total labels

    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Promotion-first", "Execution-first"], fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Service time ($\\mu$s)", fontsize=8, labelpad=8)
    ax.set_title(f"(a) Single-miss breakdown (rank={target_rank}, batch={target_batch})",
                 fontsize=9, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Total labels at bar ends - add μs suffix
    label_offset = x_max * 0.02
    ax.text(exec_total + label_offset, y_positions[1], f"{exec_total:.0f}",
            ha="left", va="center", fontsize=7, fontweight="bold")
    ax.text(prom_total + label_offset, y_positions[0], f"{prom_total:.0f}",
            ha="left", va="center", fontsize=7, fontweight="bold")

    # Direct labels inside major segments - check segment width first
    cpu_left = exec_data["other"] + exec_data["data_transfer"]
    if exec_data["cpu_compute"] > 30:
        ax.text(cpu_left + exec_data["cpu_compute"] / 2, y_positions[1],
                "CPU LoRA", ha="center", va="center", fontsize=6, color="white",
                fontweight="bold")

    h2d_left = prom_data["other"]
    if prom_data["h2d_weights"] > 30:
        ax.text(h2d_left + prom_data["h2d_weights"] / 2, y_positions[0],
                "H2D weights", ha="center", va="center", fontsize=6, color="white",
                fontweight="bold")

    gpu_left = prom_data["other"] + prom_data["h2d_weights"]
    if prom_data["gpu_compute"] > 30:
        ax.text(gpu_left + prom_data["gpu_compute"] / 2, y_positions[0],
                "GPU LoRA", ha="center", va="center", fontsize=6, color="black",
                fontweight="bold")

    # Data transfer label removed to prevent overlap

    # Manual legend below the bars - 2 columns for better fit
    legend_entries = [
        (Rectangle((0, 0), 1, 1, fc=EXEC_GROUP_COLORS["cpu_compute"],
         ec="white", lw=0.5), EXEC_GROUP_LABELS["cpu_compute"]),
        (Rectangle((0, 0), 1, 1, fc=EXEC_GROUP_COLORS["data_transfer"],
         ec="white", lw=0.5), EXEC_GROUP_LABELS["data_transfer"]),
        (Rectangle((0, 0), 1, 1, fc=EXEC_GROUP_COLORS["other"],
         ec="white", lw=0.5), EXEC_GROUP_LABELS["other"]),
        (Rectangle((0, 0), 1, 1, fc=PROM_GROUP_COLORS["h2d_weights"],
         ec="white", lw=0.5), PROM_GROUP_LABELS["h2d_weights"]),
        (Rectangle((0, 0), 1, 1, fc=PROM_GROUP_COLORS["gpu_compute"],
         ec="white", lw=0.5), PROM_GROUP_LABELS["gpu_compute"]),
        (Rectangle((0, 0), 1, 1, fc=PROM_GROUP_COLORS["other"],
         ec="white", lw=0.5), PROM_GROUP_LABELS["other"]),
    ]
    patches, labels = zip(*legend_entries)
    ax.legend(patches, labels, fontsize=5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.35), framealpha=0.9, columnspacing=1.0,
              handlelength=1.4, handleheight=0.9, borderaxespad=0.3)


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

    # Set ticks and labels with more padding
    ax.set_xticks(np.arange(len(RANKS)))
    ax.set_xticklabels([str(r) for r in RANKS], fontsize=8)
    ax.set_yticks(np.arange(len(BATCHES)))
    ax.set_yticklabels([str(b) for b in BATCHES], fontsize=8)
    ax.set_xlabel("LoRA rank", fontsize=9, labelpad=8)
    ax.set_ylabel("Batch size", fontsize=9, labelpad=8)
    ax.set_title(f"(b) Execution-first speedup (geomean {geomean:.2f}×)",
                 fontsize=9, loc="left", pad=10)

    # Annotate cells - always black text for readability in print
    for i in range(len(BATCHES)):
        for j in range(len(RANKS)):
            val = speedup_matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}×", ha="center", va="center",
                        fontsize=7, color="black")

    # Colorbar - simplified label with padding to prevent overlap
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Speedup (×)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)


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
    
    # Save both PDF (for paper) and PNG (for fast preview)
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    png_output_path = output_path.with_suffix('.png')
    fig.savefig(str(png_output_path), dpi=FIG_DPI, bbox_inches="tight")
    
    plt.close(fig)
    print(f"Wrote {output_path}")
    print(f"Wrote {png_output_path} (for preview)")


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
