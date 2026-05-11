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

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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
        "h2d_weights": 135.04,
        "gpu_compute": 175.675,
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
    "cpu_compute": "#358C7A",      # muted teal
    "data_transfer": "#2E86AB",    # muted blue
    "other": "#D9D9D9",            # muted gray: pack/merge
}
EXEC_GROUP_LABELS = {
    "cpu_compute": "CPU LoRA compute",
    "data_transfer": "Activation/Residual transfer",
    "other": "Pack/Merge",
}

# Promotion-first (GPU path): orange/red family
PROM_GROUP_COLORS = {
    "h2d_weights": "#D9654B",      # muted red
    "gpu_compute": "#E89F5C",      # muted orange
    "other": "#D9D9D9",            # muted gray: admission
}
PROM_GROUP_LABELS = {
    "h2d_weights": "H2D weights",
    "gpu_compute": "GPU LoRA compute",
    "other": "Admission",
}

FIG_DPI = 300
FIG_WIDTH = 3.5   # Single column width
FIG_HEIGHT = 4.25  # More vertical space for panel (a) + heatmap


# ---------------------------------------------------------------------------
# Panel (a): Single stacked breakdown
# ---------------------------------------------------------------------------

def _label_segment(
    ax: plt.Axes,
    left: float,
    width: float,
    y: float,
    text: str,
    *,
    min_width: float,
    fontsize: float = 6.2,
    color: str = "white",
    weight: str = "bold",
) -> None:
    """Place a label only if the segment is wide enough."""
    if width >= min_width:
        ax.text(
            left + width / 2,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            clip_on=True,
        )


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

    y_positions = [0.5, 0]
    bar_height = 0.25

    # Execution-first (bottom bar, y=0)
    left = 0.0
    for key, val in exec_segments:
        ax.barh(y_positions[1], max(val, 0.1), bar_height, left=left,
                color=EXEC_GROUP_COLORS[key],
                edgecolor="black", linewidth=0.5)
        left += val
    exec_total = sum(exec_data.values())

    # Promotion-first (top bar, y=1)
    left = 0.0
    for key, val in prom_segments:
        ax.barh(y_positions[0], max(val, 0.1), bar_height, left=left,
                color=PROM_GROUP_COLORS[key],
                edgecolor="black", linewidth=0.5)
        left += val
    prom_total = sum(prom_data.values())

    x_max = max(exec_total, prom_total) * 1.18  # More right space for total labels

    # Black border around each full bar
    ax.add_patch(Rectangle((0, y_positions[0] - bar_height / 2), prom_total, bar_height,
                           fill=False, edgecolor="black", linewidth=0.8, zorder=3))
    ax.add_patch(Rectangle((0, y_positions[1] - bar_height / 2), exec_total, bar_height,
                           fill=False, edgecolor="black", linewidth=0.8, zorder=3))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Promotion\nfirst", "Execution\nfirst"], fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Recovery time ($\\mu$s)", fontsize=8, labelpad=5)
    ax.set_title("(a) Single-miss recovery time", fontsize=9, loc="left", pad=8)
    ax.tick_params(axis="both", length=3, width=0.8, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Total labels at bar ends
    label_offset = x_max * 0.02
    ax.text(exec_total + label_offset, y_positions[1], f"{exec_total:.0f} μs",
            ha="left", va="center", fontsize=7, fontweight="bold")
    ax.text(prom_total + label_offset, y_positions[0], f"{prom_total:.0f} μs",
            ha="left", va="center", fontsize=7, fontweight="bold")

    # Direct labels inside major segments. Keep labels short to avoid overlap.

    # Promotion-first labels
    h2d_left = prom_data["other"]
    _label_segment(
        ax, h2d_left, prom_data["h2d_weights"], y_positions[0],
        "H2D", min_width=35, fontsize=6.4, color="white"
    )

    gpu_left = prom_data["other"] + prom_data["h2d_weights"]
    _label_segment(
        ax, gpu_left, prom_data["gpu_compute"], y_positions[0],
        "GPU LoRA", min_width=55, fontsize=6.8, color="black"
    )

    # Execution-first labels
    io_left = exec_data["other"]
    _label_segment(
        ax, io_left, exec_data["data_transfer"], y_positions[1],
        "I/O", min_width=35, fontsize=6.2, color="white"
    )

    cpu_left = exec_data["other"] + exec_data["data_transfer"]
    _label_segment(
        ax, cpu_left, exec_data["cpu_compute"], y_positions[1],
        "CPU\nLoRA", min_width=40, fontsize=6.4, color="white"
    )


# ---------------------------------------------------------------------------
# Panel (b): Speedup heatmap
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(
    speedup_matrix: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Panel (b): Compact heatmap of speedup = promotion-first / execution-first."""

    im = ax.imshow(
        speedup_matrix,
        cmap="YlGn",
        vmin=1.0,
        vmax=5.5,
        aspect="auto",
        origin="upper",
    )

    ax.set_xticks(np.arange(len(RANKS)))
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_yticks(np.arange(len(BATCHES)))
    ax.set_yticklabels([str(b) for b in BATCHES])

    ax.set_xlabel("LoRA rank", fontsize=8, labelpad=5)
    ax.set_ylabel("Decode batch size", fontsize=8, labelpad=5)
    ax.set_title("(b) Execution-first speedup", fontsize=9, loc="left", pad=8)

    # Thin white separators improve readability in print.
    ax.set_xticks(np.arange(-0.5, len(RANKS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BATCHES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=3, width=0.8, pad=2)

    # Annotate cells with adaptive text color.
    for i in range(len(BATCHES)):
        for j in range(len(RANKS)):
            val = speedup_matrix[i, j]
            rgba = im.cmap(im.norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if lum < 0.45 else "black"
            ax.text(
                j, i, f"{val:.1f}×",
                ha="center", va="center",
                fontsize=7.5,
                color=txt_color,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def plot_singlecol_figure(
    output_path: Path,
    use_stress: bool = False,
) -> None:
    """Build the full two-panel one-column figure."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

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

    fig.subplots_adjust(
        left=0.34,
        right=0.98,
        top=0.96,
        bottom=0.11,
        hspace=0.58,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.01)
    png_output_path = output_path.with_suffix(".png")
    fig.savefig(str(png_output_path), dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.01)

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
