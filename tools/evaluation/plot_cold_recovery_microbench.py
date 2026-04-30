#!/usr/bin/env python3
"""P2 Cold Recovery Microbenchmark plotting.

Reads microbench_cold_recovery.csv and microbench_payload_size.csv
from the sweep runner and produces paper-ready figures:

  - fig_cold_recovery_breakdown.pdf     (stacked component bars)
  - fig_cold_recovery_speedup_vs_rank.pdf
  - fig_cold_recovery_speedup_vs_batch.pdf
  - fig_cold_payload_size.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

POLICY_LABELS = {
    "cpu_first": "Execution-first",
    "load_then_run": "Promotion-first",
}
POLICY_COLORS = {
    "cpu_first": "#2A9D8F",
    "load_then_run": "#E76F51",
}

# Execution-first component colors (stacked bar)
EXEC_COMPONENT_COLORS = {
    "Tpack_us": "#264653",
    "TD2H_activation_us": "#2A9D8F",
    "Tcpu_us": "#E9C46A",
    "TH2D_residual_us": "#F4A261",
    "Tmerge_us": "#E76F51",
}
EXEC_COMPONENT_LABELS = {
    "Tpack_us": "Pack",
    "TD2H_activation_us": "D2H (activation)",
    "Tcpu_us": "CPU compute",
    "TH2D_residual_us": "H2D (residual)",
    "Tmerge_us": "Merge",
}

# Promotion-first component colors (stacked bar)
PROM_COMPONENT_COLORS = {
    "Tadmit_us": "#355070",
    "TH2D_weights_us": "#E76F51",
    "Tgpu_us": "#F4A261",
}
PROM_COMPONENT_LABELS = {
    "Tadmit_us": "Admission",
    "TH2D_weights_us": "H2D (weights)",
    "Tgpu_us": "GPU compute",
}

FIG_DPI = 150
FIG_WIDTH = 7.0
FIG_HEIGHT = 3.5


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _float_or(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Figure 1: Breakdown stacked bar
# ---------------------------------------------------------------------------

def plot_breakdown(
    agg_rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    """Multi-panel stacked bar: component breakdown at three representative configs.

    Shows: rank=8,batch=1 (small), rank=16,batch=8 (medium), rank=64,batch=8 (large).
    This covers the full regime of cold miss recovery behavior.
    """
    # Three representative configurations covering the range
    configs = [
        (8, 1, "rank=8, batch=1"),
        (16, 8, "rank=16, batch=8"),
        (64, 8, "rank=64, batch=8"),
    ]

    exec_components = list(EXEC_COMPONENT_COLORS.keys())
    prom_components = list(PROM_COMPONENT_COLORS.keys())

    # Build row lookup
    row_lookup = {}
    for row in agg_rows:
        key = (int(row.get("lora_rank", 0)), int(row.get("batch_size", 0)), row.get("policy", ""))
        row_lookup[key] = row

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH * 1.6, FIG_HEIGHT), sharey=True)

    legend_added = set()

    for ax_idx, (target_rank, target_batch, subtitle) in enumerate(configs):
        ax = axes[ax_idx]

        cpu_row = row_lookup.get((target_rank, target_batch, "cpu_first"))
        ltr_row = row_lookup.get((target_rank, target_batch, "load_then_run"))

        if cpu_row is None or ltr_row is None:
            ax.text(0.5, 0.5, f"No data\n{subtitle}", ha="center", va="center", transform=ax.transAxes)
            continue

        exec_values = [_float_or(cpu_row.get(f"{c}_median", "0")) for c in exec_components]
        prom_values = [_float_or(ltr_row.get(f"{c}_median", "0")) for c in prom_components]

        x = np.array([0, 1])
        bar_width = 0.6

        # Execution-first stacked bar
        bottom = 0.0
        for comp, val in zip(exec_components, exec_values):
            label = EXEC_COMPONENT_LABELS[comp] if comp not in legend_added else ""
            ax.bar(x[0], max(val, 0.0), bar_width, bottom=bottom,
                   color=EXEC_COMPONENT_COLORS[comp],
                   label=label,
                   edgecolor="white", linewidth=0.5)
            legend_added.add(comp)
            bottom += val

        # Promotion-first stacked bar
        bottom = 0.0
        for comp, val in zip(prom_components, prom_values):
            label = PROM_COMPONENT_LABELS[comp] if comp not in legend_added else ""
            ax.bar(x[1], max(val, 0.0), bar_width, bottom=bottom,
                   color=PROM_COMPONENT_COLORS[comp],
                   label=label,
                   edgecolor="white", linewidth=0.5)
            legend_added.add(comp)
            bottom += val

        ax.set_xticks(x)
        ax.set_xticklabels([
            POLICY_LABELS.get("cpu_first", "cpu_first"),
            POLICY_LABELS.get("load_then_run", "load_then_run"),
        ], fontsize=8)
        ax.set_title(subtitle, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Shared labels
    axes[0].set_ylabel("Single-miss recovery service time (μs)\n(lower is better)", fontsize=10)
    fig.suptitle("Isolated cold expert-LoRA miss recovery breakdown", fontsize=11, y=1.02)

    # Single legend outside the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="upper right",
               bbox_to_anchor=(0.98, 0.85), framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Speedup vs rank
# ---------------------------------------------------------------------------

def plot_speedup_vs_rank(
    agg_rows: List[Dict[str, str]],
    output_path: Path,
    fixed_batch: int = 4,
) -> None:
    """Line plot: speedup (promotion-first / execution-first) vs lora_rank.

    Uses median + Median Absolute Deviation (MAD) for robust error bars.
    MAD is converted to equivalent std (~1.4826 * MAD) for visualization.
    """
    MAD_TO_STD = 1.4826  # conversion factor for normal distribution

    by_config: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
    for row in agg_rows:
        rank = int(row.get("lora_rank", 0))
        batch = int(row.get("batch_size", 0))
        policy = row.get("policy", "")
        service_time = _float_or(row.get("service_time_us_median", "0"))
        service_mad = _float_or(row.get("service_time_us_mad", "0"))
        key = (rank, batch)
        if key not in by_config:
            by_config[key] = {}
        by_config[key][policy] = {"median": service_time, "mad": service_mad}

    ranks = sorted(set(k[0] for k in by_config if k[1] == fixed_batch and k in by_config))
    if not ranks:
        print(f"WARNING: no data for fixed_batch={fixed_batch}, skipping speedup-vs-rank")
        return

    speedups = []
    speedup_errors = []
    valid_ranks = []

    for rank in ranks:
        key = (rank, fixed_batch)
        times = by_config.get(key, {})
        ltr_data = times.get("load_then_run", {"median": 0.0, "mad": 0.0})
        cpu_data = times.get("cpu_first", {"median": 0.0, "mad": 0.0})
        ltr = ltr_data["median"]
        cpu = cpu_data["median"]

        if ltr > 0 and cpu > 0:
            speedup = ltr / cpu
            # Error propagation for ratio: err/speedup ≈ sqrt((err_ltr/ltr)^2 + (err_cpu/cpu)^2)
            ltr_err = ltr_data["mad"] * MAD_TO_STD
            cpu_err = cpu_data["mad"] * MAD_TO_STD
            rel_err = math.sqrt((ltr_err / ltr) ** 2 + (cpu_err / cpu) ** 2) if ltr > 0 and cpu > 0 else 0.0

            speedups.append(speedup)
            speedup_errors.append(speedup * rel_err)
            valid_ranks.append(rank)

    if not valid_ranks or not speedups or all(s <= 0 or math.isnan(s) for s in speedups):
        print(f"WARNING: no valid speedup values for batch={fixed_batch}, skipping speedup-vs-rank")
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.errorbar(valid_ranks, speedups, yerr=speedup_errors, fmt="o-",
                color="#2A9D8F", linewidth=2, markersize=6, capsize=3)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("LoRA rank", fontsize=10)
    ax.set_ylabel("Speedup (promotion-first / execution-first)\n(higher is better)", fontsize=10)
    ax.set_title(f"Cold recovery speedup vs rank (batch={fixed_batch}, median ± MAD)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_ranks)
    ax.set_xticklabels([str(r) for r in valid_ranks])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Speedup vs batch
# ---------------------------------------------------------------------------

def plot_speedup_vs_batch(
    agg_rows: List[Dict[str, str]],
    output_path: Path,
    fixed_rank: int = 16,
) -> None:
    """Line plot: speedup vs cold batch size with MAD error bars."""
    MAD_TO_STD = 1.4826

    by_config: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
    for row in agg_rows:
        rank = int(row.get("lora_rank", 0))
        batch = int(row.get("batch_size", 0))
        policy = row.get("policy", "")
        service_time = _float_or(row.get("service_time_us_median", "0"))
        service_mad = _float_or(row.get("service_time_us_mad", "0"))
        key = (rank, batch)
        if key not in by_config:
            by_config[key] = {}
        by_config[key][policy] = {"median": service_time, "mad": service_mad}

    batches = sorted(set(k[1] for k in by_config if k[0] == fixed_rank and k in by_config))
    if not batches:
        print(f"WARNING: no data for fixed_rank={fixed_rank}, skipping speedup-vs-batch")
        return

    speedups = []
    speedup_errors = []
    valid_batches = []

    for batch in batches:
        key = (fixed_rank, batch)
        times = by_config.get(key, {})
        ltr_data = times.get("load_then_run", {"median": 0.0, "mad": 0.0})
        cpu_data = times.get("cpu_first", {"median": 0.0, "mad": 0.0})
        ltr = ltr_data["median"]
        cpu = cpu_data["median"]

        if ltr > 0 and cpu > 0:
            speedup = ltr / cpu
            ltr_err = ltr_data["mad"] * MAD_TO_STD
            cpu_err = cpu_data["mad"] * MAD_TO_STD
            rel_err = math.sqrt((ltr_err / ltr) ** 2 + (cpu_err / cpu) ** 2) if ltr > 0 and cpu > 0 else 0.0

            speedups.append(speedup)
            speedup_errors.append(speedup * rel_err)
            valid_batches.append(batch)

    if not valid_batches or not speedups or all(s <= 0 or math.isnan(s) for s in speedups):
        print(f"WARNING: no valid speedup values for rank={fixed_rank}, skipping speedup-vs-batch")
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.errorbar(valid_batches, speedups, yerr=speedup_errors, fmt="o-",
                color="#E76F51", linewidth=2, markersize=6, capsize=3)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Cold batch size", fontsize=10)
    ax.set_ylabel("Speedup (promotion-first / execution-first)\n(higher is better)", fontsize=10)
    ax.set_title(f"Cold recovery speedup vs batch (rank={fixed_rank}, median ± MAD)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_batches)
    ax.set_xticklabels([str(b) for b in valid_batches])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Payload size comparison
# ---------------------------------------------------------------------------

def plot_payload_size(
    payload_rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    """Grouped bar: promotion weight bytes vs execution activation+residual bytes."""
    # Group by rank
    by_rank: Dict[int, Dict[str, float]] = {}
    for row in payload_rows:
        rank = int(row.get("lora_rank", 0))
        policy = row.get("policy", "")
        if rank not in by_rank:
            by_rank[rank] = {}
        if policy == "load_then_run":
            by_rank[rank]["weight_bytes"] = _float_or(row.get("promotion_weight_bytes", "0"))
        elif policy == "cpu_first":
            by_rank[rank]["act_bytes"] = _float_or(row.get("execution_activation_bytes", "0"))
            by_rank[rank]["res_bytes"] = _float_or(row.get("execution_residual_bytes", "0"))

    # Use first batch size for each rank (all batch sizes are present; pick batch=4)
    # Actually, payload_rows have entries per (policy, rank, batch). Filter to batch=4.
    by_rank_fixed: Dict[int, Dict[str, float]] = {}
    for row in payload_rows:
        batch = int(row.get("batch_size", 0))
        if batch != 4:
            continue
        rank = int(row.get("lora_rank", 0))
        policy = row.get("policy", "")
        if rank not in by_rank_fixed:
            by_rank_fixed[rank] = {}
        if policy == "load_then_run":
            by_rank_fixed[rank]["weight_bytes"] = _float_or(row.get("promotion_weight_bytes", "0"))
        elif policy == "cpu_first":
            by_rank_fixed[rank]["act_bytes"] = _float_or(row.get("execution_activation_bytes", "0"))
            by_rank_fixed[rank]["res_bytes"] = _float_or(row.get("execution_residual_bytes", "0"))

    ranks = sorted(by_rank_fixed.keys())
    if not ranks:
        # Fallback: use any batch
        ranks = sorted(by_rank.keys())
        data = by_rank
    else:
        data = by_rank_fixed

    weight_kb = [data[r].get("weight_bytes", 0) / 1024 for r in ranks]
    exec_kb = [(data[r].get("act_bytes", 0) + data[r].get("res_bytes", 0)) / 1024 for r in ranks]

    x = np.arange(len(ranks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.bar(x - width / 2, weight_kb, width, label="Promotion: full weights", color="#E76F51", edgecolor="white")
    ax.bar(x + width / 2, exec_kb, width, label="Execution: act. + residual", color="#2A9D8F", edgecolor="white")

    ax.set_xlabel("LoRA rank", fontsize=10)
    ax.set_ylabel("Payload size (KB)", fontsize=10)
    ax.set_title("Synchronous payload size comparison (batch=4)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Figure 5 (optional): Exposed time
# ---------------------------------------------------------------------------

def plot_exposed_time(
    agg_rows: List[Dict[str, str]],
    output_path: Path,
    fixed_rank: int = 16,
) -> None:
    """Grouped bar: isolated service time vs exposed time, by batch size."""
    by_config: Dict[Tuple[str, int, int], float] = {}
    for row in agg_rows:
        policy = row.get("policy", "")
        rank = int(row.get("lora_rank", 0))
        batch = int(row.get("batch_size", 0))
        exposed = _float_or(row.get("exposed_time_us_median", "0"))
        service = _float_or(row.get("service_time_us_median", "0"))
        key = (policy, rank, batch)
        by_config[key] = {"exposed": exposed, "service": service}

    batches = sorted(set(
        k[2] for k in by_config
        if k[1] == fixed_rank and k[0] == "cpu_first"
    ))
    if not batches:
        print("WARNING: no exposed time data, skipping exposed-time figure")
        return

    # Check if any exposed_time > 0 (if overlap benchmark wasn't run, skip)
    has_exposed = any(
        by_config.get(("cpu_first", fixed_rank, b), {}).get("exposed", 0) > 0
        for b in batches
    )
    if not has_exposed:
        print("WARNING: all exposed_time=0 (overlap benchmark not run), skipping figure")
        return

    service_values = [
        by_config.get(("cpu_first", fixed_rank, b), {}).get("service", 0)
        for b in batches
    ]
    exposed_values = [
        by_config.get(("cpu_first", fixed_rank, b), {}).get("exposed", 0)
        for b in batches
    ]

    x = np.arange(len(batches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.bar(x - width / 2, service_values, width,
           label="Isolated service time", color="#2A9D8F", edgecolor="white")
    ax.bar(x + width / 2, exposed_values, width,
           label="Exposed stall time (with overlap)", color="#E9C46A", edgecolor="white")

    ax.set_xlabel("Cold batch size", fontsize=10)
    ax.set_ylabel("Time (us)", fontsize=10)
    ax.set_title(f"Exposed stall vs isolated service time (rank={fixed_rank}, execution-first)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

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
        description="Plot P2 cold recovery microbenchmark figures"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/microbench_cold_recovery"),
        help="Directory containing microbench CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: same as input-dir)",
    )
    parser.add_argument("--fixed-batch", type=int, default=1)
    parser.add_argument("--fixed-rank", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    agg_csv = input_dir / "microbench_cold_recovery_aggregated.csv"
    payload_csv = input_dir / "microbench_payload_size.csv"

    agg_rows = _read_csv(agg_csv)
    payload_rows = _read_csv(payload_csv)

    plot_breakdown(agg_rows, output_dir / "fig_cold_recovery_breakdown.pdf")
    plot_speedup_vs_rank(agg_rows, output_dir / "fig_cold_recovery_speedup_vs_rank.pdf", fixed_batch=args.fixed_batch)
    plot_speedup_vs_batch(agg_rows, output_dir / "fig_cold_recovery_speedup_vs_batch.pdf", fixed_rank=args.fixed_rank)
    plot_payload_size(payload_rows, output_dir / "fig_cold_payload_size.pdf")
    plot_exposed_time(agg_rows, output_dir / "fig_cold_recovery_exposed_time.pdf", fixed_rank=args.fixed_rank)

    print(f"\nAll figures written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
