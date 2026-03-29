#!/usr/bin/env python3
"""
Temporal Locality of MoE Expert Routing During Decode
=====================================================

Microbenchmark script for an ASPLOS-style figure showing how much expert-set
overlap ("hit rate") exists between consecutive decode tokens at each MoE layer.

Metric
------
For each sequence, for each layer L, iterate through decoded tokens in order.
Compare expert set E_t with the previous token's expert set E_{t-1}:

    Hit_Rate = |intersection(E_t, E_{t-1})| / |E_t|

The figure plots the average hit rate (and 5th-95th percentile band) across
all valid token transitions and all sequences, grouped by layer_index.

Usage
-----
# Test immediately with synthetic data:
    python tools/motivation/plot_temporal_locality.py --use_mock

# Run on real router trace collected via B5 (collect_router_trace.py):
    python tools/motivation/plot_temporal_locality.py \
        --trace_path artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl

# Customize output path:
    python tools/motivation/plot_temporal_locality.py \
        --trace_path /path/to/router_trace.jsonl \
        --output /path/to/output.pdf

Data Format
-----------
The script expects JSONL where each line is a dict with at least:
    - req_idx        (int):  sequence / request identifier
    - layer_id       (int):  MoE layer index
    - token_pos      (int):  token position within the sequence
    - phase          (str):  "prefill" or "decode"
    - topk_experts   (list): list of selected expert IDs, e.g. [3,7,9,12,15,18,21,24]

This matches the canonical schema produced by:
    tools/case_study/collect_router_trace.py

To plug in your own data, either:
  (a) Write a JSONL file conforming to this schema, or
  (b) Replace load_real_data() with your own loader that returns
      List[dict] with the same keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRACE_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "case_study"
    / "router_lora_case_v1"
    / "router_trace"
    / "router_trace.jsonl"
)
_DEFAULT_OUTPUT_PATH = Path("temporal_locality_hit_rate.pdf")

# ---------------------------------------------------------------------------
# Model constants (Qwen3-VL-30B-A3B defaults)
# ---------------------------------------------------------------------------
_NUM_EXPERTS = 128     # total experts in the MoE model
_TOP_K = 8             # experts selected per token
_NUM_LAYERS = 48       # MoE layers in the model

# ---------------------------------------------------------------------------
# Paper color palette (matching assemble_paper_figures.py)
# ---------------------------------------------------------------------------
_COLOR_PRIMARY = "#355070"


# ===================================================================
# Task 1: Data Loading
# ===================================================================

def generate_mock_data(
    num_sequences: int = 5,
    tokens_per_sequence: int = 100,
    num_layers: int = _NUM_LAYERS,
    num_experts: int = _NUM_EXPERTS,
    top_k: int = _TOP_K,
    seed: int = 42,
) -> List[dict]:
    """Generate synthetic decode-phase router traces for testing.

    Creates data with realistic temporal locality: each token reuses ~60-70%
    of the previous token's experts, with the rest drawn randomly.  This
    produces a mock hit rate around 0.60-0.70 so you can visually verify the
    plot before plugging in real data.

    Returns
    -------
    List[dict]
        Records matching the canonical router_trace.jsonl schema.
    """
    rng = np.random.default_rng(seed)
    records: List[dict] = []

    for seq_id in range(num_sequences):
        for layer_id in range(num_layers):
            # Draw the first token's experts uniformly at random.
            prev_experts = rng.choice(num_experts, size=top_k, replace=False).tolist()
            records.append({
                "req_idx": seq_id,
                "layer_id": layer_id,
                "token_pos": 0,
                "phase": "decode",
                "topk_experts": list(prev_experts),
            })

            for token_pos in range(1, tokens_per_sequence):
                # Decide how many experts to keep from the previous step.
                # Draw from a binomial to create variance; mean ~5 out of 8
                # gives ~62.5% baseline hit rate.
                keep_count = int(rng.binomial(top_k, 0.625))
                keep_count = min(keep_count, top_k)

                # Randomly select which previous experts to keep.
                kept = rng.choice(prev_experts, size=keep_count, replace=False).tolist()
                kept_set = set(kept)

                # Fill the remaining slots from experts NOT already kept.
                remaining_pool = [e for e in range(num_experts) if e not in kept_set]
                new_count = top_k - keep_count
                new_experts = rng.choice(remaining_pool, size=new_count, replace=False).tolist()

                current_experts = kept + new_experts
                rng.shuffle(current_experts)

                records.append({
                    "req_idx": seq_id,
                    "layer_id": layer_id,
                    "token_pos": token_pos,
                    "phase": "decode",
                    "topk_experts": list(current_experts),
                })
                prev_experts = current_experts

    print(f"[mock] generated {len(records)} decode events "
          f"({num_sequences} seqs x {num_layers} layers x {tokens_per_sequence} tokens)")
    return records


def load_real_data(trace_path: str | Path) -> List[dict]:
    """Load decode-phase router trace events from a canonical JSONL file.

    Filters to phase=="decode" only and retains the fields needed for
    hit-rate computation: req_idx, layer_id, token_pos, topk_experts.

    Parameters
    ----------
    trace_path : str or Path
        Path to the router_trace.jsonl file produced by
        tools/case_study/collect_router_trace.py.

    Returns
    -------
    List[dict]
        Filtered decode-phase records sorted by (req_idx, layer_id, token_pos).
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"router trace not found: {trace_path}")

    records: List[dict] = []
    skipped = 0
    with trace_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                rec = json.loads(raw_line)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON at line {line_num}")
                continue

            # Only keep decode-phase events.
            if rec.get("phase") != "decode":
                skipped += 1
                continue

            records.append({
                "req_idx": int(rec["req_idx"]),
                "layer_id": int(rec["layer_id"]),
                "token_pos": int(rec["token_pos"]),
                "phase": "decode",
                "topk_experts": [int(e) for e in rec["topk_experts"]],
            })

    # Sort to guarantee sequential token ordering per (sequence, layer).
    records.sort(key=lambda r: (r["req_idx"], r["layer_id"], r["token_pos"]))

    print(f"[data] loaded {len(records)} decode events from {trace_path} "
          f"(skipped {skipped} non-decode events)")
    return records


# ===================================================================
# Task 2: Hit Rate Computation
# ===================================================================

def compute_hit_rates(records: List[dict]) -> pd.DataFrame:
    """Compute per-layer temporal locality hit rates.

    For each (sequence, layer) pair, iterates through consecutive decode
    tokens and computes:

        hit_rate_t = |E_t ∩ E_{t-1}| / |E_t|

    Then aggregates across all sequences to produce per-layer statistics.

    Parameters
    ----------
    records : List[dict]
        Decode-phase records with keys: req_idx, layer_id, token_pos,
        topk_experts. Must be sorted by (req_idx, layer_id, token_pos).

    Returns
    -------
    pd.DataFrame
        Columns: layer_id, mean_hit_rate, std, p5, p95, num_transitions
    """
    # Group records by (req_idx, layer_id).
    groups: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for rec in records:
        key = (rec["req_idx"], rec["layer_id"])
        groups[key].append(rec)

    # Collect per-transition hit rates, bucketed by layer.
    layer_hit_rates: Dict[int, List[float]] = defaultdict(list)

    for (req_idx, layer_id), group in groups.items():
        # Already sorted by token_pos from load step; enforce here for safety.
        group.sort(key=lambda r: r["token_pos"])

        for i in range(1, len(group)):
            prev_set = set(group[i - 1]["topk_experts"])
            curr_set = set(group[i]["topk_experts"])
            curr_list = group[i]["topk_experts"]

            # Hit rate: fraction of current token's experts that were also
            # selected in the previous token.
            if len(curr_list) == 0:
                continue
            hit_rate = len(curr_set & prev_set) / len(curr_list)
            layer_hit_rates[layer_id].append(hit_rate)

    # Aggregate per layer.
    rows = []
    for layer_id in sorted(layer_hit_rates.keys()):
        rates = np.array(layer_hit_rates[layer_id])
        rows.append({
            "layer_id": layer_id,
            "mean_hit_rate": float(np.mean(rates)),
            "std": float(np.std(rates)),
            "p5": float(np.percentile(rates, 5)),
            "p95": float(np.percentile(rates, 95)),
            "num_transitions": len(rates),
        })

    df = pd.DataFrame(rows)

    # Print summary statistics.
    if not df.empty:
        global_mean = df["mean_hit_rate"].mean()
        global_min = df["mean_hit_rate"].min()
        global_max = df["mean_hit_rate"].max()
        total_transitions = df["num_transitions"].sum()
        print(f"[stats] {len(df)} layers, {total_transitions} total transitions")
        print(f"[stats] global mean hit rate: {global_mean:.4f} "
              f"(min layer: {global_min:.4f}, max layer: {global_max:.4f})")

    return df


# ===================================================================
# Task 3: ASPLOS-Quality Plot
# ===================================================================

def apply_paper_style() -> None:
    """Configure matplotlib for clean, publication-ready figures.

    Matches the style conventions from assemble_paper_figures.py to keep
    all case-study figures visually consistent.
    """
    plt.rcParams.update({
        # Figure defaults
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Spine visibility
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        # Font sizes (ASPLOS legibility requirements)
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Font family
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    })


def plot_hit_rate(df: pd.DataFrame, output_path: str | Path) -> None:
    """Generate the temporal locality hit-rate figure.

    Plots Layer Index (X) vs. average routing hit rate (%) as a
    single mean trend line.

    Parameters
    ----------
    df : pd.DataFrame
        Output of compute_hit_rates() with columns:
        layer_id, mean_hit_rate, std, p5, p95, num_transitions.
    output_path : str or Path
        Where to save the PDF figure.
    """
    apply_paper_style()

    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    x = df["layer_id"].values
    y_mean = df["mean_hit_rate"].values * 100.0   # convert to %

    # Mean hit-rate line with markers.
    ax.plot(
        x, y_mean,
        color=_COLOR_PRIMARY,
        linewidth=2.0,
        marker="o",
        markersize=4,
        markerfacecolor=_COLOR_PRIMARY,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )

    # Axis formatting.
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Routing Hit Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    # X-tick spacing: show every 4th layer for readability with 48 layers.
    tick_step = max(1, len(x) // 12)
    ax.set_xticks(x[::tick_step])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved figure to {output_path}")


# ===================================================================
# CLI Entry Point
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot temporal locality (expert-set hit rate) "
            "of MoE routing during the LLM decode phase."
        ),
    )
    parser.add_argument(
        "--trace_path",
        type=str,
        default=str(_DEFAULT_TRACE_PATH),
        help=(
            "Path to router_trace.jsonl produced by "
            "tools/case_study/collect_router_trace.py. "
            f"Default: {_DEFAULT_TRACE_PATH}"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT_PATH),
        help=f"Output PDF path. Default: {_DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use synthetic mock data instead of a real trace file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load data ---
    if args.use_mock:
        records = generate_mock_data()
    else:
        records = load_real_data(args.trace_path)

    if not records:
        print("[error] no decode events found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Compute hit rates ---
    df = compute_hit_rates(records)

    if df.empty:
        print("[error] no hit-rate data computed (need >=2 tokens per "
              "sequence-layer group). Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Generate figure ---
    plot_hit_rate(df, args.output)

    # --- Save raw stats as CSV alongside the figure ---
    csv_path = Path(args.output).with_suffix(".csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[stats] saved per-layer statistics to {csv_path}")


if __name__ == "__main__":
    main()
