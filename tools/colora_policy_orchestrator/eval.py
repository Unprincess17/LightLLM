"""Deterministic eval harness for the CoLoRA hybrid recovery policy.

Reads measured CSVs (no live hardware) and computes the mean foreground
recovery latency (microseconds) under the policy's path choice across the
full (rank, n_tokens, ep_bw_pct) grid.

Verify command (run from repo root):
    python tools/colora_policy_orchestrator/eval.py

The script prints exactly one number on the last stdout line: the mean
recovery latency in microseconds (lower is better).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSS_NODE_CSV = REPO_ROOT / "results/cross_node_benchmark/cross_node_benchmark.csv"
CROSSOVER_CSV = REPO_ROOT / "results/crossover_curve/crossover_results.csv"

# Make the policy module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy  # type: ignore  # noqa: E402

# Strategy IDs in the cross-node CSV.
S1_REMOTE_WEIGHT = 1
S2_REMOTE_ACTIVATION = 2
S3_REMOTE_RELAY = 3
LOCAL_CPU = 10  # synthetic id used by the policy
LOCAL_GPU = 11  # synthetic id used by the policy

# Map seq_len used for the local crossover lookup. The deck calibrates against
# seq_len=2048 (typical decode), so we anchor the local path there.
LOCAL_SEQ_LEN = 2048


def _to_us(ms_str: str) -> float:
    return float(ms_str) * 1000.0


def load_cross_node_grid() -> dict:
    """Returns grid[(rank, n_tokens, ep)][strategy_id] = total_us."""
    grid: dict = {}
    with CROSS_NODE_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["rank"])
            n_tokens = int(row["num_miss"])
            ep = int(row["ep_bw_pct"])
            strat = int(row["strategy"])
            total_ms = float(row["total_ms"])
            cell = grid.setdefault((rank, n_tokens, ep), {})
            cell[strat] = total_ms * 1000.0
    return grid


def load_local_table() -> dict:
    """Returns local[(rank, n_tokens)] = {LOCAL_CPU: us, LOCAL_GPU: us}.

    LOCAL_CPU corresponds to t_cpu_first_us (CPU-first foreground path).
    LOCAL_GPU corresponds to t_cold_miss_us (GPU gather + compute baseline).
    """
    table: dict = {}
    with CROSSOVER_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["seq_len"]) != LOCAL_SEQ_LEN:
                continue
            rank = int(row["lora_rank"])
            n_tokens = int(row["n_tokens"])
            table[(rank, n_tokens)] = {
                LOCAL_CPU: float(row["t_cpu_first_us"]),
                LOCAL_GPU: float(row["t_cold_miss_us"]),
            }
    return table


def measured_us(cell_key, choice, grid, local_table) -> float | None:
    rank, n_tokens, ep = cell_key
    if choice in (S1_REMOTE_WEIGHT, S2_REMOTE_ACTIVATION, S3_REMOTE_RELAY):
        return grid.get(cell_key, {}).get(choice)
    if choice in (LOCAL_CPU, LOCAL_GPU):
        # Local recovery is independent of EP load (it doesn't traverse the
        # fabric), so we look up the baseline-rank/n_tokens micro number.
        # If the rank or n_tokens is outside the local sweep, return None and
        # the eval will treat it as an invalid choice.
        return local_table.get((rank, n_tokens), {}).get(choice)
    return None


def main() -> int:
    grid = load_cross_node_grid()
    local_table = load_local_table()

    grid_cells = sorted(grid.keys())
    if not grid_cells:
        print("FATAL: empty cross-node grid", file=sys.stderr)
        return 2

    total_us = 0.0
    counted = 0
    invalid = 0
    breakdown = []
    for cell in grid_cells:
        rank, n_tokens, ep = cell
        ctx = policy.RecoveryContext(rank=rank, n_tokens=n_tokens, ep_bw_pct=ep)
        choice = policy.choose_path(ctx)
        us = measured_us(cell, choice, grid, local_table)
        if us is None:
            # Penalty for invalid choices: use the worst measured strategy.
            cell_us = max(grid[cell].values())
            us = cell_us
            invalid += 1
        total_us += us
        counted += 1
        breakdown.append((cell, choice, us))

    mean_us = total_us / counted

    if os.environ.get("COLORA_VERBOSE") == "1":
        print(f"# cells={counted} invalid={invalid}", file=sys.stderr)
        for (cell, choice, us) in breakdown[:6]:
            print(f"# {cell} choice={choice} us={us:.1f}", file=sys.stderr)

    # Last stdout line MUST be the metric (μs).
    print(f"{mean_us:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
