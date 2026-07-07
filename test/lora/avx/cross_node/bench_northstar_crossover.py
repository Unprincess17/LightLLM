"""N1: Isolated live crossover — intrinsic service-time boundary.

Sweeps full R x NM grid at isolated single-request conditions.
Five paths: cpu_first, load_then_run, remote_improved, oracle, remote_original (ablation).

Primary endpoint: paired difference in trial-level median L_recovery.
Forced-cold invariant enforced. Path order randomized within paired blocks.
"""
import argparse
import csv
import os
import statistics
import sys
import time

import torch

from common.northstar_paths import (
    cpu_first_recovery, load_then_run_recovery, oracle_recovery,
    init_weights, format_weights_for_path
)
from common.forced_cold import ForcedColdWeightPool
from common.path_randomization import randomize_path_order
from common.instrumentation import NorthstarTimeline
from common.stats import classify_pair, paired_diff_ci, trial_ci, holm_correct

# --- Configuration ---

N1_CONFIG = {
    "ranks": [16, 32, 64, 128, 256],
    "nms": [1, 2, 4, 8, 16],
    "n_trials": 5,
    "n_requests_per_trial": 1000,  # >= 1000; 2000 for P99 claims
    "H": 2048,
    "I": 2048,
    "dtype_weight": torch.bfloat16,
    "dtype_act": torch.float16,
    # CPU sensitivity
    "cpu_sensitivity_points": [(16,1), (64,1), (64,8), (128,4), (256,1), (256,8)],
    "cpu_thread_counts": [1, 2, 4],
    # remote_original ablation anchors
    "remote_original_anchors": [(16,1), (64,8), (256,8)],
}

PATHS_LOCAL = {"cpu_first": cpu_first_recovery, "load_then_run": load_then_run_recovery}
PATH_REMOTE_IMPROVED = "remote_improved"
PATH_ORACLE = "oracle"
PATH_REMOTE_ORIGINAL = "remote_original"
ALL_PATHS = ["cpu_first", "load_then_run", "remote_improved", "oracle"]

# TOST margin: delta_{R,NM} = max(50us, 0.10 * calibration_median)
DELTA_ABSOLUTE_US = 50.0
DELTA_RHO = 0.10


def build_cell_grid():
    """Build the full R x NM grid (25 cells)."""
    cells = []
    for R in N1_CONFIG["ranks"]:
        for NM in N1_CONFIG["nms"]:
            cells.append((R, NM))
    return cells


def run_single_request(path_name, activation, weights, R, H, I, NM,
                       num_cores=1, remote_session=None):
    """Run a single isolated recovery request on the specified path.

    Args:
        path_name: one of "cpu_first", "load_then_run", "remote_improved", "oracle"
        activation: [NM, H] FP16 on client GPU
        weights: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 (CPU for local paths, GPU for oracle)
        R, H, I, NM: dimensions
        num_cores: CPU cores for cpu_first
        remote_session: RemoteSession for remote paths (None for local/oracle)

    Returns: (result_gpu [NM,I] FP16, NorthstarTimeline)
    """
    consumer_stream = torch.cuda.current_stream()

    if path_name == "cpu_first":
        return cpu_first_recovery(activation, weights, R, H, I, NM,
                                   num_cores=num_cores, consumer_stream=consumer_stream)
    elif path_name == "load_then_run":
        return load_then_run_recovery(activation, weights, R, H, I, NM,
                                       consumer_stream=consumer_stream)
    elif path_name == "oracle":
        gpu_weights = format_weights_for_path(weights, "oracle", device="cuda")
        return oracle_recovery(activation, gpu_weights, R, H, I, NM,
                                consumer_stream=consumer_stream)
    elif path_name == "remote_improved":
        if remote_session is None:
            raise ValueError("remote_improved requires remote_session")
        # TODO: integrate with RemoteSession
        return remote_session.run_single(activation, weights, R, H, I, NM)
    else:
        raise ValueError(f"Unknown path: {path_name}")


def run_trial(path_name, R, H, I, NM, n_requests, pool, pool_offset,
              num_cores=1, remote_session=None):
    """Run one trial: n_requests isolated requests on one path.

    Returns: list of L_recovery_us values (one per request)
    """
    latencies = []
    for i in range(n_requests):
        idx = pool_offset + i
        if path_name == "oracle":
            weights = pool.get_oracle_batch(idx, NM, device="cuda")
            activation = torch.randn(NM, H, dtype=N1_CONFIG["dtype_act"], device="cuda")
        else:
            weights = pool.get_batch(idx, NM)
            activation = torch.randn(NM, H, dtype=N1_CONFIG["dtype_act"], device="cuda")

        _, timeline = run_single_request(
            path_name, activation, weights, R, H, I, NM,
            num_cores=num_cores, remote_session=remote_session
        )
        latencies.append(timeline.l_recovery_us())
    return latencies


def run_n1(output_dir, n_trials=None, n_requests=None, remote_server_host=None):
    """Run the full N1 campaign.

    For each (R, NM) cell, run all 4 paths x n_trials x n_requests.
    Path order randomized within each trial (paired block).
    """
    n_trials = n_trials or N1_CONFIG["n_trials"]
    n_requests = n_requests or N1_CONFIG["n_requests_per_trial"]
    H = N1_CONFIG["H"]
    I_dim = N1_CONFIG["I"]
    cells = build_cell_grid()

    # Pool size must exceed n_requests per trial
    pool_size = n_requests * 2  # safety margin

    # Start remote session if server host provided
    remote_session = None
    if remote_server_host:
        from bench_decomposition import RemoteSession
        remote_session = RemoteSession(remote_server_host, cell="B3")
        remote_session.__enter__()

    try:
        results = []
        for trial_idx in range(n_trials):
            path_order = randomize_path_order(ALL_PATHS, seed=42 + trial_idx)

            for R in N1_CONFIG["ranks"]:
                for NM in N1_CONFIG["nms"]:
                    pool = ForcedColdWeightPool(
                        R, H, I_dim, pool_size=pool_size,
                        dtype=N1_CONFIG["dtype_weight"], device="cpu",
                        seed=42 + trial_idx * 100 + R
                    )

                    for path_name in path_order:
                        latencies = run_trial(
                            path_name, R, H, I_dim, NM, n_requests,
                            pool, pool_offset=0, num_cores=1,
                            remote_session=remote_session if path_name == "remote_improved" else None
                        )
                        for i, lat in enumerate(latencies):
                            results.append({
                                "trial": trial_idx,
                                "R": R, "NM": NM,
                                "path": path_name,
                                "request_idx": i,
                                "L_recovery_us": lat,
                            })

                        print(f"  trial {trial_idx} R={R} NM={NM} {path_name}: "
                              f"median={statistics.median(latencies):.1f}us "
                              f"p99={sorted(latencies)[int(0.99*len(latencies))]:.1f}us")
    finally:
        if remote_session:
            remote_session.__exit__(None, None, None)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "n1_crossover.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "R", "NM", "path", "request_idx", "L_recovery_us"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="N1: Isolated live crossover")
    parser.add_argument("--output", default="results/n1_crossover", help="Output directory")
    parser.add_argument("--trials", type=int, default=N1_CONFIG["n_trials"])
    parser.add_argument("--requests", type=int, default=N1_CONFIG["n_requests_per_trial"])
    parser.add_argument("--server-host", default=None, help="Remote server host (for remote_improved)")
    args = parser.parse_args()

    run_n1(args.output, n_trials=args.trials, n_requests=args.requests,
           remote_server_host=args.server_host)


if __name__ == "__main__":
    main()
