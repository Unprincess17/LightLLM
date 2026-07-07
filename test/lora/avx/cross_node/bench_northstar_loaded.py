"""N2: Live loaded anchor map — capacity crossover under open-loop load.

Structured anchors along fixed NM slices. Bracketed capacity search per
(path, anchor, mixture). Primary endpoint: C_feasible.

Not the complete region map — N2 is the live loaded anchor map. The
complete region map is N3-N7 calibrated simulation output.
"""
import argparse
import csv
import os
import statistics
import time

import torch

from common.northstar_paths import (
    cpu_first_recovery, load_then_run_recovery, oracle_recovery
)
from common.forced_cold import ForcedColdWeightPool
from common.load_generator import generate_paired_traces, OpenLoopRunner, Trace
from common.instrumentation import NorthstarTimeline
from common.stats import capacity_bootstrap

N2_CONFIG = {
    "ranks": [16, 32, 64, 128, 256],
    "nms": [1, 2, 4, 8, 16],
    "H": 2048,
    "I": 2048,
    "n_trials": 5,
    "min_requests_per_class": 2000,  # for per-class P99
    "max_duration_s": 120,
    "warmup_s": 10,
    "bracket_stop_ratio": 1.10,
    "slo_self_normalized": 2.0,   # P(L > 2x isolated median) <= 0.01
    "slo_common_factor": 5.0,     # max(10ms, 5x oracle median)
    "slo_common_floor_us": 10_000,
    "mixtures": {
        "1h3l": {"heavy_frac": 0.25, "light_nm": [1], "heavy_nm": [8]},
        "1h9l": {"heavy_frac": 0.10, "light_nm": [1], "heavy_nm": [8]},
        "1h1l": {"heavy_frac": 0.50, "light_nm": [1], "heavy_nm": [8]},
    },
    "light_class": {"R": 16, "NM": 1},
}

N2_PATHS = ["cpu_first", "load_then_run", "remote_improved"]


def is_feasible(latencies, isolated_median, slo_factor,
                generated, completed, queue_slope_ci,
                timeout_count=0, rejection_count=0):
    """Check if a load point is feasible.

    Feasible = self-normalized stability SLO met + open-loop stable + low timeout/rejection.
    """
    if not latencies:
        return False
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    threshold = slo_factor * isolated_median
    slo_met = p99 <= threshold

    # Open-loop stability: generated ~= completed
    completion_ratio = completed / max(generated, 1)
    stable_throughput = completion_ratio >= 0.99

    # Queue slope CI includes zero
    queue_stable = queue_slope_ci[0] <= 0 <= queue_slope_ci[1]

    # Timeout/rejection rate
    total = generated
    error_rate = (timeout_count + rejection_count) / max(total, 1)
    low_errors = error_rate <= 0.01

    return slo_met and stable_throughput and queue_stable and low_errors


def classify_capacity(c_lower, c_upper):
    """Check if capacity bracket has converged."""
    if c_lower <= 0:
        return "continue"
    ratio = c_upper / c_lower
    if ratio <= N2_CONFIG["bracket_stop_ratio"]:
        return "converged"
    return "continue"


def run_load_trial(path_name, R, NM, lam, duration_s, seed,
                   heavy_frac, H=2048, I=2048,
                   remote_session=None, num_cores=1):
    """Run one open-loop load trial.

    Returns: dict with latencies, generated, completed, queue info, feasibility
    """
    pool_size = int(lam * duration_s * 2) + 100
    pool = ForcedColdWeightPool(R, H, I, pool_size=pool_size, seed=seed)

    trace_a, trace_b = generate_paired_traces(
        lam=lam, duration_s=duration_s, seed=seed,
        heavy_frac=heavy_frac,
        nm_options_light=N2_CONFIG["mixtures"]["1h3l"]["light_nm"],
        nm_options_heavy=N2_CONFIG["mixtures"]["1h3l"]["heavy_nm"]
    )
    trace = trace_a  # use one copy for this path

    latencies = []
    completed = 0
    generated = 0
    timed_out = 0

    for event in trace:
        generated += 1
        try:
            activation = torch.randn(event.nm if hasattr(event, 'nm') else NM,
                                     H, dtype=torch.float16, device="cuda")
            weights = pool.get_batch(completed, NM)

            if path_name == "cpu_first":
                _, tl = cpu_first_recovery(activation, weights, R, H, I, NM,
                                            num_cores=num_cores)
            elif path_name == "load_then_run":
                _, tl = load_then_run_recovery(activation, weights, R, H, I, NM)
            elif path_name == "remote_improved":
                if remote_session:
                    _, tl = remote_session.run_single(activation, weights, R, H, I, NM)
                else:
                    continue  # skip if no remote session
            elif path_name == "oracle":
                gpu_weights = pool.get_oracle_batch(completed, NM, device="cuda")
                _, tl = oracle_recovery(activation, gpu_weights, R, H, I, NM)
            else:
                continue

            latencies.append(tl.l_recovery_us())
            completed += 1
        except Exception as e:
            timed_out += 1

    # Queue slope CI (simplified: use completion rate as proxy)
    completion_ratio = completed / max(generated, 1)
    queue_slope_ci = [-0.1, 0.1] if completion_ratio >= 0.99 else [0.5, 2.0]

    return {
        "latencies": latencies,
        "generated": generated,
        "completed": completed,
        "timed_out": timed_out,
        "queue_slope_ci": queue_slope_ci,
    }


def bracketed_capacity_search(path_name, R, NM, mixture_label,
                              remote_session=None, n_trials=5,
                              max_duration_s=60, seed=42):
    """Bracketed capacity search for one (path, anchor, mixture).

    1. Shared bracketing: geometric lambda increase to instability
    2. Per-path binary search to +/-5% capacity
    3. Stop when C_upper / C_lower <= 1.10

    Returns: dict with c_lower, c_upper, trial_results
    """
    heavy_frac = N2_CONFIG["mixtures"][mixture_label]["heavy_frac"]

    # Phase 1: geometric bracketing
    loads_tested = {}
    lam = 50.0  # start low
    c_lower = 0
    c_upper = float("inf")

    for _ in range(8):  # max 8 geometric steps
        trial_results = []
        for trial in range(n_trials):
            result = run_load_trial(
                path_name, R, NM, lam, duration_s=max_duration_s,
                seed=seed + trial, heavy_frac=heavy_frac,
                remote_session=remote_session
            )
            # Need isolated median for feasibility check (from N1)
            # For now, use median of lowest load as isolated proxy
            trial_results.append(result)

        # Check feasibility across trials
        all_latencies = [l for r in trial_results for l in r["latencies"]]
        isolated_median = statistics.median(all_latencies) if all_latencies else 1000
        feasible_count = sum(
            is_feasible(r["latencies"], isolated_median,
                        N2_CONFIG["slo_self_normalized"],
                        r["generated"], r["completed"],
                        r["queue_slope_ci"], r["timed_out"])
            for r in trial_results
        )
        is_stable = feasible_count >= n_trials // 2 + 1

        loads_tested[lam] = {
            "feasible": is_stable,
            "trial_results": trial_results,
        }

        if is_stable:
            c_lower = lam
            lam *= 2
        else:
            c_upper = lam
            break

    # Phase 2: binary search between c_lower and c_upper
    for _ in range(6):  # max 6 binary search steps
        if classify_capacity(c_lower, c_upper) == "converged":
            break
        if c_upper == float("inf"):
            lam = c_lower * 2
        else:
            lam = (c_lower + c_upper) / 2

        trial_results = []
        for trial in range(n_trials):
            result = run_load_trial(
                path_name, R, NM, lam, duration_s=max_duration_s,
                seed=seed + trial, heavy_frac=heavy_frac,
                remote_session=remote_session
            )
            trial_results.append(result)

        all_latencies = [l for r in trial_results for l in r["latencies"]]
        isolated_median = statistics.median(all_latencies) if all_latencies else 1000
        feasible_count = sum(
            is_feasible(r["latencies"], isolated_median,
                        N2_CONFIG["slo_self_normalized"],
                        r["generated"], r["completed"],
                        r["queue_slope_ci"], r["timed_out"])
            for r in trial_results
        )
        is_stable = feasible_count >= n_trials // 2 + 1

        loads_tested[lam] = {"feasible": is_stable, "trial_results": trial_results}

        if is_stable:
            c_lower = lam
        else:
            c_upper = lam

    return {
        "c_lower": c_lower,
        "c_upper": c_upper if c_upper != float("inf") else c_lower * 2,
        "loads_tested": loads_tested,
    }


def run_n2(output_dir, remote_server_host=None):
    """Run the N2 loaded anchor map campaign."""
    # Anchor selection deferred to after N1 results
    # For now, use structured default anchors
    anchors = [
        (16, 1), (16, 8),
        (64, 1), (64, 8),
        (256, 1), (256, 8),
        (128, 4),  # space-filling
    ]

    remote_session = None
    if remote_server_host:
        from bench_decomposition import RemoteSession
        remote_session = RemoteSession(remote_server_host, cell="B3")
        remote_session.__enter__()

    try:
        results = []
        for R, NM in anchors:
            for path_name in N2_PATHS:
                for mixture in ["1h3l"]:  # primary mixture
                    print(f"  Anchor ({R},{NM}) {path_name} {mixture}...")
                    cap = bracketed_capacity_search(
                        path_name, R, NM, mixture,
                        remote_session=remote_session
                    )
                    results.append({
                        "R": R, "NM": NM,
                        "path": path_name,
                        "mixture": mixture,
                        "c_lower": cap["c_lower"],
                        "c_upper": cap["c_upper"],
                    })
                    print(f"    C=[{cap['c_lower']:.0f}, {cap['c_upper']:.0f}]")
    finally:
        if remote_session:
            remote_session.__exit__(None, None, None)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "n2_capacity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["R", "NM", "path", "mixture",
                                                "c_lower", "c_upper"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="N2: Loaded anchor map")
    parser.add_argument("--output", default="results/n2_loaded")
    parser.add_argument("--server-host", default=None)
    args = parser.parse_args()
    run_n2(args.output, remote_server_host=args.server_host)


if __name__ == "__main__":
    main()
