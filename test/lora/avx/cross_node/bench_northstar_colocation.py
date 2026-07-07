"""N2.5: Live co-location calibration — inference + recovery interference.

Measures both recovery and inference metrics when a real inference engine
is co-located on the client node. Calibrates the simulator's interference model.

Primary endpoint: incremental P99 TPOT relative to paired inference-only baseline.

Key design:
  - Real LightLLM serving stack on client GPU (Qwen3-VL-MoE decode)
  - Token-coupled miss injection: each miss attached to an actual decode token
  - Faulting request cannot advance until T1 (recovery complete)
  - Inference-only baseline in every cell (for delta metrics)
  - 4 operating points x 3 intensities x 2 paths + baseline = ~270 trials
"""
import argparse
import csv
import os
import statistics
import time

import torch

from common.northstar_paths import cpu_first_recovery, load_then_run_recovery
from common.forced_cold import ForcedColdWeightPool
from common.instrumentation import NorthstarTimeline

# --- Configuration ---

N2_5_CONFIG = {
    "operating_points": [
        # (R, NM) — selected from N2 results after N2 gate
        # Placeholder defaults; updated after N2 completes
        {"label": "cpu_preferred", "R": 16, "NM": 1},
        {"label": "below_crossover", "R": 64, "NM": 1},
        {"label": "above_crossover", "R": 128, "NM": 8},
        {"label": "remote_preferred", "R": 256, "NM": 8},
    ],
    "intensities": ["low", "moderate", "near_knee"],
    "intensity_factors": {"low": 0.3, "moderate": 0.6, "near_knee": 0.9},
    "recovery_lambda_factor": 0.7,  # 0.7 * min(C_cpu, C_remote)
    "mixture": "1h3l",
    "heavy_frac": 0.25,
    "n_trials": 5,
    "H": 2048,
    "I": 2048,
    "max_decode_duration_s": 60,
    "inference_slo_factor": 2.0,  # P99 TPOT <= 2x inference-only baseline
    "paths": ["cpu_first", "remote_improved"],
    "calibration_split": {
        "calibration": {"points": ["cpu_preferred", "below_crossover", "above_crossover"],
                         "intensities": ["low", "moderate"]},
        "validation_spatial": {"points": ["remote_preferred"],
                                "intensities": ["low", "moderate"]},
        "validation_intensity": {"points": "all", "intensities": ["near_knee"]},
    },
}


def compute_inference_intensity(label, knee):
    """Compute inference offered rate from intensity label and knee.

    Args:
        label: "low", "moderate", or "near_knee"
        knee: inference-only stable capacity (requests/s)

    Returns: offered arrival rate (requests/s)
    """
    factor = N2_5_CONFIG["intensity_factors"][label]
    return factor * knee


def run_inference_only_baseline(inference_rate, duration_s, seed,
                                model_config=None):
    """Run inference-only (no recovery) to establish baseline TPOT.

    This uses the LightLLM serving stack to run decode-only workloads
    at the specified arrival rate, without any LoRA miss injection.

    Returns: dict with tpot_samples, throughput, gpu_util
    """
    # TODO: integrate with LightLLM serving stack
    # For now, return placeholder structure
    # Real implementation will start the serving engine, feed decode
    # requests at the specified rate, and collect TPOT samples.
    return {
        "tpot_samples": [],  # ms per token
        "throughput": 0.0,   # tokens/s
        "gpu_util": 0.0,
        "inference_rate": inference_rate,
        "duration_s": duration_s,
    }


def run_colocation_trial(path_name, R, NM, inference_rate, recovery_lam,
                         duration_s, seed, model_config=None,
                         remote_session=None, num_cores=1):
    """Run one co-location trial: inference + recovery on same node.

    The recovery miss stream is token-coupled: each miss is attached to
    an actual decode token and gates that token's progress until T1.

    Returns: dict with recovery_latencies, tpot_samples, throughput, metrics
    """
    H = N2_5_CONFIG["H"]
    I_dim = N2_5_CONFIG["I"]
    pool_size = int(recovery_lam * duration_s * 2) + 100
    pool = ForcedColdWeightPool(R, H, I_dim, pool_size=pool_size, seed=seed)

    recovery_latencies = []
    tpot_samples = []

    # TODO: integrate with LightLLM serving stack
    # Real implementation:
    # 1. Start serving engine with decode workload at inference_rate
    # 2. Inject LoRA misses at recovery_lam, attached to decode tokens
    # 3. Faulting token waits until recovery T1 before advancing
    # 4. Other GPU-ready tokens continue per real scheduler
    # 5. Collect recovery L_recovery + inference TPOT

    # Placeholder: run recovery requests independently (no real inference)
    n_recovery = int(recovery_lam * duration_s)
    for i in range(n_recovery):
        activation = torch.randn(NM, H, dtype=torch.float16, device="cuda")
        weights = pool.get_batch(i, NM)

        if path_name == "cpu_first":
            _, tl = cpu_first_recovery(activation, weights, R, H, I_dim, NM,
                                        num_cores=num_cores)
        elif path_name == "remote_improved":
            if remote_session:
                _, tl = remote_session.run_single(activation, weights, R, H, I_dim, NM)
            else:
                continue
        else:
            continue

        recovery_latencies.append(tl.l_recovery_us())

    return {
        "recovery_latencies": recovery_latencies,
        "tpot_samples": tpot_samples,
        "inference_rate": inference_rate,
        "recovery_lam": recovery_lam,
        "duration_s": duration_s,
    }


def run_n2_5(output_dir, remote_server_host=None,
             n2_capacity_results=None):
    """Run the full N2.5 co-location calibration campaign.

    Args:
        output_dir: output directory
        remote_server_host: remote recovery server host
        n2_capacity_results: dict mapping (path, R, NM) -> C_feasible
                             (from N2 results, for computing recovery lambda)
    """
    if n2_capacity_results is None:
        # Default capacities (updated from N2 results)
        n2_capacity_results = {}

    # Step 1: Inference-only knee characterization
    print("Phase 1: Inference-only knee characterization...")
    knee_results = {}
    for intensity_label in N2_5_CONFIG["intensities"]:
        # Preliminary sweep to find inference-only stable capacity
        for trial_rate in [100, 200, 500, 1000, 2000]:
            baseline = run_inference_only_baseline(
                inference_rate=trial_rate, duration_s=30, seed=42
            )
            knee_results[(intensity_label, trial_rate)] = baseline

    # Determine knee (highest stable rate)
    # TODO: implement knee detection from baseline results
    inference_knee = 1000  # placeholder

    # Step 2: Co-location trials
    print("Phase 2: Co-location trials...")
    results = []

    remote_session = None
    if remote_server_host:
        from bench_decomposition import RemoteSession
        remote_session = RemoteSession(remote_server_host, cell="B3")
        remote_session.__enter__()

    try:
        for op in N2_5_CONFIG["operating_points"]:
            R, NM = op["R"], op["NM"]

            # Recovery lambda = 0.7 * min(C_cpu, C_remote)
            c_cpu = n2_capacity_results.get(("cpu_first", R, NM), 500)
            c_remote = n2_capacity_results.get(("remote_improved", R, NM), 800)
            recovery_lam = N2_5_CONFIG["recovery_lambda_factor"] * min(c_cpu, c_remote)

            for intensity in N2_5_CONFIG["intensities"]:
                inference_rate = compute_inference_intensity(intensity, inference_knee)

                # Inference-only baseline
                for trial in range(N2_5_CONFIG["n_trials"]):
                    baseline = run_inference_only_baseline(
                        inference_rate=inference_rate,
                        duration_s=N2_5_CONFIG["max_decode_duration_s"],
                        seed=42 + trial
                    )
                    results.append({
                        "op_label": op["label"], "R": R, "NM": NM,
                        "intensity": intensity, "trial": trial,
                        "path": "inference_only",
                        "recovery_lam": 0,
                        "inference_rate": inference_rate,
                        **baseline,
                    })

                # Co-location trials
                for path_name in N2_5_CONFIG["paths"]:
                    for trial in range(N2_5_CONFIG["n_trials"]):
                        trial_result = run_colocation_trial(
                            path_name, R, NM, inference_rate, recovery_lam,
                            duration_s=N2_5_CONFIG["max_decode_duration_s"],
                            seed=42 + trial,
                            remote_session=remote_session
                        )
                        results.append({
                            "op_label": op["label"], "R": R, "NM": NM,
                            "intensity": intensity, "trial": trial,
                            "path": path_name,
                            "recovery_lam": recovery_lam,
                            "inference_rate": inference_rate,
                            **trial_result,
                        })

                        print(f"  {op['label']} ({R},{NM}) {intensity} {path_name} "
                              f"trial {trial}: "
                              f"recovery_median={statistics.median(trial_result['recovery_latencies']):.1f}us"
                              if trial_result['recovery_latencies'] else "  (no data)")
    finally:
        if remote_session:
            remote_session.__exit__(None, None, None)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "n2_5_colocation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "op_label", "R", "NM", "intensity", "trial", "path",
            "recovery_lam", "inference_rate",
            "recovery_p50_us", "recovery_p99_us",
            "tpot_p50_ms", "tpot_p99_ms",
            "throughput", "gpu_util",
        ])
        writer.writeheader()
        for r in results:
            lats = r.get("recovery_latencies", [])
            tpots = r.get("tpot_samples", [])
            writer.writerow({
                "op_label": r["op_label"],
                "R": r["R"], "NM": r["NM"],
                "intensity": r["intensity"],
                "trial": r["trial"],
                "path": r["path"],
                "recovery_lam": r.get("recovery_lam", 0),
                "inference_rate": r.get("inference_rate", 0),
                "recovery_p50_us": statistics.median(lats) if lats else "",
                "recovery_p99_us": sorted(lats)[int(0.99*len(lats))] if lats else "",
                "tpot_p50_ms": statistics.median(tpots) if tpots else "",
                "tpot_p99_ms": sorted(tpots)[int(0.99*len(tpots))] if tpots else "",
                "throughput": r.get("throughput", ""),
                "gpu_util": r.get("gpu_util", ""),
            })
    print(f"Results written to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="N2.5: Co-location calibration")
    parser.add_argument("--output", default="results/n2_5_colocation")
    parser.add_argument("--server-host", default=None)
    parser.add_argument("--n2-results", default=None,
                        help="Path to N2 capacity CSV for recovery lambda computation")
    args = parser.parse_args()

    # Load N2 capacities if provided
    n2_caps = {}
    if args.n2_results and os.path.exists(args.n2_results):
        with open(args.n2_results) as f:
            for r in csv.DictReader(f):
                key = (r["path"], int(r["R"]), int(r["NM"]))
                n2_caps[key] = float(r["c_lower"])

    run_n2_5(args.output, remote_server_host=args.server_host,
             n2_capacity_results=n2_caps)


if __name__ == "__main__":
    main()
