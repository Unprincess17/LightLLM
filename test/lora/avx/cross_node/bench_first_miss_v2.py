"""S1: First-miss tax re-measurement with dual timing domains.

5 variants x 3 cells x 4 NM x 5 trials = 300 runs.
Cells: B1, B2, B5 (B6 deferred to Plan 2).
Variants: baseline, same_weights, allocator-reset, device-cache-perturbation, cuda_graph.

Primary metric: per-MISS first/rest ratio (miss_0 GPU time vs median(miss_1..7) GPU time
within a single request). This matches the original study's measurement.

Secondary metric: per-ITERATION first/rest ratio (first request E2E vs subsequent).
This measures request-level warmup (QP creation, CUDA init).
"""
import argparse
import csv
import os
import statistics

from bench_decomposition import DecompositionConfig, run_cell, CELLS, RemoteSession
from common.instrumentation import account_request, RunMetadata
from common.stats import trial_ci, percentile_ci

VARIANTS = ["baseline", "same_weights", "allocator-reset",
            "device-cache-perturbation", "cuda_graph"]
CELLS_S1 = ["B1", "B2", "B5"]  # B6 added in Plan 2
NMS = [1, 2, 4, 8]
N_TRIALS = 5
N_ITERS = 50


def _extract_per_miss_gpu(segs_all, nm):
    """Extract per-miss (first_gpu, rest_median_gpu) from server segments.

    Returns list of (first, rest_median) per request.
    """
    results = []
    for segs in segs_all:
        miss_gpu = {}
        for s in segs:
            name = s.get("name", "")
            if name.startswith("miss_"):
                parts = name.split("_")
                idx = int(parts[1])
                miss_gpu[idx] = miss_gpu.get(idx, 0.0) + s.get("gpu_us", s.get("cpu_us", 0.0))
        if 0 in miss_gpu and len(miss_gpu) > 1:
            first = miss_gpu[0]
            rest_med = statistics.median([miss_gpu[i] for i in range(1, nm) if i in miss_gpu])
            results.append((first, rest_med))
    return results


def _extract_graph_replay_gpu(segs_all):
    """For cuda_graph variant: extract graph_replay GPU time per request."""
    results = []
    for segs in segs_all:
        for s in segs:
            if s.get("name") == "graph_replay":
                results.append(s.get("gpu_us", 0.0))
                break
    return results


def run_s1(output_dir="results/s1_first_miss", server_host=None, ssh_host=None,
           n_trials=None, n_iters=None):
    n_trials = n_trials or N_TRIALS
    n_iters = n_iters or N_ITERS
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for cell in CELLS_S1:
        cell_spec = CELLS[cell]
        for variant in VARIANTS:
            session_kwargs = {}
            if server_host is not None:
                session_kwargs["server_host"] = server_host
            if ssh_host is not None:
                session_kwargs["ssh_host"] = ssh_host

            try:
                with RemoteSession(
                    cell_spec, cell=cell, variant=variant,
                    max_nm=max(NMS), **session_kwargs,
                ) as session:
                    for nm in NMS:
                        trial_p50s = []
                        trial_p99s = []
                        trial_per_miss_ratios = []
                        trial_per_miss_first = []
                        trial_per_miss_rest = []
                        trial_per_iter_ratios = []
                        trial_graph_replay_gpu = []

                        for trial in range(n_trials):
                            try:
                                result = session.run(nm, n_iters=n_iters)
                            except Exception as e:
                                print(f"  {cell} {variant} NM={nm} trial {trial}: "
                                      f"skipped ({e})")
                                continue
                            lats = result.get("latencies_us", [])
                            segs_all = result.get("segments_per_request", [])
                            if not lats:
                                continue

                            # Per-iteration ratio (secondary: warmup effect)
                            first_iter = lats[0]
                            rest_iter_med = statistics.median(lats[1:]) if len(lats) > 1 else first_iter
                            iter_ratio = first_iter / rest_iter_med if rest_iter_med > 0 else float("nan")
                            trial_per_iter_ratios.append(iter_ratio)

                            # Per-miss ratio (primary: the actual first-miss tax)
                            if variant == "cuda_graph":
                                # Graph variant: only has graph_replay, no per-miss segments
                                graph_gpus = _extract_graph_replay_gpu(segs_all)
                                if graph_gpus:
                                    trial_graph_replay_gpu.append(statistics.median(graph_gpus))
                                # Per-miss ratio not applicable for graph (single replay segment)
                                trial_per_miss_ratios.append(float("nan"))
                                trial_per_miss_first.append(float("nan"))
                                trial_per_miss_rest.append(float("nan"))
                            else:
                                per_miss = _extract_per_miss_gpu(segs_all, nm)
                                if per_miss:
                                    ratios = [f / r if r > 0 else float("nan")
                                             for f, r in per_miss]
                                    firsts = [f for f, r in per_miss]
                                    rests = [r for f, r in per_miss]
                                    trial_per_miss_ratios.append(statistics.median(ratios))
                                    trial_per_miss_first.append(statistics.median(firsts))
                                    trial_per_miss_rest.append(statistics.median(rests))

                            trial_p50s.append(statistics.median(lats))
                            trial_p99s.append(sorted(lats)[int(0.99 * len(lats)) - 1])

                        if not trial_p50s:
                            print(f"{cell} {variant} NM={nm}: no successful trials")
                            continue

                        # Compute CIs
                        p50_lo, p50_hi = trial_ci(trial_p50s)
                        p99_lo, p99_hi = trial_ci(trial_p99s)

                        # Per-miss ratio (primary)
                        valid_pm_ratios = [r for r in trial_per_miss_ratios if r == r]  # filter NaN
                        if valid_pm_ratios:
                            pm_ratio_med = statistics.median(valid_pm_ratios)
                            pm_ratio_lo, pm_ratio_hi = trial_ci(valid_pm_ratios)
                            pm_first_med = statistics.median([v for v in trial_per_miss_first if v == v])
                            pm_rest_med = statistics.median([v for v in trial_per_miss_rest if v == v])
                        else:
                            pm_ratio_med = pm_ratio_lo = pm_ratio_hi = float("nan")
                            pm_first_med = pm_rest_med = float("nan")

                        # Per-iteration ratio (secondary)
                        valid_pi_ratios = [r for r in trial_per_iter_ratios if r == r]
                        if valid_pi_ratios:
                            pi_ratio_med = statistics.median(valid_pi_ratios)
                            pi_ratio_lo, pi_ratio_hi = trial_ci(valid_pi_ratios)
                        else:
                            pi_ratio_med = pi_ratio_lo = pi_ratio_hi = float("nan")

                        # Graph replay GPU (for cuda_graph variant)
                        graph_gpu_med = statistics.median(trial_graph_replay_gpu) if trial_graph_replay_gpu else float("nan")

                        rows.append({
                            "cell": cell, "variant": variant, "nm": nm,
                            "p50_e2e_us": statistics.median(trial_p50s),
                            "p50_ci_lo": p50_lo, "p50_ci_hi": p50_hi,
                            "p99_e2e_us": statistics.median(trial_p99s),
                            "p99_ci_lo": p99_lo, "p99_ci_hi": p99_hi,
                            "per_miss_ratio": pm_ratio_med,
                            "per_miss_ratio_ci_lo": pm_ratio_lo,
                            "per_miss_ratio_ci_hi": pm_ratio_hi,
                            "per_miss_first_gpu_us": pm_first_med,
                            "per_miss_rest_gpu_us": pm_rest_med,
                            "per_iter_ratio": pi_ratio_med,
                            "per_iter_ratio_ci_lo": pi_ratio_lo,
                            "per_iter_ratio_ci_hi": pi_ratio_hi,
                            "graph_replay_gpu_us": graph_gpu_med,
                        })

                        ratio_str = f"{pm_ratio_med:.2f}x" if pm_ratio_med == pm_ratio_med else "N/A"
                        print(f"{cell} {variant} NM={nm}: "
                              f"per_miss={ratio_str}  "
                              f"per_iter={pi_ratio_med:.2f}x  "
                              f"p50={statistics.median(trial_p50s):.0f}us")
            except Exception as e:
                print(f"  {cell} {variant}: session failed ({e})")
                continue
    if not rows:
        print("No data collected; CSV not written.")
        return
    csv_path = os.path.join(output_dir, "first_miss.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="S1 first-miss tax re-measurement")
    parser.add_argument("--output", default="results/s1_first_miss")
    parser.add_argument("--server-host", default=None)
    parser.add_argument("--ssh-host", default=None)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    run_s1(args.output, args.server_host, args.ssh_host,
           n_trials=args.trials, n_iters=args.iters)


if __name__ == "__main__":
    main()
