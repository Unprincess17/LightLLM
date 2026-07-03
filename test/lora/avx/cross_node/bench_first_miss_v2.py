"""S1: First-miss tax re-measurement with dual timing domains.

5 variants x 3 cells x 4 NM x 5 trials = 300 runs.
Cells: B1, B2, B5 (B6 deferred to Plan 2).
Variants: baseline, same_weights, allocator-reset, device-cache-perturbation, cuda_graph.
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


def run_s1(output_dir="results/s1_first_miss", server_host=None, ssh_host=None):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for cell in CELLS_S1:
        cell_spec = CELLS[cell]
        for variant in VARIANTS:
            # Start server + QP pool once per (cell, variant) — 15 cycles
            # instead of 300 (I3 fix).
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
                        trial_p50s, trial_p99s, first_rest_ratios = [], [], []
                        for trial in range(N_TRIALS):
                            try:
                                result = session.run(nm, n_iters=50)
                            except Exception as e:
                                print(f"  {cell} {variant} NM={nm} trial {trial}: "
                                      f"skipped ({e})")
                                continue
                            lats = result.get("latencies_us", [])
                            if not lats:
                                continue
                            first = lats[0]
                            rest_mean = statistics.mean(lats[1:]) if len(lats) > 1 else first
                            ratio = first / rest_mean if rest_mean > 0 else float("nan")
                            trial_p50s.append(statistics.median(lats))
                            trial_p99s.append(sorted(lats)[int(0.99 * len(lats)) - 1])
                            first_rest_ratios.append(ratio)
                        if not trial_p50s:
                            print(f"{cell} {variant} NM={nm}: no successful trials")
                            continue
                        p50_lo, p50_hi = trial_ci(trial_p50s)
                        p99_lo, p99_hi = trial_ci(trial_p99s)
                        ratio_lo, ratio_hi = trial_ci(first_rest_ratios)
                        rows.append({
                            "cell": cell, "variant": variant, "nm": nm,
                            "p50_median": statistics.median(trial_p50s),
                            "p50_ci_lo": p50_lo, "p50_ci_hi": p50_hi,
                            "p99_median": statistics.median(trial_p99s),
                            "p99_ci_lo": p99_lo, "p99_ci_hi": p99_hi,
                            "first_rest_ratio_median": statistics.median(first_rest_ratios),
                            "ratio_ci_lo": ratio_lo, "ratio_ci_hi": ratio_hi,
                        })
                        print(f"{cell} {variant} NM={nm}: "
                              f"ratio={statistics.median(first_rest_ratios):.2f}x")
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
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="S1 first-miss tax re-measurement")
    parser.add_argument("--output", default="results/s1_first_miss")
    parser.add_argument("--server-host", default=None)
    parser.add_argument("--ssh-host", default=None)
    args = parser.parse_args()
    run_s1(args.output, args.server_host, args.ssh_host)


if __name__ == "__main__":
    main()
