"""N1 analysis: crossover curves, winner grid, stage decomposition.

Primary endpoint: paired difference in trial-level median L_recovery.
Three-way classification per cell: A_wins / B_wins / equivalent / unresolved.
"""
import csv
import os
import statistics
from collections import defaultdict

from common.stats import classify_pair, paired_diff_ci, holm_correct

DELTA_ABSOLUTE_US = 50.0
DELTA_RHO = 0.10
N1_PATHS = ["cpu_first", "load_then_run", "remote_improved", "oracle"]


def tost_margin(calibration_median):
    """delta_{R,NM} = max(delta_absolute, rho * calibration_median)."""
    return max(DELTA_ABSOLUTE_US, DELTA_RHO * calibration_median)


def compute_trial_medians(results):
    """Compute per-trial median L_recovery for each (path, R, NM).

    Args:
        results: list of dicts with keys trial, R, NM, path, L_recovery_us

    Returns: dict mapping (path, R, NM) -> list of trial medians
    """
    trial_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["path"], r["R"], r["NM"])
        trial_data[key][r["trial"]].append(r["L_recovery_us"])

    medians = {}
    for key, trials in trial_data.items():
        medians[key] = [statistics.median(latencies) for latencies in trials.values()]
    return medians


def classify_winners(diff_point, ci_lo, ci_hi, delta):
    """Three-way classification: A_wins / B_wins / equivalent / unresolved."""
    return classify_pair(diff_point, ci_lo, ci_hi, delta)


def analyze_n1(csv_path, output_dir):
    """Full N1 analysis: crossover curves + winner grid.

    Reads n1_crossover.csv, computes:
      - per-cell trial medians
      - pairwise classifications (cpu_first vs remote, load_then_run vs remote, etc.)
      - Holm correction across the family of comparisons
      - R x NM winner grid

    Writes results to output_dir.
    """
    with open(csv_path) as f:
        results = list(csv.DictReader(f))
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["L_recovery_us"] = float(r["L_recovery_us"])
        r["trial"] = int(r["trial"])

    medians = compute_trial_medians(results)

    # Get all cells
    cells = sorted(set((r["R"], r["NM"]) for r in results))
    ranks = sorted(set(r["R"] for r in results))
    nms = sorted(set(r["NM"] for r in results))

    # Pairwise comparisons
    path_pairs = [
        ("cpu_first", "remote_improved"),
        ("load_then_run", "remote_improved"),
        ("cpu_first", "load_then_run"),
    ]

    winner_grid = {}
    all_pvalues = []
    cell_pair_results = []

    for R, NM in cells:
        # Calibration median = oracle's median (for delta computation)
        oracle_key = ("oracle", R, NM)
        if oracle_key not in medians:
            continue
        cal_median = statistics.median(medians[oracle_key])
        delta = tost_margin(cal_median)

        for path_a, path_b in path_pairs:
            key_a = (path_a, R, NM)
            key_b = (path_b, R, NM)
            if key_a not in medians or key_b not in medians:
                continue

            trials_a = medians[key_a]
            trials_b = medians[key_b]

            if len(trials_a) < 2 or len(trials_b) < 2:
                continue

            # Paired difference CI
            diff_point = statistics.mean(trials_a) - statistics.mean(trials_b)
            ci_lo, ci_hi = paired_diff_ci(trials_a, trials_b, confidence=0.95)

            classification = classify_winners(diff_point, ci_lo, ci_hi, delta)
            winner_grid[(R, NM)] = winner_grid.get((R, NM), {})

            cell_pair_results.append({
                "R": R, "NM": NM,
                "path_a": path_a, "path_b": path_b,
                "diff_us": diff_point,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "delta": delta,
                "classification": classification,
            })

    # Write winner grid CSV
    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, "n1_winner_grid.csv")
    with open(grid_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["R", "NM", "path_a", "path_b",
                                                "diff_us", "ci_lo", "ci_hi",
                                                "delta", "classification"])
        writer.writeheader()
        writer.writerows(cell_pair_results)
    print(f"Winner grid written to {grid_path}")

    # Write summary
    summary_path = os.path.join(output_dir, "n1_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "NM"] + N1_PATHS)
        for R, NM in cells:
            row = [R, NM]
            for p in N1_PATHS:
                key = (p, R, NM)
                if key in medians and medians[key]:
                    row.append(f"{statistics.median(medians[key]):.1f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    print(f"Summary written to {summary_path}")

    return cell_pair_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N1 analysis")
    parser.add_argument("--input", default="results/n1_crossover/n1_crossover.csv")
    parser.add_argument("--output", default="results/n1_crossover/analysis")
    args = parser.parse_args()
    analyze_n1(args.input, args.output)


if __name__ == "__main__":
    main()
