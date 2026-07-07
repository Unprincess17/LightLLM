"""N1 analysis: crossover curves, winner grid, stage decomposition.

Primary endpoint: paired difference in trial-level median L_recovery.
Three-way classification per cell: A_wins / B_wins / equivalent / unresolved.
Holm-Bonferroni step-down correction applied across the full family of
comparisons (alpha = 0.05).
"""
import csv
import math
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


def _paired_pvalue(trials_a, trials_b):
    """Paired t-test p-value for the family of comparisons.

    Uses scipy.stats.ttest_rel on trial medians.  Falls back to a CI-based
    heuristic when scipy is unavailable or returns nan (degenerate
    zero-variance input): p = 0.0 if the CI excludes zero, else 1.0.
    """
    try:
        from scipy import stats as sp_stats
        result = sp_stats.ttest_rel(trials_a, trials_b)
        p = result.pvalue if hasattr(result, "pvalue") else result[1]
        if math.isnan(p):
            raise ValueError("nan p-value")
        return float(p)
    except Exception:
        ci_lo, ci_hi = paired_diff_ci(trials_a, trials_b, confidence=0.95)
        return 0.0 if (ci_lo > 0 or ci_hi < 0) else 1.0


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
    raw_comparisons = []

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
            pvalue = _paired_pvalue(trials_a, trials_b)

            raw_comparisons.append({
                "R": R, "NM": NM,
                "path_a": path_a, "path_b": path_b,
                "diff_us": diff_point,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "delta": delta,
                "pvalue": pvalue,
            })
            all_pvalues.append(pvalue)

    # Apply Holm-Bonferroni step-down correction across the full family
    holm_rejected = holm_correct(all_pvalues, alpha=0.05) if all_pvalues else []

    cell_pair_results = []
    for i, raw in enumerate(raw_comparisons):
        rejected = holm_rejected[i] if i < len(holm_rejected) else False
        if rejected:
            classification = classify_winners(
                raw["diff_us"], raw["ci_lo"], raw["ci_hi"], raw["delta"]
            )
        else:
            # Holm did not reject — cannot claim A_wins or B_wins.
            # Downgrade to "equivalent" if CI lies within [-delta, +delta],
            # otherwise "unresolved".
            if raw["ci_lo"] >= -raw["delta"] and raw["ci_hi"] <= raw["delta"]:
                classification = "equivalent"
            else:
                classification = "unresolved"

        cell_pair_results.append({
            "R": raw["R"], "NM": raw["NM"],
            "path_a": raw["path_a"], "path_b": raw["path_b"],
            "diff_us": raw["diff_us"],
            "ci_lo": raw["ci_lo"], "ci_hi": raw["ci_hi"],
            "delta": raw["delta"],
            "pvalue": raw["pvalue"],
            "holm_rejected": rejected,
            "classification": classification,
        })

    # Write winner grid CSV
    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, "n1_winner_grid.csv")
    with open(grid_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["R", "NM", "path_a", "path_b",
                                                "diff_us", "ci_lo", "ci_hi",
                                                "delta", "pvalue",
                                                "holm_rejected", "classification"])
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
