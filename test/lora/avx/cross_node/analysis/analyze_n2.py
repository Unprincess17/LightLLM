"""N2 analysis: capacity brackets, region map, winner classification.

Primary endpoint: C_feasible (highest lambda satisfying stability + SLO).
Capacity winner = path with highest C_feasible.
Operational winner at a particular lambda = feasible path with lowest P99.
"""
import csv
import os
import warnings
from collections import defaultdict

from common.stats import capacity_bootstrap

N2_PATHS = ["cpu_first", "load_then_run", "remote_improved"]


def compute_capacity_brackets(results):
    """Extract capacity brackets from N2 results.

    Args:
        results: list of dicts with R, NM, path, mixture, c_lower, c_upper

    Returns: dict mapping (path, R, NM) -> {c_lower, c_upper}
    """
    brackets = {}
    for r in results:
        key = (r["path"], r["R"], r["NM"])
        brackets[key] = {
            "c_lower": float(r["c_lower"]),
            "c_upper": float(r["c_upper"]),
        }
    return brackets


def _load_raw_trials(raw_path):
    """Load raw trial results and group by (path, R, NM).

    Returns: dict mapping (path, R, NM) -> {load -> [bool, ...]}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            key = (row["path"], int(row["R"]), int(row["NM"]))
            load = float(row["load"])
            feasible = row["feasible"].strip().lower() in ("true", "1", "yes")
            grouped[key][load].append(feasible)
    return {k: dict(v) for k, v in grouped.items()}


def capacity_winner(brackets_for_cell):
    """Determine capacity winner for one cell.

    Args:
        brackets_for_cell: dict mapping path -> {c_lower, c_upper}

    Returns: path name with highest c_lower (C_feasible)
    """
    best_path = None
    best_c = -1
    for path, bracket in brackets_for_cell.items():
        if bracket["c_lower"] > best_c:
            best_c = bracket["c_lower"]
            best_path = path
    return best_path


def analyze_n2(csv_path, output_dir):
    """Full N2 analysis: capacity brackets + region map.

    If n2_raw_trials.csv exists alongside csv_path, applies capacity_bootstrap
    to compute CI bounds for c_lower/c_upper.
    """
    with open(csv_path) as f:
        results = list(csv.DictReader(f))
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["c_lower"] = float(r["c_lower"])
        r["c_upper"] = float(r["c_upper"])

    brackets = compute_capacity_brackets(results)

    # Check for raw trials file for bootstrap
    raw_path = os.path.join(os.path.dirname(csv_path), "n2_raw_trials.csv")
    raw_trials = None
    if os.path.exists(raw_path):
        raw_trials = _load_raw_trials(raw_path)
    else:
        warnings.warn(
            f"n2_raw_trials.csv not found at {raw_path}; "
            "falling back to point estimates only (no bootstrap CIs)."
        )

    # Compute bootstrap CIs per (path, R, NM)
    bootstrap_cis = {}  # (path, R, NM) -> CI dict
    if raw_trials:
        for key, trial_results in raw_trials.items():
            bootstrap_cis[key] = capacity_bootstrap(trial_results)

    # Group by cell
    cells = sorted(set((r["R"], r["NM"]) for r in results))

    # Region map
    region_map = []
    for R, NM in cells:
        cell_brackets = {}
        for path in N2_PATHS:
            key = (path, R, NM)
            if key in brackets:
                cell_brackets[path] = brackets[key]

        if not cell_brackets:
            continue

        winner = capacity_winner(cell_brackets)
        n_feasible = sum(1 for b in cell_brackets.values() if b["c_lower"] > 0)

        row = {
            "R": R, "NM": NM,
            "capacity_winner": winner,
            "n_feasible": n_feasible,
        }
        for p in N2_PATHS:
            b = cell_brackets.get(p)
            if b is not None:
                row[f"{p}_c_lower"] = b["c_lower"]
            else:
                row[f"{p}_c_lower"] = "N/A"

            ci = bootstrap_cis.get((p, R, NM))
            if ci is not None:
                row[f"{p}_c_lower_ci_lo"] = ci["c_lower_ci_lo"]
                row[f"{p}_c_lower_ci_hi"] = ci["c_lower_ci_hi"]
                row[f"{p}_c_upper_ci_lo"] = ci["c_upper_ci_lo"]
                row[f"{p}_c_upper_ci_hi"] = ci["c_upper_ci_hi"]
            else:
                row[f"{p}_c_lower_ci_lo"] = "N/A"
                row[f"{p}_c_lower_ci_hi"] = "N/A"
                row[f"{p}_c_upper_ci_lo"] = "N/A"
                row[f"{p}_c_upper_ci_hi"] = "N/A"

        region_map.append(row)

    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "n2_region_map.csv")
    fields = ["R", "NM", "capacity_winner", "n_feasible"] + \
             [f"{p}_c_lower" for p in N2_PATHS] + \
             [f"{p}_c_lower_ci_lo" for p in N2_PATHS] + \
             [f"{p}_c_lower_ci_hi" for p in N2_PATHS] + \
             [f"{p}_c_upper_ci_lo" for p in N2_PATHS] + \
             [f"{p}_c_upper_ci_hi" for p in N2_PATHS]
    with open(map_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(region_map)
    print(f"Region map written to {map_path}")
    return region_map


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N2 analysis")
    parser.add_argument("--input", default="results/n2_loaded/n2_capacity.csv")
    parser.add_argument("--output", default="results/n2_loaded/analysis")
    args = parser.parse_args()
    analyze_n2(args.input, args.output)


if __name__ == "__main__":
    main()
