"""N2 analysis: capacity brackets, region map, winner classification.

Primary endpoint: C_feasible (highest lambda satisfying stability + SLO).
Capacity winner = path with highest C_feasible.
Operational winner at a particular lambda = feasible path with lowest P99.
"""
import csv
import os
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
    """Full N2 analysis: capacity brackets + region map."""
    with open(csv_path) as f:
        results = list(csv.DictReader(f))
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["c_lower"] = float(r["c_lower"])
        r["c_upper"] = float(r["c_upper"])

    brackets = compute_capacity_brackets(results)

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

        region_map.append({
            "R": R, "NM": NM,
            "capacity_winner": winner,
            "n_feasible": n_feasible,
            **{f"{p}_c_lower": cell_brackets.get(p, {}).get("c_lower", "N/A")
               for p in N2_PATHS},
        })

    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "n2_region_map.csv")
    with open(map_path, "w", newline="") as f:
        fields = ["R", "NM", "capacity_winner", "n_feasible"] + \
                 [f"{p}_c_lower" for p in N2_PATHS]
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
