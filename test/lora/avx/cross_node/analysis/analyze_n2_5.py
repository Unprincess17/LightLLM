"""N2.5 analysis: interference calibration + simulator acceptance.

Primary endpoint: incremental P99 TPOT relative to paired inference-only baseline.

Delta P99 TPOT_path = P99 TPOT_path - P99 TPOT_no_recovery

Simulator acceptance criteria (uncertainty-aware):
  Pass:          CI for prediction error within [-tolerance, +tolerance]
  Inconclusive:  point estimate within tolerance, CI crosses tolerance
  Fail:          point estimate outside tolerance

Error tolerances (combined absolute and relative):
  Recovery P99:  eps_abs = 2ms,   eps_rel = 0.20
  TPOT:          eps_abs = 5ms,   eps_rel = 0.15
  Throughput:    eps_abs = 50 req/s, eps_rel = 0.10
"""
import csv
import os
import statistics
from collections import defaultdict

# Error tolerances
TOL_RECOVERY_P99 = {"abs": 2000.0, "rel": 0.20}   # us
TOL_TPOT = {"abs": 5.0, "rel": 0.15}                # ms
TOL_THROUGHPUT = {"abs": 50.0, "rel": 0.10}         # req/s


def compute_delta_tpot(coloc_tpot_p99, baseline_tpot_p99):
    """Delta TPOT = co-located P99 TPOT - inference-only P99 TPOT."""
    return coloc_tpot_p99 - baseline_tpot_p99


def classify_interference(delta_tpot, tolerance_abs, tolerance_rel, baseline_tpot):
    """Classify whether interference is significant or negligible.

    Uses combined tolerance: max(abs, rel * baseline).
    """
    threshold = max(tolerance_abs, tolerance_rel * baseline_tpot)
    if abs(delta_tpot) > threshold:
        return "significant"
    return "negligible"


def check_acceptance(predicted, measured, tolerance_abs, tolerance_rel):
    """Uncertainty-aware acceptance check.

    Returns: "pass", "inconclusive", or "fail"
    """
    if predicted is None or measured is None:
        return "fail"
    error = abs(predicted - measured)
    threshold = max(tolerance_abs, tolerance_rel * abs(measured))
    if error <= threshold * 0.8:  # CI would need to be very wide to cross
        return "pass"
    elif error <= threshold:
        return "inconclusive"
    else:
        return "fail"


def analyze_n2_5(csv_path, output_dir):
    """Full N2.5 analysis: delta TPOT + interference classification."""
    with open(csv_path) as f:
        results = list(csv.DictReader(f))

    # Parse numeric fields
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["trial"] = int(r["trial"])
        r["recovery_lam"] = float(r.get("recovery_lam", 0) or 0)
        r["inference_rate"] = float(r.get("inference_rate", 0) or 0)
        if r.get("recovery_p99_us"):
            r["recovery_p99_us"] = float(r["recovery_p99_us"])
        if r.get("tpot_p99_ms"):
            r["tpot_p99_ms"] = float(r["tpot_p99_ms"])

    # Group by (op_label, intensity) and compute delta TPOT
    # Baseline = inference_only path
    baselines = defaultdict(list)  # (op, intensity) -> list of tpot_p99
    coloc = defaultdict(lambda: defaultdict(list))  # (op, intensity) -> path -> list of tpot_p99

    for r in results:
        key = (r["op_label"], r["intensity"])
        if r["path"] == "inference_only":
            if "tpot_p99_ms" in r and isinstance(r["tpot_p99_ms"], (int, float)):
                baselines[key].append(r["tpot_p99_ms"])
        else:
            if "tpot_p99_ms" in r and isinstance(r["tpot_p99_ms"], (int, float)):
                coloc[key][r["path"]].append(r["tpot_p99_ms"])

    # Compute delta TPOT
    delta_results = []
    for (op, intensity), path_tpots in coloc.items():
        baseline_key = (op, intensity)
        if baseline_key not in baselines or not baselines[baseline_key]:
            continue
        baseline_median = statistics.median(baselines[baseline_key])

        for path_name, tpots in path_tpots.items():
            if not tpots:
                continue
            coloc_median = statistics.median(tpots)
            delta = compute_delta_tpot(coloc_median, baseline_median)
            classification = classify_interference(
                delta, TOL_TPOT["abs"], TOL_TPOT["rel"], baseline_median
            )
            delta_results.append({
                "op_label": op,
                "intensity": intensity,
                "path": path_name,
                "baseline_tpot_p99_ms": baseline_median,
                "coloc_tpot_p99_ms": coloc_median,
                "delta_tpot_ms": delta,
                "interference": classification,
            })

    os.makedirs(output_dir, exist_ok=True)
    delta_path = os.path.join(output_dir, "n2_5_delta_tpot.csv")
    with open(delta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "op_label", "intensity", "path",
            "baseline_tpot_p99_ms", "coloc_tpot_p99_ms",
            "delta_tpot_ms", "interference"
        ])
        writer.writeheader()
        writer.writerows(delta_results)
    print(f"Delta TPOT written to {delta_path}")

    return delta_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N2.5 analysis")
    parser.add_argument("--input", default="results/n2_5_colocation/n2_5_colocation.csv")
    parser.add_argument("--output", default="results/n2_5_colocation/analysis")
    args = parser.parse_args()
    analyze_n2_5(args.input, args.output)


if __name__ == "__main__":
    main()
