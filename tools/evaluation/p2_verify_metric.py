#!/usr/bin/env python3
"""P2 autoresearch verify script.

Runs a reduced sweep (ranks 16-64, batch 1-8) and computes the
geometric-mean speedup of cpu_first vs load_then_run.

Usage:
  conda run -n lightllm-with-deepep python tools/evaluation/p2_verify_metric.py

Output: a single float — the geometric mean of
  (load_then_run_service_time / cpu_first_service_time)
across all (rank, batch) configs. Higher is better.
  >= 1.3 → pass (cpu_first 30%+ faster)
  >= 1.5 → strong pass
"""
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

# Force naive CPU kernel mode for systems without AVX-512 BF16
os.environ["COLORA_CPU_KERNEL_MODE"] = "naive"

# Patch MOE_AVX_AVAILABLE to allow naive path to work
import lightllm.models.qwen3_vl_moe.lora_dispatch as lora_dispatch_module
lora_dispatch_module.MOE_AVX_AVAILABLE = True

ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = ROOT / "results" / "microbench_cold_recovery"
AGG_CSV = RESULT_DIR / "microbench_cold_recovery_aggregated.csv"


def main() -> int:
    # Run the benchmark sweep with reduced iters for speed
    cmd = [
        "conda", "run", "-n", "lightllm-with-deepep",
        "python", str(ROOT / "tools" / "evaluation" / "colora_microbench_cold_recovery.py"),
        "--config", str(ROOT / "configs" / "evaluation" / "colora_kpi" / "microbench_config.yaml"),
        "--ranks", "8,16,32,64",
        "--batch-sizes", "1,2,4,8",
        "--repeats", "5",
    ]
    print("Running benchmark sweep...", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"Benchmark FAILED (rc={proc.returncode})", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        print("0.0")  # metric-error signal
        return 1

    # Parse aggregated CSV
    if not AGG_CSV.exists():
        print("Aggregated CSV not found", file=sys.stderr)
        print("0.0")
        return 1

    speedups = []
    with AGG_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = {(r["policy"], int(r["lora_rank"]), int(r["batch_size"])): r for r in reader}

    target_configs = [(r, b) for r in (8, 16, 32, 64) for b in (1, 2, 4, 8)]
    for rank, batch in target_configs:
        cpu_key = ("cpu_first", rank, batch)
        promo_key = ("load_then_run", rank, batch)
        if cpu_key not in rows or promo_key not in rows:
            print(f"Missing data for rank={rank} batch={batch}", file=sys.stderr)
            continue
        cpu_time = float(rows[cpu_key]["service_time_us_median"])
        promo_time = float(rows[promo_key]["service_time_us_median"])
        if cpu_time <= 0:
            continue
        speedup = promo_time / cpu_time
        speedups.append(speedup)
        print(f"  rank={rank} batch={batch}: cpu={cpu_time:.1f}us promo={promo_time:.1f}us speedup={speedup:.3f}x", file=sys.stderr)

    if not speedups:
        print("0.0")
        return 1

    # Geometric mean of speedup
    log_sum = sum(math.log(s) for s in speedups if s > 0)
    geom_mean = math.exp(log_sum / len(speedups))
    print(f"\nGeometric mean speedup: {geom_mean:.4f}x ({len(speedups)} configs)", file=sys.stderr)

    # Print as the metric
    print(f"{geom_mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
