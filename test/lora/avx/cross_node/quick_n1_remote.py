"""Quick N1 comparison: cpu_first vs load_then_run vs remote_improved vs oracle.

Uses the existing S1-S6 RemoteSession for remote_improved (B3 cell).
All paths use H=2048, I=1536 to match the remote infrastructure.
"""
import torch
import statistics
import time
import sys

# Add parent dir to path
sys.path.insert(0, '.')

from common.northstar_paths import cpu_first_recovery, load_then_run_recovery, oracle_recovery
from common.forced_cold import ForcedColdWeightPool
from common.instrumentation import NorthstarTimeline

H = 2048
I_DIM = 1536  # Match INTERMEDIATE_DIM from bench_decomposition.py

CELLS = [(16, 1), (64, 1), (64, 8), (128, 8), (256, 8)]
N_REQUESTS = 50

def run_local_path(path_name, R, NM, pool, n_requests):
    """Run n_requests on a local path (cpu_first, load_then_run, oracle)."""
    latencies = []
    for i in range(n_requests):
        activation = torch.randn(NM, H, dtype=torch.float16, device="cuda")
        if path_name == "cpu_first":
            weights = pool.get_batch(i, NM)
            _, tl = cpu_first_recovery(activation, weights, R, H, I_DIM, NM, num_cores=1)
        elif path_name == "load_then_run":
            weights = pool.get_batch(i, NM)
            _, tl = load_then_run_recovery(activation, weights, R, H, I_DIM, NM)
        elif path_name == "oracle":
            weights = pool.get_oracle_batch(i, NM, device="cuda")
            _, tl = oracle_recovery(activation, weights, R, H, I_DIM, NM)
        latencies.append(tl.l_recovery_us())
    return latencies

def run_remote_path(session, R, NM, n_requests):
    """Run n_requests on the remote_improved path via RemoteSession."""
    result = session.run(nm=NM, n_iters=n_requests, rank=R)
    return result["latencies_us"]

def main():
    print("=" * 100)
    print("N1 QUICK COMPARISON: cpu_first vs load_then_run vs remote_improved vs oracle")
    print(f"H={H}, I={I_DIM}, {N_REQUESTS} requests per cell, 1 trial")
    print("=" * 100)

    # Start remote session
    print("\nStarting remote server on UM251...")
    from bench_decomposition import RemoteSession, CELLS as BENCH_CELLS
    session = RemoteSession(
        cell_spec=BENCH_CELLS["B3"],
        cell="B3",
        variant="baseline",
        max_nm=16,
        rank=64,  # will be overridden per cell
        scheduling_policy="fifo",
        heavy_cap=2,
        heavy_threshold=4,
    )

    try:
        session.__enter__()
        print("Remote server started successfully!")

        # Header
        print(f"\n{'R':>5} {'NM':>4} {'cpu_first':>12} {'load_then':>12} {'remote_imp':>12} {'oracle':>12} {'winner':>12}")
        print("-" * 80)

        for R, NM in CELLS:
            pool = ForcedColdWeightPool(R, H, I_DIM, pool_size=N_REQUESTS * 2, seed=42)

            results = {}

            # Local paths
            for path in ["cpu_first", "load_then_run", "oracle"]:
                lats = run_local_path(path, R, NM, pool, N_REQUESTS)
                results[path] = statistics.median(lats)

            # Remote path
            try:
                lats = run_remote_path(session, R, NM, N_REQUESTS)
                results["remote_improved"] = statistics.median(lats) if lats else 0
            except Exception as e:
                print(f"  Remote error for R={R} NM={NM}: {e}")
                results["remote_improved"] = float('inf')

            winner = min(results.keys(), key=lambda p: results[p])

            print(f"{R:>5} {NM:>4} {results['cpu_first']:>12.1f} {results['load_then_run']:>12.1f} {results['remote_improved']:>12.1f} {results['oracle']:>12.1f} {winner:>12}")

    finally:
        session.__exit__(None, None, None)
        print("\nRemote server stopped.")

if __name__ == "__main__":
    main()
