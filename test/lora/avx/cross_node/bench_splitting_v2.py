"""S2: Splitting re-test -- atomic vs stateful chunking vs server slicing vs fan-out.

Mechanisms:
  atomic: 1 RPC, 1 READ/WRITE, no yield (baseline)
  stateful_chunking: N RPCs, 1 READ/WRITE, client chooses boundaries
  server_slicing: 1 RPC, 1 READ/WRITE, server chooses boundaries (quantum)
  persistent_fanout: N RPCs, N READs/WRITEs, concurrent (H1 control)

Chunk/quantum sizes: 1, 2, 4 (q=8 = atomic, used for equivalence)
Compositions: 1h8l (1 heavy + 8 light), 2h0l (2 heavy, all-heavy control)
Arrival: synchronized, Poisson
"""
import argparse
import csv
import os
import socket
import statistics
import time

from bench_decomposition import (
    CELLS, RemoteSession, DecompositionConfig,
    _run_persistent, _run_per_request, _run_stateful_chunking, _run_persistent_fanout,
    _make_request, HIDDEN_DIM, INTERMEDIATE_DIM, BYTES_PER_PARAM, new_request_id,
)
from common.transport import PersistentTransport
from common.stats import trial_ci

MECHANISMS = ["atomic", "stateful_chunking", "server_slicing", "persistent_fanout"]
CHUNK_SIZES = [1, 2, 4]
CELLS_S2 = ["B1", "B2", "B3"]  # B3 for concurrent slicing
N_TRIALS = 5
N_ITERS = 50


def run_atomic(session, nm, n_iters, variant="baseline"):
    """Run atomic (no splitting). Uses the standard s4a_pooled handler."""
    # RemoteSession.run() does not accept variant -- it is set at construction.
    return session.run(nm, n_iters=n_iters, n_trials=1)


def run_server_slicing(session, nm, quantum, n_iters, variant="baseline"):
    """Run server cooperative slicing with given quantum."""
    config = DecompositionConfig(cell=session.cell, nm=nm, rank=session.rank,
                                 n_trials=1, n_iters=n_iters)
    cell_spec = CELLS[session.cell]
    act_bytes = HIDDEN_DIM * BYTES_PER_PARAM
    result_bytes = nm * INTERMEDIATE_DIM * BYTES_PER_PARAM

    sock = socket.create_connection((session.server_host, session.server_port))
    transport = PersistentTransport(sock)

    latencies = []
    segments_per_request = []

    try:
        for _ in range(n_iters):
            req_id = new_request_id()
            pool_id, gpu_t, _ = session.qp_pool.borrow()
            try:
                msg = _make_request(req_id, config, cell_spec, variant)
                msg["quantum"] = quantum
                t0 = time.perf_counter()
                response = transport.request(msg)
                t1 = time.perf_counter()
            finally:
                session.qp_pool.return_transport(pool_id, gpu_t)
            latencies.append((t1 - t0) * 1e6)
            segments_per_request.append(response.get("segments", []))
    finally:
        transport.close()

    return {"latencies_us": latencies, "segments_per_request": segments_per_request}


def run_stateful_chunking(session, nm, chunk_size, n_iters,
                          policy="contiguous", variant="baseline"):
    """Run stateful client chunking."""
    config = DecompositionConfig(cell=session.cell, nm=nm, rank=session.rank,
                                 n_trials=1, n_iters=n_iters)
    cell_spec = CELLS[session.cell]
    act_bytes = HIDDEN_DIM * BYTES_PER_PARAM
    result_bytes = nm * INTERMEDIATE_DIM * BYTES_PER_PARAM

    latencies, segments, _ = _run_stateful_chunking(
        config, cell_spec, variant, "persistent_tcp",
        session.server_host, session.server_port,
        session.qp_pool, act_bytes, result_bytes,
        chunk_size=chunk_size, policy=policy)
    return {"latencies_us": latencies, "segments_per_request": segments}


def run_persistent_fanout(session, nm, chunk_size, n_iters, variant="baseline"):
    """Run matched persistent fan-out (H1 control)."""
    config = DecompositionConfig(cell=session.cell, nm=nm, rank=session.rank,
                                 n_trials=1, n_iters=n_iters)
    cell_spec = CELLS[session.cell]
    act_bytes = HIDDEN_DIM * BYTES_PER_PARAM
    result_bytes = nm * INTERMEDIATE_DIM * BYTES_PER_PARAM

    latencies, segments, _ = _run_persistent_fanout(
        config, cell_spec, variant,
        session.server_host, session.server_port,
        session.qp_pool, act_bytes, result_bytes,
        chunk_size=chunk_size)
    return {"latencies_us": latencies, "segments_per_request": segments}


def run_s2(output_dir="results/s2_splitting", n_trials=None, n_iters=None):
    n_trials = n_trials or N_TRIALS
    n_iters = n_iters or N_ITERS
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for cell in CELLS_S2:
        cell_spec = CELLS[cell]
        for mechanism in MECHANISMS:
            # Determine chunk sizes for this mechanism
            if mechanism == "atomic":
                sizes = [8]  # atomic = no splitting, only NM=8
            else:
                sizes = CHUNK_SIZES

            for size in sizes:
                for nm in [8]:  # S2 focuses on NM=8 (the heavy request)
                    trial_p50s = []
                    trial_p99s = []

                    for trial in range(n_trials):
                        try:
                            with RemoteSession(cell_spec, cell=cell,
                                               variant="baseline") as session:
                                if mechanism == "atomic":
                                    result = run_atomic(session, nm, n_iters)
                                elif mechanism == "server_slicing":
                                    result = run_server_slicing(session, nm, size, n_iters)
                                elif mechanism == "stateful_chunking":
                                    result = run_stateful_chunking(session, nm, size, n_iters)
                                elif mechanism == "persistent_fanout":
                                    result = run_persistent_fanout(session, nm, size, n_iters)

                                lats = result.get("latencies_us", [])
                                if lats:
                                    trial_p50s.append(statistics.median(lats))
                                    trial_p99s.append(sorted(lats)[int(0.99 * len(lats)) - 1])
                        except Exception as e:
                            print(f"  {cell} {mechanism} size={size} trial {trial}: skipped ({e})")
                            continue

                    if not trial_p50s:
                        print(f"{cell} {mechanism} size={size}: no successful trials")
                        continue

                    p50_lo, p50_hi = trial_ci(trial_p50s)
                    p99_lo, p99_hi = trial_ci(trial_p99s)
                    rows.append({
                        "cell": cell, "mechanism": mechanism,
                        "chunk_size": size, "nm": nm, "variant": "eager",
                        "p50_median": statistics.median(trial_p50s),
                        "p50_ci_lo": p50_lo, "p50_ci_hi": p50_hi,
                        "p99_median": statistics.median(trial_p99s),
                        "p99_ci_lo": p99_lo, "p99_ci_hi": p99_hi,
                    })
                    print(f"{cell} {mechanism} size={size}: "
                          f"P50={statistics.median(trial_p50s):.0f}us "
                          f"P99={statistics.median(trial_p99s):.0f}us")

        # --- CUDA-graph sub-study (Task 5) ---
        # Compare eager vs graph for atomic NM=8.
        # Graph+slicing requires server-side per-quantum graph capture (future work).
        for variant in ["baseline", "cuda_graph"]:
            trial_p50s = []
            trial_p99s = []
            for trial in range(n_trials):
                try:
                    with RemoteSession(cell_spec, cell=cell, variant=variant) as session:
                        result = run_atomic(session, 8, n_iters)
                        lats = result.get("latencies_us", [])
                        if lats:
                            trial_p50s.append(statistics.median(lats))
                            trial_p99s.append(sorted(lats)[int(0.99 * len(lats)) - 1])
                except Exception as e:
                    print(f"  {cell} graph_substudy {variant} trial {trial}: skipped ({e})")
                    continue
            if trial_p50s:
                p50_lo, p50_hi = trial_ci(trial_p50s)
                p99_lo, p99_hi = trial_ci(trial_p99s)
                rows.append({
                    "cell": cell, "mechanism": "atomic_graph_substudy",
                    "chunk_size": 8, "nm": 8, "variant": variant,
                    "p50_median": statistics.median(trial_p50s),
                    "p50_ci_lo": p50_lo, "p50_ci_hi": p50_hi,
                    "p99_median": statistics.median(trial_p99s),
                    "p99_ci_lo": p99_lo, "p99_ci_hi": p99_hi,
                })
                print(f"{cell} graph_substudy {variant}: "
                      f"P50={statistics.median(trial_p50s):.0f}us")

    if not rows:
        print("No data collected.")
        return

    csv_path = os.path.join(output_dir, "splitting.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="S2 splitting re-test")
    parser.add_argument("--output", default="results/s2_splitting")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    run_s2(args.output, args.trials, args.iters)


if __name__ == "__main__":
    main()
