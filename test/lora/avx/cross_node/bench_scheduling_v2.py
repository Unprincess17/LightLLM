"""S3: Scheduling study -- FIFO / Client-SJF / Server-SJF.

3 policies x 2 cells (B2, B3) x 3 compositions.
Primary: atomic, non-sliced. Synchronized mode (exploratory).

Policies:
  fifo:       standard dispatch, server FIFO
  client_sjf: client sorts batch by NM ascending, server FIFO
  server_sjf: client FIFO, server PriorityDispatcher (shortest predicted first)

Compositions:
  1h8l:    1x NM=8 + 8x NM=1 (sparse heavy)
  2h0l:    2x NM=8 (all-heavy control)
  medium:  1x NM=8 + 2x NM=4 + 8x NM=1

Factorial analysis:
  Delta_sched(c) = P99(FIFO,c) - P99(Server-SJF,c) at c in {1, N}
  Delta_conc(policy) = P99(policy,N) - P99(policy,1)
  Interaction = Delta_sched(N) - Delta_sched(1)
  Per class (light, heavy), using common-load traces only.
"""
import argparse
import csv
import os
import socket
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from bench_decomposition import (
    CELLS, RemoteSession, DecompositionConfig,
    _make_sjf_request, HIDDEN_DIM, INTERMEDIATE_DIM, BYTES_PER_PARAM,
)
from common.protocol import new_request_id
from common.transport import PersistentTransport
from common.stats import trial_ci
from common.scheduling_analysis import (
    compute_priority_fidelity, compute_inversion_count, compute_order_correlation,
)

POLICIES = ["fifo", "client_sjf", "server_sjf"]
CELLS_S3 = ["B2", "B3"]
COMPOSITIONS = {
    "1h8l":   [8, 1, 1, 1, 1, 1, 1, 1, 1],
    "2h0l":   [8, 8],
    "medium": [8, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1],
}
N_TRIALS = 5
N_ITERS = 10  # iterations per trial (each iteration = one full composition batch)


# ---------------------------------------------------------------------------
# Concurrent policy runner
# ---------------------------------------------------------------------------

def run_policy(session, composition, n_iters, policy="fifo", variant="baseline"):
    """Run one scheduling policy with concurrent submission.

    For B2 (conc=1): sequential (max_workers=1).
    For B3 (conc=8): concurrent (max_workers=8).

    All requests in one iteration are submitted simultaneously via
    ThreadPoolExecutor. PersistentTransport supports concurrent request()
    calls via request-ID multiplexing.

    policy:
      fifo:       composition as-is, server FIFO
      client_sjf: composition sorted by NM ascending, server FIFO
      server_sjf: composition as-is, server PriorityDispatcher
    """
    cell_spec = CELLS[session.cell]
    rank = session.rank
    conc = cell_spec["conc"]

    # Client-SJF: sort composition so shorter jobs are submitted first
    if policy == "client_sjf":
        composition = sorted(composition)

    latencies = []
    scheduling_data = []

    sock = socket.create_connection((session.server_host, session.server_port))
    sock.settimeout(120.0)
    transport = PersistentTransport(sock)

    try:
        for _ in range(n_iters):
            # Assign req_ids in order (controls submission order)
            req_specs = []
            for nm in composition:
                req_id = new_request_id()
                req_specs.append((req_id, nm))

            # Submit all requests concurrently
            def _do_one(req_id, nm):
                pool_id, gpu_t, _ = session.qp_pool.borrow()
                try:
                    msg = _make_sjf_request(req_id, nm, rank, cell_spec, variant)
                    t0 = time.perf_counter()
                    response = transport.request(msg)
                    t1 = time.perf_counter()
                finally:
                    session.qp_pool.return_transport(pool_id, gpu_t)
                e2e_us = (t1 - t0) * 1e6
                return {
                    "req_id": req_id,
                    "nm": nm,
                    "submission_seq": req_id,
                    "handler_start_seq": response.get("handler_start_seq", 0),
                    "completion_seq": response.get("completion_seq", 0),
                    "e2e_us": e2e_us,
                }

            with ThreadPoolExecutor(max_workers=max(1, conc)) as executor:
                futures = {executor.submit(_do_one, rid, nm): (rid, nm)
                           for rid, nm in req_specs}
                iter_data = []
                for fut in as_completed(futures):
                    data = fut.result()
                    iter_data.append(data)
                    latencies.append(data["e2e_us"])

            # Sort by req_id to get submission order
            iter_data.sort(key=lambda d: d["req_id"])
            scheduling_data.extend(iter_data)
    finally:
        transport.close()

    return latencies, scheduling_data


def run_fifo(session, composition, n_iters, variant="baseline"):
    """FIFO: send in arrival order, server FIFO."""
    return run_policy(session, composition, n_iters, policy="fifo", variant=variant)

def run_client_sjf(session, composition, n_iters, variant="baseline"):
    """Client-SJF: sort composition by NM ascending, then send concurrently."""
    return run_policy(session, composition, n_iters, policy="client_sjf", variant=variant)

def run_server_sjf(session, composition, n_iters, variant="baseline"):
    """Server-SJF: send in arrival order, server PriorityDispatcher reorders."""
    return run_policy(session, composition, n_iters, policy="server_sjf", variant=variant)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _p99(values):
    """Simple P99: index into sorted list, fallback for small samples."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    idx = int(0.99 * len(values)) - 1
    idx = max(0, min(idx, len(values) - 1))
    return sorted(values)[idx]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_s3(output_dir="results/s3_scheduling", n_trials=None, n_iters=None):
    n_trials = n_trials or N_TRIALS
    n_iters = n_iters or N_ITERS
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for cell in CELLS_S3:
        cell_spec = CELLS[cell]
        for policy in POLICIES:
            for comp_name, composition in COMPOSITIONS.items():
                trial_light_p99s = []
                trial_heavy_p99s = []
                trial_p50s = []
                trial_fidelities = []
                trial_inversions = []

                for trial in range(n_trials):
                    try:
                        # Configure scheduling policy for this session
                        scheduling_policy = None
                        s_hat = None
                        if policy == "server_sjf":
                            scheduling_policy = "server_sjf"
                            s_hat = {1: 1.0, 4: 4.0, 8: 8.0}

                        with RemoteSession(
                            cell_spec, cell=cell, variant="baseline",
                            scheduling_policy=scheduling_policy,
                            s_hat=s_hat,
                        ) as session:
                            if policy == "fifo":
                                lats, sdata = run_fifo(session, composition, n_iters)
                            elif policy == "client_sjf":
                                lats, sdata = run_client_sjf(session, composition, n_iters)
                            elif policy == "server_sjf":
                                lats, sdata = run_server_sjf(session, composition, n_iters)

                            # Split by class
                            light_lats = [d["e2e_us"] for d in sdata if d["nm"] == 1]
                            heavy_lats = [d["e2e_us"] for d in sdata if d["nm"] >= 4]
                            all_lats = [d["e2e_us"] for d in sdata]

                            if light_lats:
                                trial_light_p99s.append(_p99(light_lats))
                            if heavy_lats:
                                trial_heavy_p99s.append(_p99(heavy_lats))
                            if all_lats:
                                trial_p50s.append(statistics.median(all_lats))

                            # Scheduling analysis
                            fidelity = compute_priority_fidelity(sdata)
                            inversions = compute_inversion_count(sdata)
                            trial_fidelities.append(fidelity)
                            trial_inversions.append(inversions)

                    except Exception as e:
                        print(f"  {cell} {policy} {comp_name} trial {trial}: skipped ({e})")
                        continue

                if not trial_p50s:
                    print(f"{cell} {policy} {comp_name}: no successful trials")
                    continue

                row = {
                    "cell": cell,
                    "policy": policy,
                    "composition": comp_name,
                    "p50_median": statistics.median(trial_p50s),
                    "light_p99_median": statistics.median(trial_light_p99s) if trial_light_p99s else float("nan"),
                    "heavy_p99_median": statistics.median(trial_heavy_p99s) if trial_heavy_p99s else float("nan"),
                    "priority_fidelity": statistics.median(trial_fidelities) if trial_fidelities else float("nan"),
                    "inversion_count": statistics.median(trial_inversions) if trial_inversions else float("nan"),
                }
                rows.append(row)
                print(f"{cell} {policy} {comp_name}: "
                      f"P50={row['p50_median']:.0f}us "
                      f"light_P99={row['light_p99_median']:.0f}us "
                      f"heavy_P99={row['heavy_p99_median']:.0f}us "
                      f"fidelity={row['priority_fidelity']:.2f} "
                      f"inversions={row['inversion_count']:.0f}")

    if not rows:
        print("No data collected.")
        return

    csv_path = os.path.join(output_dir, "scheduling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    # Factorial analysis (exploratory)
    _factorial_analysis(rows, output_dir)


def _factorial_analysis(rows, output_dir):
    """Compute factorial deltas: scheduling effect, concurrency effect, interaction."""
    # Index rows by (cell, policy, composition)
    idx = {(r["cell"], r["policy"], r["composition"]): r for r in rows}

    analysis_rows = []
    for comp_name in COMPOSITIONS:
        for cell in CELLS_S3:
            fifo = idx.get((cell, "fifo", comp_name))
            ssjf = idx.get((cell, "server_sjf", comp_name))
            if fifo and ssjf:
                # Scheduling effect: how much Server-SJF improves over FIFO
                d_light = fifo["light_p99_median"] - ssjf["light_p99_median"]
                d_heavy = fifo["heavy_p99_median"] - ssjf["heavy_p99_median"]
                analysis_rows.append({
                    "cell": cell,
                    "composition": comp_name,
                    "delta_light_p99": d_light,
                    "delta_heavy_p99": d_heavy,
                    "fifo_light_p99": fifo["light_p99_median"],
                    "ssjf_light_p99": ssjf["light_p99_median"],
                    "fifo_heavy_p99": fifo["heavy_p99_median"],
                    "ssjf_heavy_p99": ssjf["heavy_p99_median"],
                })

    # Concurrency effect: B3 vs B2 for each policy
    for policy in POLICIES:
        for comp_name in COMPOSITIONS:
            b2 = idx.get(("B2", policy, comp_name))
            b3 = idx.get(("B3", policy, comp_name))
            if b2 and b3:
                analysis_rows.append({
                    "cell": "B3-vs-B2",
                    "composition": comp_name,
                    "policy": policy,
                    "delta_conc_light": b3["light_p99_median"] - b2["light_p99_median"],
                    "delta_conc_heavy": b3["heavy_p99_median"] - b2["heavy_p99_median"],
                })

    if analysis_rows:
        analysis_path = os.path.join(output_dir, "factorial_analysis.csv")
        with open(analysis_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(analysis_rows[0].keys()))
            w.writeheader()
            w.writerows(analysis_rows)
        print(f"Wrote {analysis_path} ({len(analysis_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="S3 scheduling study")
    parser.add_argument("--output", default="results/s3_scheduling")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    run_s3(args.output, args.trials, args.iters)


if __name__ == "__main__":
    main()
