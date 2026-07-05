"""S4: Heavy-lane re-test -- H sweep, Pareto frontier, class-aware vs global cap.

H in {1,2,4,8} (H=8 = no heavy sub-cap).
Poisson mixtures: 10%, 25%, 50%, 100% heavy.
Cells: B3 (persistent TCP, python_executor, conc=8).
Stage C: global cap K in {1,2,4} (no class distinction) vs heavy-lane.
"""
import argparse, csv, os, statistics, time, socket, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from bench_decomposition import (
    CELLS, RemoteSession, DecompositionConfig,
    _make_sjf_request, HIDDEN_DIM, INTERMEDIATE_DIM, BYTES_PER_PARAM,
)
from common.protocol import new_request_id
from common.transport import PersistentTransport
from common.stats import trial_ci

H_VALUES = [1, 2, 4, 8]
MIXTURES = {"10pct": 0.10, "25pct": 0.25, "50pct": 0.50, "all_heavy": 1.0}
GLOBAL_CAPS = [1, 2, 4]  # Stage C: global cap without heavy sub-cap
CELLS_S4 = ["B3"]
N_TRIALS = 3
N_ITERS = 50  # requests per trial
HEAVY_THRESHOLD = 4  # NM >= 4 is heavy


def run_batch(session, n_requests, heavy_frac, n_iters):
    """Run n_iters batches of n_requests with given heavy fraction.

    Each batch: generate n_requests with heavy_frac heavy (NM=8) and rest light (NM=1).
    Submit all concurrently via ThreadPoolExecutor(max_workers=conc).
    Returns per-class latencies.
    """
    cell_spec = CELLS[session.cell]
    rank = session.rank
    conc = cell_spec["conc"]

    sock = socket.create_connection((session.server_host, session.server_port))
    sock.settimeout(120.0)
    transport = PersistentTransport(sock)

    light_lats = []
    heavy_lats = []

    try:
        for _ in range(n_iters):
            # Generate batch
            batch = []
            for i in range(n_requests):
                nm = 8 if random.random() < heavy_frac else 1
                batch.append(nm)

            def _do_one(nm):
                req_id = new_request_id()
                pool_id, gpu_t, _ = session.qp_pool.borrow()
                try:
                    msg = _make_sjf_request(req_id, nm, rank, cell_spec, "baseline")
                    t0 = time.perf_counter()
                    response = transport.request(msg)
                    t1 = time.perf_counter()
                finally:
                    session.qp_pool.return_transport(pool_id, gpu_t)
                return nm, (t1 - t0) * 1e6

            with ThreadPoolExecutor(max_workers=max(1, conc)) as executor:
                futures = [executor.submit(_do_one, nm) for nm in batch]
                for fut in as_completed(futures):
                    nm, e2e = fut.result()
                    if nm >= HEAVY_THRESHOLD:
                        heavy_lats.append(e2e)
                    else:
                        light_lats.append(e2e)
    finally:
        transport.close()

    return light_lats, heavy_lats


def _p99(vals):
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    return sorted(vals)[max(0, int(0.99 * len(vals)) - 1)]


def run_s4(output_dir="results/s4_heavy_lane", n_trials=None, n_iters=None):
    n_trials = n_trials or N_TRIALS
    n_iters = n_iters or N_ITERS
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    n_requests = 16  # requests per batch (2x conc for queueing pressure)

    random.seed(42)  # reproducible batch generation

    for cell in CELLS_S4:
        cell_spec = CELLS[cell]

        # --- Heavy-lane sweep: H in {1,2,4,8} ---
        for H in H_VALUES:
            for mix_name, heavy_frac in MIXTURES.items():
                trial_light_p99s = []
                trial_heavy_p99s = []
                trial_p50s = []

                for trial in range(n_trials):
                    try:
                        with RemoteSession(cell_spec, cell=cell, variant="baseline",
                                           heavy_cap=H,
                                           heavy_threshold=HEAVY_THRESHOLD) as session:
                            light_lats, heavy_lats = run_batch(
                                session, n_requests, heavy_frac, n_iters)
                            if light_lats:
                                trial_light_p99s.append(_p99(light_lats))
                            if heavy_lats:
                                trial_heavy_p99s.append(_p99(heavy_lats))
                            all_lats = light_lats + heavy_lats
                            if all_lats:
                                trial_p50s.append(statistics.median(all_lats))
                    except Exception as e:
                        print(f"  {cell} H={H} {mix_name} trial {trial}: skipped ({e})")
                        continue

                if not trial_p50s:
                    print(f"{cell} H={H} {mix_name}: no successful trials")
                    continue

                row = {
                    "cell": cell, "policy": "heavy_lane", "H": H,
                    "mixture": mix_name, "heavy_frac": heavy_frac,
                    "p50_median": statistics.median(trial_p50s),
                    "light_p99": statistics.median(trial_light_p99s) if trial_light_p99s else float("nan"),
                    "heavy_p99": statistics.median(trial_heavy_p99s) if trial_heavy_p99s else float("nan"),
                }
                rows.append(row)
                print(f"{cell} heavy_lane H={H} {mix_name}: "
                      f"P50={row['p50_median']:.0f}us "
                      f"light_P99={row['light_p99']:.0f}us "
                      f"heavy_P99={row['heavy_p99']:.0f}us")

        # --- Stage C: global cap K in {1,2,4} (no heavy sub-cap) ---
        for K in GLOBAL_CAPS:
            for mix_name, heavy_frac in [("25pct", 0.25)]:  # one representative mixture
                trial_light_p99s = []
                trial_heavy_p99s = []
                trial_p50s = []

                for trial in range(n_trials):
                    try:
                        # Global cap: heavy_threshold=0 means ALL jobs are "heavy"
                        # so heavy_cap=K becomes the total effective cap
                        with RemoteSession(cell_spec, cell=cell, variant="baseline",
                                           heavy_cap=K,
                                           heavy_threshold=0) as session:
                            light_lats, heavy_lats = run_batch(
                                session, n_requests, heavy_frac, n_iters)
                            if light_lats:
                                trial_light_p99s.append(_p99(light_lats))
                            if heavy_lats:
                                trial_heavy_p99s.append(_p99(heavy_lats))
                            all_lats = light_lats + heavy_lats
                            if all_lats:
                                trial_p50s.append(statistics.median(all_lats))
                    except Exception as e:
                        print(f"  {cell} global_cap K={K} {mix_name} trial {trial}: skipped ({e})")
                        continue

                if not trial_p50s:
                    print(f"{cell} global_cap K={K} {mix_name}: no successful trials")
                    continue

                row = {
                    "cell": cell, "policy": "global_cap", "H": K,
                    "mixture": mix_name, "heavy_frac": heavy_frac,
                    "p50_median": statistics.median(trial_p50s),
                    "light_p99": statistics.median(trial_light_p99s) if trial_light_p99s else float("nan"),
                    "heavy_p99": statistics.median(trial_heavy_p99s) if trial_heavy_p99s else float("nan"),
                }
                rows.append(row)
                print(f"{cell} global_cap K={K} {mix_name}: "
                      f"P50={row['p50_median']:.0f}us "
                      f"light_P99={row['light_p99']:.0f}us "
                      f"heavy_P99={row['heavy_p99']:.0f}us")

    if not rows:
        print("No data collected.")
        return

    csv_path = os.path.join(output_dir, "heavy_lane.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="S4 heavy-lane re-test")
    parser.add_argument("--output", default="results/s4_heavy_lane")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    run_s4(args.output, args.trials, args.iters)


if __name__ == "__main__":
    main()
