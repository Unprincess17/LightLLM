"""S6: Per-class capacity recalibration -- true open-loop, bracketed C, drain protocol.

5 workloads: NM=1-only, NM=8-only, 10% heavy, 25% heavy, 50% heavy.
Per-(cell, mixture) empirical capacity via bracketed binary search (+/-5%).
"""
import argparse, csv, os, statistics, time, socket, threading, random, sys
from bench_decomposition import (
    CELLS, RemoteSession, DecompositionConfig,
    _make_sjf_request, HIDDEN_DIM, INTERMEDIATE_DIM, BYTES_PER_PARAM, new_request_id,
    _start_concurrent_server, _stop_concurrent_server, _setup_qp_pool, _teardown_qp_pool,
    DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT, DEFAULT_SSH_HOST, DEFAULT_LOCAL_IP, DEFAULT_BASE_CONTROL_PORT,
)
from common.transport import PersistentTransport
from common.load_generator import generate_poisson_trace, Trace, OpenLoopRunner
from common.stats import trial_ci
from concurrent.futures import ThreadPoolExecutor

WORKLOADS = {
    "nm1_only": {"heavy_frac": 0.0},
    "nm8_only": {"heavy_frac": 1.0},
    "10pct":    {"heavy_frac": 0.10},
    "25pct":    {"heavy_frac": 0.25},
    "50pct":    {"heavy_frac": 0.50},
}
CELLS_S6 = ["B3"]
N_TRIALS = 2  # capacity runs are long; fewer trials
DURATION_S = 10  # seconds per probe (short for first run)
DRAIN_TIMEOUT_S = 5.0
LATENCY_P99_THRESHOLD_US = 200_000  # 200ms -- if P99 exceeds this, system is overloaded
SEARCH_HI = 5000.0  # upper bound for capacity search


def _p99(vals):
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    return sorted(vals)[max(0, int(0.99 * len(vals)) - 1)]


class ServerSink:
    """Sink that sends requests to the live server via PersistentTransport.

    Thread-safe: multiple consumer threads call submit() concurrently.
    """
    def __init__(self, transport, qp_pool, cell_spec, rank, heavy_frac):
        self.transport = transport
        self.qp_pool = qp_pool
        self.cell_spec = cell_spec
        self.rank = rank
        self.heavy_frac = heavy_frac
        self._latencies = []
        self._class_lats = {"light": [], "heavy": []}
        self._lock = threading.Lock()
        self._rng = random.Random(42)

    def reset(self):
        """Clear accumulated latencies.  Called before each probe so that
        bracket probes do not contaminate the final measurement."""
        with self._lock:
            self._latencies.clear()
            self._class_lats["light"].clear()
            self._class_lats["heavy"].clear()

    def submit(self, event):
        """Called by OpenLoopRunner consumer. event has arrival_time, job_class, req_id."""
        # Determine NM based on heavy_frac (override job_class if needed)
        nm = 8 if self._rng.random() < self.heavy_frac else 1
        pool_id, gpu_t, _ = self.qp_pool.borrow()
        try:
            msg = _make_sjf_request(event.req_id, nm, self.rank, self.cell_spec, "baseline")
            t0 = time.perf_counter()
            response = self.transport.request(msg)
            t1 = time.perf_counter()
        finally:
            self.qp_pool.return_transport(pool_id, gpu_t)
        e2e_us = (t1 - t0) * 1e6
        cls = "heavy" if nm >= 4 else "light"
        with self._lock:
            self._latencies.append(e2e_us)
            self._class_lats[cls].append(e2e_us)

    def get_latencies(self):
        with self._lock:
            return list(self._latencies)

    def get_class_latencies(self):
        with self._lock:
            return {k: list(v) for k, v in self._class_lats.items()}


def run_at_rate(rate, duration_s, sink, runner_kwargs):
    """Run open-loop at given rate for duration_s seconds. Returns (counters, latencies)."""
    sink.reset()
    trace = generate_poisson_trace(lam=rate, duration_s=duration_s, seed=42,
                                   classes=["light", "heavy"],
                                   heavy_frac=sink.heavy_frac)
    runner = OpenLoopRunner(trace, sink, ingress_capacity=len(trace) + 1000,
                            drain_timeout_s=DRAIN_TIMEOUT_S, **runner_kwargs)
    counters = runner.run()
    return counters, sink.get_latencies()


def check_stability(counters, latencies=None):
    """Check if the run was stable.

    Besides the counter-based checks (generated~=completed, low unfinished),
    also checks P99 latency if provided.  A system that completes all requests
    during drain but has exploded tail latency is NOT stable.
    """
    if counters.generated == 0:
        return False
    # generated ~= admitted (within 1%)
    if counters.c0_inserted < counters.generated * 0.99:
        return False
    # admitted ~= completed (within 1%)
    if counters.completed < counters.c0_inserted * 0.99:
        return False
    # rejection + timeout <= 1%
    total = counters.generated
    if total > 0 and (counters.rejected + counters.timed_out) / total > 0.01:
        return False
    # unfinished should be small (< 5% of generated)
    if counters.unfinished > total * 0.05:
        return False
    # Latency tail check: P99 must be below threshold
    if latencies and len(latencies) > 10:
        p99 = _p99(latencies)
        if p99 > LATENCY_P99_THRESHOLD_US:
            return False
    return True


def bracket_capacity(sink, runner_kwargs, duration_s=10):
    """Binary search for the highest stable offered rate. Returns (c_low, c_high)."""
    lo, hi = 10.0, SEARCH_HI
    c_low = lo
    c_high = hi

    # Phase 1: geometric probe (double until unstable)
    rate = lo
    while rate <= hi:
        counters, lats = run_at_rate(rate, duration_s, sink, runner_kwargs)
        stable = check_stability(counters, lats)
        print(f"    probe rate={rate:.0f}: gen={counters.generated} comp={counters.completed} "
              f"unfinished={counters.unfinished} stable={stable}", flush=True)
        if stable:
            c_low = rate
            rate *= 2
        else:
            c_high = rate
            break
    else:
        c_high = hi

    # Phase 2: binary search to +/-5%
    while (c_high - c_low) / max(c_low, 1) > 0.05:
        mid = (c_low + c_high) / 2
        counters, lats = run_at_rate(mid, duration_s, sink, runner_kwargs)
        stable = check_stability(counters, lats)
        print(f"    binary rate={mid:.0f}: gen={counters.generated} comp={counters.completed} "
              f"unfinished={counters.unfinished} stable={stable}", flush=True)
        if stable:
            c_low = mid
        else:
            c_high = mid

    return c_low, c_high


def run_s6(output_dir="results/s6_capacity", n_trials=None):
    n_trials = n_trials or N_TRIALS
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for cell in CELLS_S6:
        cell_spec = CELLS[cell]

        for wl_name, wl in WORKLOADS.items():
            print(f"\n=== {cell} {wl_name} (heavy_frac={wl['heavy_frac']}) ===", flush=True)

            # Start server + QP pool
            _start_concurrent_server(host=DEFAULT_SSH_HOST, listen_ip=DEFAULT_SERVER_HOST,
                                      port=DEFAULT_SERVER_PORT)
            try:
                config = DecompositionConfig(cell=cell, nm=8, rank=64)
                act_bytes = HIDDEN_DIM * 2
                result_bytes = 8 * INTERMEDIATE_DIM * 2
                gpu_buffer_bytes = act_bytes + result_bytes
                qp_pool = _setup_qp_pool(config, cell_spec, DEFAULT_SERVER_HOST,
                                         DEFAULT_SERVER_PORT, DEFAULT_LOCAL_IP,
                                         DEFAULT_BASE_CONTROL_PORT, gpu_buffer_bytes)
                try:
                    sock = socket.create_connection((DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT))
                    sock.settimeout(120.0)
                    transport = PersistentTransport(sock)

                    sink = ServerSink(transport, qp_pool, cell_spec, 64, wl["heavy_frac"])
                    runner_kwargs = {"n_consumers": cell_spec["conc"]}

                    # Bracketed capacity search
                    c_low, c_high = bracket_capacity(sink, runner_kwargs, duration_s=DURATION_S)
                    c_est = (c_low + c_high) / 2
                    print(f"  Capacity: C_low={c_low:.0f} C_high={c_high:.0f} C_est={c_est:.0f} req/s", flush=True)

                    # Run at C_low for final P99
                    counters, lats = run_at_rate(c_low, DURATION_S, sink, runner_kwargs)
                    class_lats = sink.get_class_latencies()

                    light_p99 = _p99(class_lats.get("light", []))
                    heavy_p99 = _p99(class_lats.get("heavy", []))

                    rows.append({
                        "cell": cell, "workload": wl_name,
                        "heavy_frac": wl["heavy_frac"],
                        "c_low": c_low, "c_high": c_high, "c_est": c_est,
                        "generated": counters.generated, "completed": counters.completed,
                        "unfinished": counters.unfinished,
                        "light_p99_us": light_p99,
                        "heavy_p99_us": heavy_p99,
                        "p50_us": statistics.median(lats) if lats else float("nan"),
                    })
                    print(f"  P50={statistics.median(lats):.0f}us "
                          f"light_P99={light_p99:.0f}us heavy_P99={heavy_p99:.0f}us", flush=True)

                    transport.close()
                finally:
                    _teardown_qp_pool(qp_pool, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT)
            finally:
                _stop_concurrent_server(host=DEFAULT_SSH_HOST)

    # Linear mixture baseline
    if rows:
        c1 = next((r["c_est"] for r in rows if r["workload"] == "nm1_only"), None)
        c8 = next((r["c_est"] for r in rows if r["workload"] == "nm8_only"), None)
        if c1 and c8:
            print(f"\n=== Linear Mixture Baseline ===", flush=True)
            print(f"  C(NM=1)={c1:.0f} C(NM=8)={c8:.0f}", flush=True)
            for r in rows:
                if r["workload"] not in ("nm1_only", "nm8_only"):
                    p = r["heavy_frac"]
                    c_pred = 1.0 / ((1 - p) / c1 + p / c8)
                    r["c_pred"] = c_pred
                    r["interaction_ratio"] = r["c_est"] / c_pred
                    print(f"  {r['workload']}: C_obs={r['c_est']:.0f} C_pred={c_pred:.0f} "
                          f"ratio={r['interaction_ratio']:.2f}", flush=True)

    if not rows:
        print("No data collected.")
        return

    csv_path = os.path.join(output_dir, "capacity.csv")
    # Collect all fieldnames across rows (mixture rows have extra fields)
    all_fields = []
    for r in rows:
        for k in r.keys():
            if k not in all_fields:
                all_fields.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="S6 capacity recalibration")
    parser.add_argument("--output", default="results/s6_capacity")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    run_s6(args.output, args.trials)


if __name__ == "__main__":
    main()
