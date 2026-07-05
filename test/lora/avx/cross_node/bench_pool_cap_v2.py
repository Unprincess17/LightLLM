"""S5: Physical-pool x active-cap x executor x stream matrix.

Three separate matched sweeps (not full cross-product, because min(P,A,E)
makes many combinations structurally capped):

  Sweep 1 (physical-pool): A=8, E=8, P in {8,16,32,64}
  Sweep 2 (active-concurrency): P=64, E=32, A in {8,16,32}
  Sweep 3 (executor-width): P=64, A=8, E in {8,16,32}

Central test: Does P=32,A=8 match P=8,A=8? (TOST equivalence)
Historical reproduction: E=256, pool.borrow() gate (no active_cap), P in {8,16,32}
Stream sensitivity: (P=8,A=8) and (P=32,A=8) at S in {1,2,4}
"""
import argparse, csv, os, socket, statistics, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from bench_decomposition import (
    HIDDEN_DIM, INTERMEDIATE_DIM, BYTES_PER_PARAM,
    _make_sjf_request,
    _start_concurrent_server, _stop_concurrent_server,
    _teardown_qp_pool, _sock_recv,
    DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT, DEFAULT_SSH_HOST,
    DEFAULT_LOCAL_IP, DEFAULT_BASE_CONTROL_PORT,
)
from common.protocol import new_request_id
from common.transport import PersistentTransport
from common.stats import tost_equivalence

N_TRIALS = 3
N_ITERS = 50
NM = 8  # heavy request for all S5 tests


# ---------------------------------------------------------------------------
# Custom QP pool setup -- decouples pool_size from active_cap
# ---------------------------------------------------------------------------

def _setup_qp_pool_s5(server_host, server_port, local_ip, base_control_port,
                      gpu_buffer_bytes, pool_size, active_cap):
    """Create QPPoolClient with independent pool_size and active_cap.

    Unlike _setup_qp_pool in bench_decomposition.py which derives pool_size
    from cell_spec["conc"], this function takes pool_size as a separate
    parameter to allow decoupling physical pool size from active concurrency.
    """
    from qppool import QPPoolClient
    try:
        from rdma_qp_generator.gpudirect_transport import GPUDirectTransport  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "rdma_qp_generator.gpudirect_transport is not installed; "
            "cannot create QP pool for remote cells B1/B2/B5"
        )

    qp_pool = QPPoolClient(
        size=pool_size,
        local_ip=local_ip,
        remote_ip=server_host,
        base_control_port=base_control_port,
        gpu_buffer_bytes=gpu_buffer_bytes,
        mode="preconnected",
        active_cap=active_cap,
    )

    # Open TCP socket for setup_pool message
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60.0)
    sock.connect((server_host, server_port))
    try:
        qp_pool.setup(sock)  # sends setup_pool + accepts QP connections
        response = _sock_recv(sock)
        if response.get("status") != "ok":
            raise RuntimeError(f"setup_pool failed: {response}")
    finally:
        sock.close()

    print(f"[s5] QP pool ready: size={pool_size}, active_cap={active_cap}")
    return qp_pool


# ---------------------------------------------------------------------------
# Single-configuration runner
# ---------------------------------------------------------------------------

def run_config(server_host, server_port, ssh_host, local_ip, base_control_port,
               pool_size, active_cap, n_iters=50, executor_width=1, n_streams=1):
    """Run n_iters NM=8 requests with given pool_size and active_cap.

    If executor_width > 1, sends requests concurrently using ThreadPoolExecutor.
    Returns list of E2E latencies in microseconds.
    """
    # cell_spec with conc=active_cap (used for request message construction)
    cell_spec = {"transport": "persistent_tcp",
                 "runtime": "python_executor",
                 "conc": active_cap}

    _start_concurrent_server(host=ssh_host, listen_ip=server_host, port=server_port)
    try:
        act_bytes = HIDDEN_DIM * BYTES_PER_PARAM
        result_bytes = NM * INTERMEDIATE_DIM * BYTES_PER_PARAM
        gpu_buffer_bytes = act_bytes + result_bytes

        qp_pool = _setup_qp_pool_s5(server_host, server_port, local_ip,
                                    base_control_port, gpu_buffer_bytes,
                                    pool_size, active_cap)
        try:
            sock = socket.create_connection((server_host, server_port))
            sock.settimeout(120.0)
            transport = PersistentTransport(sock)

            latencies = []
            try:
                if executor_width > 1:
                    # Concurrent mode (for historical reproduction)
                    def _do_one():
                        req_id = new_request_id()
                        pool_id, gpu_t, _ = qp_pool.borrow()
                        try:
                            msg = _make_sjf_request(req_id, NM, 64, cell_spec, "baseline")
                            if n_streams != 1:
                                msg["n_streams"] = n_streams
                            t0 = time.perf_counter()
                            transport.request(msg)
                            t1 = time.perf_counter()
                        finally:
                            qp_pool.return_transport(pool_id, gpu_t)
                        return (t1 - t0) * 1e6

                    with ThreadPoolExecutor(max_workers=executor_width) as pool:
                        futures = [pool.submit(_do_one) for _ in range(n_iters)]
                        for fut in as_completed(futures):
                            latencies.append(fut.result())
                else:
                    # Sequential mode (main sweeps)
                    for _ in range(n_iters):
                        req_id = new_request_id()
                        pool_id, gpu_t, _ = qp_pool.borrow()
                        try:
                            msg = _make_sjf_request(req_id, NM, 64, cell_spec, "baseline")
                            if n_streams != 1:
                                msg["n_streams"] = n_streams
                            t0 = time.perf_counter()
                            transport.request(msg)
                            t1 = time.perf_counter()
                        finally:
                            qp_pool.return_transport(pool_id, gpu_t)
                        latencies.append((t1 - t0) * 1e6)
            finally:
                transport.close()

            return latencies
        finally:
            _teardown_qp_pool(qp_pool, server_host, server_port)
    finally:
        _stop_concurrent_server(host=ssh_host, listen_ip=server_host, port=server_port)


# ---------------------------------------------------------------------------
# Main S5 campaign
# ---------------------------------------------------------------------------

def run_s5(output_dir="results/s5_pool_cap", n_trials=None, n_iters=None):
    n_trials = n_trials or N_TRIALS
    n_iters = n_iters or N_ITERS
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    # ---- Sweep 1: Physical-pool effect (A=8, E=8, P varies) ----
    print("=== Sweep 1: Physical-pool effect ===")
    sweep1_results = {}
    for P in [8, 16, 32, 64]:
        trial_p50s = []
        for trial in range(n_trials):
            try:
                lats = run_config(DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
                                  DEFAULT_SSH_HOST, DEFAULT_LOCAL_IP,
                                  DEFAULT_BASE_CONTROL_PORT,
                                  pool_size=P, active_cap=8, n_iters=n_iters)
                p50 = statistics.median(lats)
                trial_p50s.append(p50)
                print(f"  P={P} trial {trial}: P50={p50:.0f}us")
            except Exception as e:
                print(f"  P={P} trial {trial}: skipped ({e})")
        if trial_p50s:
            med = statistics.median(trial_p50s)
            sweep1_results[P] = trial_p50s
            rows.append({"sweep": "physical_pool", "P": P, "A": 8, "E": 8,
                         "S": 1, "p50_median": med,
                         "p50_ci_lo": min(trial_p50s), "p50_ci_hi": max(trial_p50s)})
            print(f"  P={P} A=8: P50={med:.0f}us")

    # TOST equivalence: P=8 vs P=32
    if 8 in sweep1_results and 32 in sweep1_results:
        margin = 0.05 * statistics.mean(sweep1_results[8])
        equiv = tost_equivalence(sweep1_results[8], sweep1_results[32], margin=margin)
        print(f"  TOST P=8 vs P=32 (margin={margin:.0f}us): "
              f"{'EQUIVALENT' if equiv else 'NOT equivalent'}")

    # ---- Sweep 2: Active-concurrency effect (P=64, E=32, A varies) ----
    print("\n=== Sweep 2: Active-concurrency effect ===")
    for A in [8, 16, 32]:
        trial_p50s = []
        for trial in range(n_trials):
            try:
                lats = run_config(DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
                                  DEFAULT_SSH_HOST, DEFAULT_LOCAL_IP,
                                  DEFAULT_BASE_CONTROL_PORT,
                                  pool_size=64, active_cap=A, n_iters=n_iters)
                p50 = statistics.median(lats)
                trial_p50s.append(p50)
                print(f"  A={A} trial {trial}: P50={p50:.0f}us")
            except Exception as e:
                print(f"  A={A} trial {trial}: skipped ({e})")
        if trial_p50s:
            med = statistics.median(trial_p50s)
            rows.append({"sweep": "active_concurrency", "P": 64, "A": A, "E": 32,
                         "S": 1, "p50_median": med,
                         "p50_ci_lo": min(trial_p50s), "p50_ci_hi": max(trial_p50s)})
            print(f"  P=64 A={A}: P50={med:.0f}us")

    # ---- Sweep 3: Executor-width effect (P=64, A=8, E varies) ----
    # E is a label for sequential mode -- executor_width=1 for all since
    # requests are sent one-at-a-time.  The sweep demonstrates that E
    # does not affect per-request latency when there is no contention.
    print("\n=== Sweep 3: Executor-width effect ===")
    for E in [8, 16, 32]:
        trial_p50s = []
        for trial in range(n_trials):
            try:
                lats = run_config(DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
                                  DEFAULT_SSH_HOST, DEFAULT_LOCAL_IP,
                                  DEFAULT_BASE_CONTROL_PORT,
                                  pool_size=64, active_cap=8, n_iters=n_iters)
                p50 = statistics.median(lats)
                trial_p50s.append(p50)
                print(f"  E={E} trial {trial}: P50={p50:.0f}us")
            except Exception as e:
                print(f"  E={E} trial {trial}: skipped ({e})")
        if trial_p50s:
            med = statistics.median(trial_p50s)
            rows.append({"sweep": "executor_width", "P": 64, "A": 8, "E": E,
                         "S": 1, "p50_median": med,
                         "p50_ci_lo": min(trial_p50s), "p50_ci_hi": max(trial_p50s)})
            print(f"  P=64 A=8 E={E}: P50={med:.0f}us")

    # ---- Historical reproduction: E=256, pool.borrow() gate, P in {8,16,32} ----
    # active_cap = pool_size => active semaphore is redundant with pool
    # semaphore, simulating the old design where pool.borrow() was the only gate.
    print("\n=== Historical reproduction: E=256, pool.borrow() gate ===")
    for P in [8, 16, 32]:
        trial_p50s = []
        for trial in range(n_trials):
            try:
                lats = run_config(DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
                                  DEFAULT_SSH_HOST, DEFAULT_LOCAL_IP,
                                  DEFAULT_BASE_CONTROL_PORT,
                                  pool_size=P, active_cap=P, n_iters=n_iters,
                                  executor_width=256)
                p50 = statistics.median(lats)
                trial_p50s.append(p50)
                print(f"  P={P} trial {trial}: P50={p50:.0f}us")
            except Exception as e:
                print(f"  P={P} trial {trial}: skipped ({e})")
        if trial_p50s:
            med = statistics.median(trial_p50s)
            rows.append({"sweep": "historical_repro", "P": P, "A": P, "E": 256,
                         "S": 1, "p50_median": med,
                         "p50_ci_lo": min(trial_p50s), "p50_ci_hi": max(trial_p50s)})
            print(f"  P={P} A=P E=256: P50={med:.0f}us")

    # ---- Stream sensitivity: (P=8,A=8) and (P=32,A=8) at S in {1,2,4} ----
    print("\n=== Stream sensitivity ===")
    for P in [8, 32]:
        for S in [1, 2, 4]:
            trial_p50s = []
            for trial in range(n_trials):
                try:
                    lats = run_config(DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
                                      DEFAULT_SSH_HOST, DEFAULT_LOCAL_IP,
                                      DEFAULT_BASE_CONTROL_PORT,
                                      pool_size=P, active_cap=8, n_iters=n_iters,
                                      n_streams=S)
                    p50 = statistics.median(lats)
                    trial_p50s.append(p50)
                    print(f"  P={P} S={S} trial {trial}: P50={p50:.0f}us")
                except Exception as e:
                    print(f"  P={P} S={S} trial {trial}: skipped ({e})")
            if trial_p50s:
                med = statistics.median(trial_p50s)
                rows.append({"sweep": "stream_sensitivity", "P": P, "A": 8, "E": 8,
                             "S": S, "p50_median": med,
                             "p50_ci_lo": min(trial_p50s), "p50_ci_hi": max(trial_p50s)})
                print(f"  P={P} A=8 S={S}: P50={med:.0f}us")

    # ---- Write CSV ----
    if not rows:
        print("No data collected.")
        return

    csv_path = os.path.join(output_dir, "pool_cap.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="S5 pool/cap matrix")
    parser.add_argument("--output", default="results/s5_pool_cap")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    run_s5(args.output, args.trials, args.iters)


if __name__ == "__main__":
    main()
