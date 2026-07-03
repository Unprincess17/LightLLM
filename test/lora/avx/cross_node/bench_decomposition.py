"""B0-B5 decomposition harness for the matched-contrast matrix.

B0: local buffers, Python direct, conc=1 (lower bound, external)
B1: persistent TCP, Python direct, conc=1
B2: persistent TCP, Python executor, conc=1
B3: persistent TCP, Python executor, conc=N
B4: per-request TCP, Python executor, conc=1
B5: per-request TCP, Python executor, conc=N (current architecture)

B6-B9 (C++ matched worker) are in a separate plan.
"""
import argparse
import json
import socket
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional

from common.transport import PersistentTransport
from common.protocol import send_message, recv_message, new_request_id
from common.instrumentation import RequestTimeline, account_request

CELLS = {
    "B0": {"transport": "local", "runtime": "python_direct", "conc": 1},
    "B1": {"transport": "persistent_tcp", "runtime": "python_direct", "conc": 1},
    "B2": {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 1},
    "B3": {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 8},
    "B4": {"transport": "per_request_tcp", "runtime": "python_executor", "conc": 1},
    "B5": {"transport": "per_request_tcp", "runtime": "python_executor", "conc": 8},
}

# Model dimensions (Qwen3-30B-A3B)
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 1536
BYTES_PER_PARAM = 2  # bf16

# Server connection defaults — UM251 is the GPU server, UM253 is the client.
DEFAULT_SERVER_HOST = "10.10.1.3"    # UM251 IB IP
DEFAULT_SERVER_PORT = 30200
DEFAULT_SSH_HOST = "10.113.164.251"  # UM251 management IP
DEFAULT_SSH_USER = "shufan"
DEFAULT_LOCAL_IP = "10.10.1.1"       # UM253 IB IP
DEFAULT_BASE_CONTROL_PORT = 31000    # QP handshake ports


@dataclass
class DecompositionConfig:
    cell: str
    nm: int = 8
    rank: int = 64
    n_trials: int = 5
    n_iters: int = 50

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(CELLS[self.cell])
        return d


def run_cell(config: DecompositionConfig, pool=None, act_bf16=None,
             variant: str = "baseline", **kwargs) -> dict:
    """Run one decomposition cell. Returns per-iteration latencies and accounting.

    B0 (local) is fully implemented here. B1/B2/B5 connect to the live
    concurrent_server on UM251. B3/B4/B6-B9 are in later plans.

    Extra keyword arguments (e.g. server_host, ssh_host) are forwarded to
    _run_remote_cell for B1/B2/B5.
    """
    if config.cell not in CELLS:
        raise ValueError(f"unknown cell {config.cell}; valid: {list(CELLS)}")
    cell_spec = CELLS[config.cell]

    if config.cell == "B0":
        return _run_b0_local(config, cell_spec)
    if config.cell in ("B1", "B2", "B5"):
        return _run_remote_cell(config, cell_spec, variant=variant, **kwargs)
    raise NotImplementedError(f"{config.cell} requires C++ worker (Plan 2)")


# ---------------------------------------------------------------------------
# B0: local baseline (no network)
# ---------------------------------------------------------------------------

def _run_b0_local(config: DecompositionConfig, cell_spec: dict) -> dict:
    """B0: local buffers, Python direct, conc=1. No network, no RDMA.
    Lower bound for the decomposition matrix."""
    import torch
    H, I, R, NM = HIDDEN_DIM, INTERMEDIATE_DIM, config.rank, config.nm
    device = "cuda"

    x = torch.randn(1, H, dtype=torch.float16, device=device)

    latencies_us = []
    outputs = []
    t0_last = None
    t18_last = None
    for trial in range(config.n_trials):
        weights_A = [torch.randn(R, H, dtype=torch.float32, device=device) for _ in range(NM)]
        weights_B = [torch.randn(R, I, dtype=torch.float32, device=device) for _ in range(NM)]

        for _ in range(config.n_iters):
            t0 = time.perf_counter()
            x_f32 = x.to(torch.float32)
            miss_outputs = []
            for i in range(NM):
                inter = x_f32 @ weights_A[i].T
                y = inter @ weights_B[i]
                miss_outputs.append(y)
            torch.cuda.synchronize()
            t18 = time.perf_counter()
            latencies_us.append((t18 - t0) * 1e6)
            if len(outputs) < 1:
                outputs = miss_outputs
            t0_last = t0
            t18_last = t18

    # Construct a real RequestTimeline for the last iteration.
    # B0 is local: no network send (t5=t0), no network receive (t6=t0),
    # no network response (t17=t18). cross_domain_residual should be ~0.
    tl = RequestTimeline(req_id=0, cell="B0")
    tl.set("t0", t0_last)
    tl.set("t5", t0_last)
    tl.set("t6", t0_last)
    tl.set("t17", t18_last)
    tl.set("t18", t18_last)
    accounting = account_request(tl)
    return {
        "config": config.to_dict(),
        "latencies_us": latencies_us,
        "outputs": outputs,
        "accounting": accounting,
        "cell_spec": cell_spec,
    }


# ---------------------------------------------------------------------------
# Raw TCP socket helpers (framing compatible with concurrent_server)
# ---------------------------------------------------------------------------

def _sock_send(sock: socket.socket, obj: dict) -> None:
    """Send JSON with 4-byte big-endian length prefix."""
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _sock_recv(sock: socket.socket) -> dict:
    """Receive JSON with 4-byte length prefix."""
    header = _recv_exact_sock(sock, 4)
    msg_len = struct.unpack("!I", header)[0]
    return json.loads(_recv_exact_sock(sock, msg_len).decode("utf-8"))


def _recv_exact_sock(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a socket."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Server lifecycle helpers (SSH to UM251)
# ---------------------------------------------------------------------------

def _ssh(cmd: str, host: str = DEFAULT_SSH_HOST,
         user: str = DEFAULT_SSH_USER, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command on the remote server via SSH (key-based auth)."""
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _start_concurrent_server(host: str = DEFAULT_SSH_HOST,
                             listen_ip: str = DEFAULT_SERVER_HOST,
                             port: int = DEFAULT_SERVER_PORT) -> None:
    """Start concurrent_server.py on the remote GPU node (UM251)."""
    _ssh("pkill -f concurrent_server.py || true", host=host)
    time.sleep(0.5)
    repo = "/home/shufan/LightLLM-integrate-to-SLoRA/test/lora/avx/cross_node"
    # Use ssh -f -n so SSH forks to background after auth and returns immediately.
    # The server's stdout/stderr go to a log file; stdin from /dev/null.
    cmd = (
        f"cd {repo} && python concurrent_server.py "
        f"--listen-ip {listen_ip} --listen-port {port} "
        f"> /tmp/concurrent_server.log 2>&1 < /dev/null"
    )
    # ssh -f forks after auth; -n redirects stdin from /dev/null
    proc = subprocess.Popen(
        ["ssh", "-f", "-n", "-o", "StrictHostKeyChecking=no",
         f"{DEFAULT_SSH_USER}@{host}", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    proc.wait(timeout=10)
    time.sleep(2)
    result = _ssh("pgrep -f concurrent_server.py", host=host)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start concurrent_server on {host}")
    print(f"[bench] Server started on {listen_ip}:{port}")


def _stop_concurrent_server(host: str = DEFAULT_SSH_HOST,
                            listen_ip: str = DEFAULT_SERVER_HOST,
                            port: int = DEFAULT_SERVER_PORT) -> None:
    """Stop the concurrent_server on the remote GPU node."""
    # Try graceful shutdown first
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((listen_ip, port))
        try:
            _sock_send(sock, {"type": "shutdown"})
            _sock_recv(sock)
        finally:
            sock.close()
    except Exception:
        pass
    # Fallback: kill via SSH
    _ssh("pkill -f concurrent_server.py || true", host=host)
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# QP pool setup / teardown
# ---------------------------------------------------------------------------

def _setup_qp_pool(config: DecompositionConfig, cell_spec: dict,
                   server_host: str, server_port: int,
                   local_ip: str, base_control_port: int,
                   gpu_buffer_bytes: int):
    """Create QPPoolClient on the inference node and send setup_pool to server.

    Returns the QPPoolClient instance.  The QP pool uses pool_size=1; the
    server-side CentralDispatcher (active_cap=cell_spec['conc']) manages
    concurrency.
    """
    from qppool import QPPoolClient
    try:
        from rdma_qp_generator.gpudirect_transport import GPUDirectTransport
    except ImportError:
        raise RuntimeError(
            "rdma_qp_generator.gpudirect_transport is not installed; "
            "cannot create QP pool for remote cells B1/B2/B5"
        )

    pool_size = max(1, cell_spec["conc"])  # QPs must cover the active cap for true parallelism
    qp_pool = QPPoolClient(
        size=pool_size,
        local_ip=local_ip,
        remote_ip=server_host,
        base_control_port=base_control_port,
        gpu_buffer_bytes=gpu_buffer_bytes,
        mode="preconnected",
        active_cap=cell_spec["conc"],
    )
    # Open TCP socket for setup_pool message
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60.0)
    sock.connect((server_host, server_port))
    try:
        qp_pool.setup(sock)  # sends setup_pool + accepts QP connections
        # Read setup_pool response (server sends after QP connections established)
        response = _sock_recv(sock)
        if response.get("status") != "ok":
            raise RuntimeError(f"setup_pool failed: {response}")
    finally:
        sock.close()
    print(f"[bench] QP pool ready: size={pool_size}, active_cap={cell_spec['conc']}")
    return qp_pool


def _teardown_qp_pool(qp_pool, server_host: str, server_port: int) -> None:
    """Send teardown_pool to server and destroy local QP pool."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((server_host, server_port))
        try:
            _sock_send(sock, {"type": "teardown_pool"})
            _sock_recv(sock)
        finally:
            sock.close()
    except Exception as e:
        print(f"[bench] Warning: teardown_pool failed: {e}")
    try:
        qp_pool.teardown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def _make_request(req_id: int, config: DecompositionConfig,
                  cell_spec: dict, variant: str) -> dict:
    """Build an s4a_pooled request message."""
    return {
        "type": "s4a_pooled",
        "req_id": req_id,
        "rank": config.rank,
        "num_miss": config.nm,
        "hidden_dim": HIDDEN_DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "decompose": True,
        "decompose_level": "fine",
        "variant": variant,
        "active_cap": cell_spec["conc"],
    }


# ---------------------------------------------------------------------------
# B1/B2: persistent TCP transport
# ---------------------------------------------------------------------------

def _run_persistent(config: DecompositionConfig, cell_spec: dict, variant: str,
                    server_host: str, server_port: int,
                    qp_pool, act_bytes: int, result_bytes: int):
    """B1/B2: persistent TCP transport.

    B1 (python_direct): sequential, no executor.
    B2 (python_executor): ThreadPoolExecutor(max_workers=1).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120.0)
    sock.connect((server_host, server_port))
    transport = PersistentTransport(sock)

    latencies_us = []
    segments_per_request = []
    accountings = []

    try:
        if cell_spec["runtime"] == "python_direct":
            # B1: sequential, no executor — the "quiet baseline"
            for _trial in range(config.n_trials):
                for _i in range(config.n_iters):
                    req_id = new_request_id()
                    t0 = time.perf_counter()

                    pool_id, gpu_transport, _ = qp_pool.borrow()
                    try:
                        msg = _make_request(req_id, config, cell_spec, variant)
                        t5 = time.perf_counter()
                        response = transport.request(msg)
                        t18 = time.perf_counter()
                    finally:
                        qp_pool.return_transport(pool_id, gpu_transport)

                    e2e_us = (t18 - t0) * 1e6
                    latencies_us.append(e2e_us)
                    segments_per_request.append(response.get("segments", []))

                    tl = RequestTimeline(req_id=req_id, cell=config.cell)
                    tl.set("t0", t0)
                    tl.set("t5", t5)
                    tl.set("t6", t5)    # approximate: server recv ~ client send
                    tl.set("t17", t18)  # approximate: server send ~ client recv
                    tl.set("t18", t18)
                    accountings.append(account_request(tl))

        else:
            # B2: python_executor with max_workers=1
            def _do_one(transport=transport, qp_pool=qp_pool):
                req_id = new_request_id()
                t0 = time.perf_counter()
                pool_id, gpu_transport, _ = qp_pool.borrow()
                try:
                    msg = _make_request(req_id, config, cell_spec, variant)
                    t5 = time.perf_counter()
                    response = transport.request(msg)
                    t18 = time.perf_counter()
                finally:
                    qp_pool.return_transport(pool_id, gpu_transport)
                e2e_us = (t18 - t0) * 1e6
                segments = response.get("segments", [])
                tl = RequestTimeline(req_id=req_id, cell=config.cell)
                tl.set("t0", t0)
                tl.set("t5", t5)
                tl.set("t6", t5)
                tl.set("t17", t18)
                tl.set("t18", t18)
                return e2e_us, segments, account_request(tl)

            with ThreadPoolExecutor(max_workers=cell_spec["conc"]) as executor:
                futures = []
                for _trial in range(config.n_trials):
                    for _i in range(config.n_iters):
                        futures.append(executor.submit(_do_one))
                for fut in futures:
                    e2e_us, segments, acc = fut.result()
                    latencies_us.append(e2e_us)
                    segments_per_request.append(segments)
                    accountings.append(acc)
    finally:
        transport.close()

    return latencies_us, segments_per_request, accountings


# ---------------------------------------------------------------------------
# B5: per-request TCP transport
# ---------------------------------------------------------------------------

def _run_per_request(config: DecompositionConfig, cell_spec: dict, variant: str,
                     server_host: str, server_port: int,
                     qp_pool, act_bytes: int, result_bytes: int):
    """B5: per-request TCP transport with ThreadPoolExecutor(max_workers=conc).

    Each request opens its own TCP socket, sends s4a_pooled, receives the
    response, and closes the socket.
    """
    def _do_one():
        req_id = new_request_id()
        t0 = time.perf_counter()

        pool_id, gpu_transport, _ = qp_pool.borrow()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(120.0)
            t2 = time.perf_counter()
            sock.connect((server_host, server_port))
            t3 = time.perf_counter()

            msg = _make_request(req_id, config, cell_spec, variant)
            t5 = time.perf_counter()
            _sock_send(sock, msg)
            response = _sock_recv(sock)
            t18 = time.perf_counter()
            sock.close()
        finally:
            qp_pool.return_transport(pool_id, gpu_transport)

        e2e_us = (t18 - t0) * 1e6
        segments = response.get("segments", [])

        tl = RequestTimeline(req_id=req_id, cell=config.cell)
        tl.set("t0", t0)
        tl.set("t2", t2)
        tl.set("t3", t3)
        tl.set("t5", t5)
        tl.set("t6", t5)    # approximate
        tl.set("t17", t18)  # approximate
        tl.set("t18", t18)
        return e2e_us, segments, account_request(tl)

    latencies_us = []
    segments_per_request = []
    accountings = []

    with ThreadPoolExecutor(max_workers=cell_spec["conc"]) as executor:
        futures = []
        for _trial in range(config.n_trials):
            for _i in range(config.n_iters):
                futures.append(executor.submit(_do_one))
        for fut in futures:
            e2e_us, segments, acc = fut.result()
            latencies_us.append(e2e_us)
            segments_per_request.append(segments)
            accountings.append(acc)

    return latencies_us, segments_per_request, accountings


# ---------------------------------------------------------------------------
# _run_remote_cell: B1/B2/B5 dispatch
# ---------------------------------------------------------------------------

def _run_remote_cell(config: DecompositionConfig, cell_spec: dict,
                     variant: str = "baseline",
                     server_host: str = DEFAULT_SERVER_HOST,
                     server_port: int = DEFAULT_SERVER_PORT,
                     ssh_host: str = DEFAULT_SSH_HOST,
                     local_ip: str = DEFAULT_LOCAL_IP,
                     base_control_port: int = DEFAULT_BASE_CONTROL_PORT) -> dict:
    """Run a remote cell (B1/B2/B5) against the live concurrent_server on UM251.

    1. Starts the concurrent_server via SSH.
    2. Sets up the QP pool (RDMA data channel) with active_cap=cell_spec['conc'].
    3. Sends s4a_pooled requests via the cell-appropriate transport:
       - B1/B2: PersistentTransport (one TCP connection, multiplexed)
       - B5: per-request TCP (new socket per request)
    4. Collects per-segment CPU/GPU timings and E2E latencies.
    5. Constructs RequestTimeline and calls account_request for accounting.
    6. Tears down QP pool and stops server.
    """
    act_bytes = HIDDEN_DIM * BYTES_PER_PARAM
    result_bytes = config.nm * INTERMEDIATE_DIM * BYTES_PER_PARAM
    gpu_buffer_bytes = act_bytes + result_bytes

    # 1. Start server
    _start_concurrent_server(host=ssh_host, listen_ip=server_host, port=server_port)

    try:
        # 2. Set up QP pool
        qp_pool = _setup_qp_pool(
            config, cell_spec, server_host, server_port,
            local_ip, base_control_port, gpu_buffer_bytes,
        )

        try:
            # 3. Run requests
            if cell_spec["transport"] == "persistent_tcp":
                latencies, segments, accountings = _run_persistent(
                    config, cell_spec, variant, server_host, server_port,
                    qp_pool, act_bytes, result_bytes,
                )
            else:  # per_request_tcp
                latencies, segments, accountings = _run_per_request(
                    config, cell_spec, variant, server_host, server_port,
                    qp_pool, act_bytes, result_bytes,
                )
        finally:
            # 5. Tear down QP pool
            _teardown_qp_pool(qp_pool, server_host, server_port)
    finally:
        # 6. Stop server
        _stop_concurrent_server(host=ssh_host, listen_ip=server_host, port=server_port)

    # Aggregate accounting (mean across all requests)
    if accountings:
        avg_acc = {
            k: sum(a[k] for a in accountings) / len(accountings)
            for k in accountings[0]
        }
    else:
        avg_acc = {}

    return {
        "config": config.to_dict(),
        "latencies_us": latencies,
        "segments_per_request": segments,
        "accounting": avg_acc,
        "cell_spec": cell_spec,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B0-B5 decomposition harness")
    parser.add_argument("--cell", required=True, choices=list(CELLS))
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--variant", default="baseline",
                        help="Timing variant (baseline, cuda_graph, cache_flush, ...)")
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--output", default="results/decomposition/decomposition.csv")
    args = parser.parse_args()

    config = DecompositionConfig(
        cell=args.cell, nm=args.nm, rank=args.rank,
        n_trials=args.trials, n_iters=args.iters,
    )
    result = run_cell(config, variant=args.variant)
    print(f"Cell {args.cell}: {len(result['latencies_us'])} samples")
    print(f"Spec: {result['cell_spec']}")


if __name__ == "__main__":
    main()
