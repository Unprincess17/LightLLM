#!/usr/bin/env python3
"""Concurrent LoRA Recovery Server — pool-aware server for UM251.

Supports:
  - setup_pool: pre-create N GPUDirect QP pairs (connected to client)
  - s4a_pooled: execute one S4a recovery using a pooled QP
  - teardown_pool: destroy the pool
  - shutdown: stop the server

No GLOO dependency — pure RDMA via GPUDirectTransport.
"""

import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from collections import deque


def _signal_handler(signum, frame):
    """Log received signals for debugging."""
    print(f"[pool_server] SIGNAL {signum} received", flush=True)
    if signum in (signal.SIGTERM, signal.SIGINT):
        sys.exit(128 + signum)
    # For other signals, just log and continue


# Install signal handlers for debugging
for _s in range(1, 32):
    try:
        signal.signal(_s, _signal_handler)
    except (OSError, ValueError):
        pass
from typing import Any

import torch

BYTES_PER_PARAM = 2  # bf16

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qppool import QPPoolServer, _send_json, _recv_json, _recv_exact

try:
    from rdma_qp_generator.gdr_sync import copy_gpu_to_gpu
except ImportError:
    copy_gpu_to_gpu = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# central admission dispatcher
# ---------------------------------------------------------------------------

class CentralDispatcher:
    """Central admission gate before the executor.

    Caps active concurrency independent of physical QP pool size.
    Per spec S3/S4/S5."""

    def __init__(self, active_cap: int):
        self.active_cap = active_cap
        self._sem = threading.Semaphore(active_cap)
        self._lock = threading.Lock()
        self.wait_us: deque = deque(maxlen=100000)  # per-request wait times (bounded)

    def acquire(self) -> float:
        t0 = time.perf_counter()
        self._sem.acquire()
        t1 = time.perf_counter()
        wait = (t1 - t0) * 1e6
        with self._lock:
            self.wait_us.append(wait)
        return wait

    def release(self) -> None:
        self._sem.release()


# Module-level dispatcher, set by handle_setup_pool when active_cap is provided
_dispatcher: CentralDispatcher | None = None

# Thread-local for req_id (set per-request in handle_request, used by
# send_json_response to echo req_id back for PersistentTransport matching).
_req_id_local = threading.local()


def build_timing_response(nm: int, segments: list, variant: str) -> dict:
    """Build the timing response with per-segment CPU and GPU timings."""
    return {"nm": nm, "variant": variant, "segments": segments}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def send_json_response(conn: socket.socket, obj: dict[str, Any]) -> None:
    """Send JSON response.  Echoes req_id from thread-local if set (for
    PersistentTransport request-ID matching)."""
    req_id = getattr(_req_id_local, "req_id", None)
    if req_id is not None:
        obj["req_id"] = req_id
    _send_json(conn, obj)


# ---------------------------------------------------------------------------
# request handlers
# ---------------------------------------------------------------------------

_pool: QPPoolServer | None = None
_pool_lock = threading.Lock()

# 64MB scratch buffer for L2 cache flush (RTX 6000 Ada has 48MB L2)
_scratch_buf = None

def _get_scratch_buf():
    global _scratch_buf
    if _scratch_buf is None:
        _scratch_buf = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _scratch_buf


_l2_scratch = None

def _get_l2_scratch():
    """Scratch buffer sized to the device's L2 cache (runtime-queried)."""
    global _l2_scratch
    if _l2_scratch is None:
        l2_size = torch.cuda.get_device_properties(0).l2_cache_size
        _l2_scratch = torch.empty(l2_size // 4, dtype=torch.float32, device="cuda")
    return _l2_scratch


# ---------------------------------------------------------------------------
# CUDA graph cache: graphs are captured ONCE per (hidden_dim, intermediate_dim,
# rank, num_miss) and replayed across requests.  This avoids the 121ms
# per-request capture cost.
# ---------------------------------------------------------------------------

_graph_cache = {}            # key -> (graphs, static_act, static_a, static_b, static_inter, static_out)
_graph_cache_lock = threading.Lock()   # serializes graph replay for thread safety (B5)
_graph_capture_count = {}   # per-key: how many times captured (should be 1)
_graph_replay_count = {}    # per-key: how many times replayed


def _get_graph_cache_key(hidden_dim, intermediate_dim, rank, num_miss):
    return (hidden_dim, intermediate_dim, rank, num_miss)


def _get_or_capture_graphs(key, hidden_dim, intermediate_dim, rank, num_miss,
                           act_f32_template, device="cuda"):
    """Get cached graphs for this key, or capture them if not yet cached.

    Returns (graphs, static_act, static_a, static_b, static_inter, static_out).
    The caller must copy request data into static buffers before replay.
    """
    if key in _graph_cache:
        return _graph_cache[key]

    with _graph_cache_lock:
        if key in _graph_cache:  # double-check after acquiring lock
            return _graph_cache[key]

        # Allocate static buffers (stable addresses for graph capture)
        static_act = torch.empty(1, hidden_dim, dtype=torch.float32, device=device)
        static_a = [torch.empty(rank, hidden_dim, dtype=torch.float32, device=device)
                    for _ in range(num_miss)]
        static_b = [torch.empty(rank, intermediate_dim, dtype=torch.float32, device=device)
                    for _ in range(num_miss)]
        static_inter = [torch.empty(1, rank, dtype=torch.float32, device=device)
                        for _ in range(num_miss)]
        static_out = [torch.empty(1, intermediate_dim, dtype=torch.float32, device=device)
                      for _ in range(num_miss)]

        # Warmup (required before graph capture)
        for i in range(num_miss):
            torch.mm(static_act, static_a[i].T, out=static_inter[i])
            torch.mm(static_inter[i], static_b[i], out=static_out[i].view(-1))
        torch.cuda.synchronize()

        # Capture one graph per miss
        graphs = []
        for i in range(num_miss):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                torch.mm(static_act, static_a[i].T, out=static_inter[i])
                torch.mm(static_inter[i], static_b[i], out=static_out[i].view(-1))
            graphs.append(g)

        _graph_cache[key] = (graphs, static_act, static_a, static_b,
                             static_inter, static_out)
        _graph_capture_count[key] = 1
        _graph_replay_count[key] = 0
        return _graph_cache[key]


def handle_setup_pool(conn, params, listen_ip, remote_ip):
    """Create the server-side QP pool."""
    global _pool, _dispatcher
    with _pool_lock:
        if _pool is not None:
            _pool.teardown()

        params["remote_ip"] = remote_ip
        _pool = QPPoolServer(local_ip=listen_ip)
        _pool.setup(params)

    # Configure central admission dispatcher
    active_cap = params.get("active_cap")
    if active_cap is not None:
        _dispatcher = CentralDispatcher(int(active_cap))
        print(f"[pool_server] Central dispatcher: active_cap={active_cap}",
              flush=True)
    else:
        _dispatcher = None

    send_json_response(conn, {
        "status": "ok",
        "pool_size": _pool.size,
        "mode": params.get("mode", "preconnected"),
    })
    print(f"[pool_server] Pool ready: size={_pool.size}, mode={params.get('mode', 'preconnected')}",
          flush=True)


def _gpu_copy_in(transport, hidden_dim, act_bytes):
    """Helper: copy activation from transport buffer to a fresh GPU tensor."""
    act_bf16 = torch.empty(hidden_dim, dtype=torch.bfloat16, device="cuda")
    copy_gpu_to_gpu(act_bf16.data_ptr(), transport.gpu_ptr(), act_bytes)
    return act_bf16


def handle_s4a_pooled(conn, params):
    """Execute S4a using a pooled QP.

    If *params* contains ``decompose: true``, per-segment timing is returned
    in the response under the ``segments`` key.  When ``decompose_level`` is
    ``"fine"``, eager variants use dual-domain timing: CPU via
    ``time.perf_counter()`` (no sync) and GPU via ``torch.cuda.Event``
    (read after a single end-of-loop sync).  The ``cuda_graph`` variant
    reports a single ``graph_replay`` segment with CPU submission time and
    total GPU replay time.
    """
    global _pool, _dispatcher
    if _pool is None:
        send_json_response(conn, {"status": "error", "message": "pool not initialized"})
        return

    rank = params["rank"]
    num_miss = params["num_miss"]
    hidden_dim = params["hidden_dim"]
    intermediate_dim = params["intermediate_dim"]
    decompose = params.get("decompose", False)
    decompose_level = params.get("decompose_level", "coarse")
    variant = params.get("variant", "baseline")
    timing_method = params.get("timing_method", "new")  # "new" (CUDA events) or "old" (sync+perf_counter per segment)
    total_perturbation_us = 0.0  # I2: accumulates perturbation cost across misses

    act_bytes = hidden_dim * BYTES_PER_PARAM
    result_bytes = num_miss * intermediate_dim * BYTES_PER_PARAM

    handshake_port = params.get("handshake_port")
    if _dispatcher is not None:
        _dispatcher.acquire()
    pool_id = None
    transport = None
    try:
        pool_id, transport = _pool.borrow(handshake_port=handshake_port)
        # Pre-generate LoRA weights on GPU (one-off cost; could be cached)
        a_gpu = torch.randn(
            num_miss, rank, hidden_dim, dtype=torch.bfloat16, device="cuda"
        )
        b_gpu = torch.randn(
            num_miss, rank, intermediate_dim, dtype=torch.bfloat16, device="cuda"
        )

        segments = []  # list of {"name", "cpu_us", "gpu_us"} when decompose=True

        def _seg(name, fn, *args):
            """Run *fn*, optionally timing with CUDA sync (single-domain)."""
            if decompose:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ret = fn(*args)
            if decompose:
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e6
            segments.append({"name": name, "cpu_us": dt, "gpu_us": dt})
            return ret

        # RDMA READ activation from client's GPU buffer
        if decompose:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            transport.read_from_remote(0, 0, act_bytes)
            dt = (time.perf_counter() - t0) * 1e6
            segments.append({"name": "rdma_read", "cpu_us": dt, "gpu_us": dt})
        else:
            transport.read_from_remote(0, 0, act_bytes)

        # Read activation from local GPU buffer
        if decompose:
            act_bf16 = _seg("gpu_copy_in", _gpu_copy_in, transport, hidden_dim, act_bytes)
        else:
            act_bf16 = torch.empty(hidden_dim, dtype=torch.bfloat16, device="cuda")
            copy_gpu_to_gpu(act_bf16.data_ptr(), transport.gpu_ptr(), act_bytes)

        # GPU compute: act @ A^T @ B^T
        act_f32 = act_bf16.to(torch.float32).view(1, hidden_dim)
        result = torch.zeros(
            num_miss, intermediate_dim, dtype=torch.float32, device="cuda"
        )

        if decompose and decompose_level == "fine":
            if variant == "cuda_graph":
                key = _get_graph_cache_key(hidden_dim, intermediate_dim, rank, num_miss)
                graphs, static_act, static_a, static_b, static_inter, static_out = \
                    _get_or_capture_graphs(key, hidden_dim, intermediate_dim,
                                           rank, num_miss, act_f32)

                # Copy request data into static buffers BEFORE replay
                static_act.copy_(act_f32)
                for i in range(num_miss):
                    w_idx = 0 if variant == "same_weights" else i
                    static_a[i].copy_(a_gpu[w_idx].to(torch.float32))
                    static_b[i].copy_(b_gpu[w_idx].to(torch.float32))

                # Replay (serialized for thread safety)
                ev_s = torch.cuda.Event(enable_timing=True)
                ev_e = torch.cuda.Event(enable_timing=True)
                with _graph_cache_lock:
                    ev_s.record()
                    t0 = time.perf_counter()
                    for i in range(num_miss):
                        graphs[i].replay()
                    t1 = time.perf_counter()
                    ev_e.record()
                    torch.cuda.synchronize()
                    _graph_replay_count[key] = _graph_replay_count.get(key, 0) + 1

                segments.append({
                    "name": "graph_replay",
                    "cpu_us": (t1 - t0) * 1e6,
                    "gpu_us": ev_s.elapsed_time(ev_e) * 1000,
                })

                # Copy results from static output buffers to result tensor
                for i in range(num_miss):
                    result[i] = static_out[i]

            elif timing_method == "old":
                # OLD timing: sync + perf_counter per segment (original method).
                # This is the ablation path — identical compute, different timing.
                for i in range(num_miss):
                    # Variant-specific perturbation (same as new method)
                    if variant == "allocator-reset":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    elif variant == "device-cache-perturbation":
                        _get_l2_scratch().zero_()
                        torch.cuda.synchronize()

                    # alloc (sync + perf_counter)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    t1 = time.perf_counter()
                    torch.cuda.synchronize()
                    segments.append({"name": f"miss_{i}_alloc",
                                     "cpu_us": (t1 - t0) * 1e6,
                                     "gpu_us": (t1 - t0) * 1e6})

                    # dtype
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    w_idx = 0 if variant == "same_weights" else i
                    a_f32 = a_gpu[w_idx].to(torch.float32)
                    b_f32 = b_gpu[w_idx].to(torch.float32)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    segments.append({"name": f"miss_{i}_dtype",
                                     "cpu_us": (t1 - t0) * 1e6,
                                     "gpu_us": (t1 - t0) * 1e6})

                    # mm1
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    inter = act_f32 @ a_f32.T
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    segments.append({"name": f"miss_{i}_mm1",
                                     "cpu_us": (t1 - t0) * 1e6,
                                     "gpu_us": (t1 - t0) * 1e6})

                    # mm2
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result[i] = (inter @ b_f32).view(-1)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    segments.append({"name": f"miss_{i}_mm2",
                                     "cpu_us": (t1 - t0) * 1e6,
                                     "gpu_us": (t1 - t0) * 1e6})

            else:
                # Eager variants: dual-domain timing per sub-segment (NEW method)
                # CPU: perf_counter (no sync inside)
                # GPU: CUDA events (read after a single end-of-loop sync)
                timing_entries = []  # (name, cpu_us, start_event, end_event)

                for i in range(num_miss):
                    # Variant-specific perturbation (synced, before timing)
                    if variant == "cache_flush":
                        torch.cuda.empty_cache()
                        _get_scratch_buf().zero_()
                        torch.cuda.synchronize()
                    elif variant == "allocator-reset":
                        _p0 = time.perf_counter()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        total_perturbation_us += (time.perf_counter() - _p0) * 1e6
                    elif variant == "device-cache-perturbation":
                        _p0 = time.perf_counter()
                        _get_l2_scratch().zero_()
                        torch.cuda.synchronize()
                        total_perturbation_us += (time.perf_counter() - _p0) * 1e6

                    # --- alloc --- (no-op placeholder, same as original)
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    t0 = time.perf_counter()
                    # alloc was a no-op in the original code
                    t1 = time.perf_counter()
                    ev_e.record()
                    timing_entries.append(
                        (f"miss_{i}_alloc", (t1 - t0) * 1e6, ev_s, ev_e))

                    # --- dtype ---
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    t0 = time.perf_counter()
                    w_idx = 0 if variant == "same_weights" else i
                    a_f32 = a_gpu[w_idx].to(torch.float32)
                    b_f32 = b_gpu[w_idx].to(torch.float32)
                    t1 = time.perf_counter()
                    ev_e.record()
                    timing_entries.append(
                        (f"miss_{i}_dtype", (t1 - t0) * 1e6, ev_s, ev_e))

                    # --- mm1 ---
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    t0 = time.perf_counter()
                    inter = act_f32 @ a_f32.T
                    t1 = time.perf_counter()
                    ev_e.record()
                    timing_entries.append(
                        (f"miss_{i}_mm1", (t1 - t0) * 1e6, ev_s, ev_e))

                    # --- mm2 ---
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    t0 = time.perf_counter()
                    result[i] = (inter @ b_f32).view(-1)
                    t1 = time.perf_counter()
                    ev_e.record()
                    timing_entries.append(
                        (f"miss_{i}_mm2", (t1 - t0) * 1e6, ev_s, ev_e))

                # Batch sync: read all GPU event elapsed times at once
                torch.cuda.synchronize()
                for name, cpu_us, ev_s, ev_e in timing_entries:
                    segments.append({
                        "name": name,
                        "cpu_us": cpu_us,
                        "gpu_us": ev_s.elapsed_time(ev_e) * 1000,
                    })

        elif decompose:
            # Coarse decomposition (existing behavior)
            for i in range(num_miss):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                a_f32 = a_gpu[i].to(torch.float32)
                b_f32 = b_gpu[i].to(torch.float32)
                inter = act_f32 @ a_f32.T
                result[i] = (inter @ b_f32).view(-1)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e6
                segments.append({"name": f"miss_{i}", "cpu_us": dt, "gpu_us": dt})
        else:
            # No decomposition (existing fast path)
            for i in range(num_miss):
                a_f32 = a_gpu[i].to(torch.float32)
                b_f32 = b_gpu[i].to(torch.float32)
                inter = act_f32 @ a_f32.T
                result[i] = (inter @ b_f32).view(-1)

        # Write result to buffer + RDMA WRITE to client
        if decompose:
            result_bf16 = result.to(torch.bfloat16)
            _seg("gpu_copy_out", copy_gpu_to_gpu,
                 transport.gpu_ptr() + act_bytes,
                 result_bf16.data_ptr(),
                 result_bytes)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            transport.write_to_remote(act_bytes, act_bytes, result_bytes)
            dt = (time.perf_counter() - t0) * 1e6
            segments.append({"name": "rdma_write", "cpu_us": dt, "gpu_us": dt})
        else:
            result_bf16 = result.to(torch.bfloat16)
            copy_gpu_to_gpu(
                transport.gpu_ptr() + act_bytes,
                result_bf16.data_ptr(),
                result_bytes,
            )
            transport.write_to_remote(act_bytes, act_bytes, result_bytes)

        if decompose:
            resp = build_timing_response(
                nm=num_miss, segments=segments, variant=variant)
            resp["status"] = "ok"
            resp["result_bytes"] = result_bytes
            resp["perturbation_cost_us"] = total_perturbation_us  # I2
            resp["graph_captures"] = sum(_graph_capture_count.values())
            resp["graph_replays"] = sum(_graph_replay_count.values())
            send_json_response(conn, resp)
        else:
            send_json_response(conn, {"status": "ok", "result_bytes": result_bytes})
    finally:
        if pool_id is not None:
            _pool.return_transport(pool_id, transport)
        if _dispatcher is not None:
            _dispatcher.release()


def handle_teardown_pool(conn):
    """Destroy the QP pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.teardown()
            _pool = None
    send_json_response(conn, {"status": "ok"})


# ---------------------------------------------------------------------------
# per-connection dispatch
# ---------------------------------------------------------------------------

def handle_request(
    conn: socket.socket,
    addr: tuple,
    listen_ip: str,
    log_fn=print,
) -> None:
    """Receive JSON messages, dispatch, send responses.

    Loops for persistent connections (B1/B2 PersistentTransport).
    For per-request TCP (B5), the client sends one message and closes;
    the loop exits cleanly on EOFError.
    """
    try:
        while True:
            try:
                header = _recv_exact(conn, 4)
                msg_len = struct.unpack("!I", header)[0]
                payload = _recv_exact(conn, msg_len)
                params = json.loads(payload.decode("utf-8"))
            except (EOFError, ConnectionResetError) as e:
                # Connection closed by client — normal for per-request TCP
                return
            except json.JSONDecodeError as e:
                log_fn(f"[pool_server] Bad request from {addr}: {e}")
                return

            # Set req_id for this request (thread-local, used by
            # send_json_response to echo it back for PersistentTransport).
            _req_id_local.req_id = params.get("req_id")

            msg_type = params.get("type", "")
            log_fn(f"[pool_server] {msg_type} from {addr}")

            if msg_type == "setup_pool":
                remote_ip = params.get("client_ip", addr[0])
                handle_setup_pool(conn, params, listen_ip, remote_ip)
            elif msg_type == "s4a_pooled":
                handle_s4a_pooled(conn, params)
            elif msg_type == "teardown_pool":
                handle_teardown_pool(conn)
            elif msg_type == "shutdown":
                handle_teardown_pool(conn)
                send_json_response(conn, {"status": "ok"})
                log_fn("[pool_server] Shutting down")
                # Give response time to flush
                time.sleep(0.1)
                os._exit(0)
            else:
                send_json_response(conn, {
                    "status": "error",
                    "message": f"unknown message type: {msg_type}",
                })
    finally:
        try:
            conn.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Concurrent LoRA Recovery Server")
    parser.add_argument("--listen-ip", type=str, default="10.10.1.3",
                        help="IP to bind (default: 10.10.1.3)")
    parser.add_argument("--listen-port", type=int, default=30200,
                        help="TCP control port (default: 30200)")
    parser.add_argument("--mlx-device", type=str, default="mlx5_0")
    parser.add_argument("--ib-port", type=int, default=1)
    args = parser.parse_args()

    print(f"[pool_server] Starting on {args.listen_ip}:{args.listen_port}", flush=True)
    print(f"[pool_server] RDMA device: {args.mlx_device}, port: {args.ib_port}", flush=True)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.listen_ip, args.listen_port))
    server_sock.listen(64)  # up to 64 concurrent connections
    print(f"[pool_server] Listening (backlog=64)", flush=True)

    try:
        while True:
            conn, addr = server_sock.accept()
            thread = threading.Thread(
                target=handle_request,
                args=(conn, addr, args.listen_ip, print),
                daemon=True,
            )
            thread.start()
    except KeyboardInterrupt:
        print("\n[pool_server] Interrupted", flush=True)
    finally:
        # Cleanup pool if still alive
        global _pool
        if _pool is not None:
            _pool.teardown()
            _pool = None
        server_sock.close()


if __name__ == "__main__":
    main()
