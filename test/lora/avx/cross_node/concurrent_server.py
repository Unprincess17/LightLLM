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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def send_json_response(conn: socket.socket, obj: dict[str, Any]) -> None:
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

    If *params* contains ``decompose: true``, each phase is wrapped with
    ``torch.cuda.synchronize()`` + ``perf_counter`` and per-segment durations
    (in microseconds) are returned in the response under the ``segments`` key.
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

        segments = []  # list of [name, duration_us] when decompose=True

        def _seg(name, fn, *args):
            """Run *fn*, optionally timing with CUDA sync."""
            if decompose:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ret = fn(*args)
            if decompose:
                torch.cuda.synchronize()
            segments.append([name, (time.perf_counter() - t0) * 1e6])
            return ret

        # RDMA READ activation from client's GPU buffer
        if decompose:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            transport.read_from_remote(0, 0, act_bytes)
            segments.append(["rdma_read", (time.perf_counter() - t0) * 1e6])
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
                # Warmup: run one miss to trigger cuBLAS algorithm selection
                a_warm = a_gpu[0].to(torch.float32)
                b_warm = b_gpu[0].to(torch.float32)
                inter_warm = act_f32 @ a_warm.T
                _ = (inter_warm @ b_warm).view(-1)
                torch.cuda.synchronize()

                # Capture one graph per weight index.
                # CUDA graphs require static tensor addresses — all intermediate
                # and output tensors must be pre-allocated and reused via in-place ops.
                graphs = []
                static_a = []
                static_b = []
                static_inter = []
                static_out = []
                for i in range(num_miss):
                    sa = torch.empty(rank, hidden_dim, dtype=torch.float32, device="cuda")
                    sb = torch.empty(rank, intermediate_dim, dtype=torch.float32, device="cuda")
                    si = torch.empty(1, rank, dtype=torch.float32, device="cuda")
                    so = torch.empty(intermediate_dim, dtype=torch.float32, device="cuda")
                    sa.copy_(a_gpu[i].to(torch.float32))
                    sb.copy_(b_gpu[i].to(torch.float32))
                    static_a.append(sa)
                    static_b.append(sb)
                    static_inter.append(si)
                    static_out.append(so)

                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        torch.mm(act_f32, sa.T, out=si)
                        torch.mm(si, sb, out=so.view(1, -1))
                    graphs.append(g)

                torch.cuda.synchronize()

                # Timed loop: replay graphs
                for i in range(num_miss):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    graphs[i].replay()
                    torch.cuda.synchronize()
                    segments.append([f"miss_{i}_alloc", (time.perf_counter() - t0) * 1e6])
                    segments.append([f"miss_{i}_dtype", 0.0])
                    segments.append([f"miss_{i}_mm1", 0.0])
                    segments.append([f"miss_{i}_mm2", 0.0])

                # Copy results from static output buffers to result tensor
                for i in range(num_miss):
                    result[i] = static_out[i]

            else:
                # Non-cuda_graph variants: alloc, dtype, mm1, mm2 per miss
                for i in range(num_miss):
                    if variant == "cache_flush":
                        torch.cuda.empty_cache()
                        _get_scratch_buf().zero_()
                        torch.cuda.synchronize()

                    # --- alloc ---
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    torch.cuda.synchronize()
                    segments.append([f"miss_{i}_alloc", (time.perf_counter() - t0) * 1e6])

                    # --- dtype ---
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    w_idx = 0 if variant == "same_weights" else i
                    a_f32 = a_gpu[w_idx].to(torch.float32)
                    b_f32 = b_gpu[w_idx].to(torch.float32)
                    torch.cuda.synchronize()
                    segments.append([f"miss_{i}_dtype", (time.perf_counter() - t0) * 1e6])

                    # --- mm1 ---
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    inter = act_f32 @ a_f32.T
                    torch.cuda.synchronize()
                    segments.append([f"miss_{i}_mm1", (time.perf_counter() - t0) * 1e6])

                    # --- mm2 ---
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result[i] = (inter @ b_f32).view(-1)
                    torch.cuda.synchronize()
                    segments.append([f"miss_{i}_mm2", (time.perf_counter() - t0) * 1e6])

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
                segments.append([f"miss_{i}", (time.perf_counter() - t0) * 1e6])
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
            segments.append(["rdma_write", (time.perf_counter() - t0) * 1e6])
        else:
            result_bf16 = result.to(torch.bfloat16)
            copy_gpu_to_gpu(
                transport.gpu_ptr() + act_bytes,
                result_bf16.data_ptr(),
                result_bytes,
            )
            transport.write_to_remote(act_bytes, act_bytes, result_bytes)

        if decompose:
            send_json_response(conn, {
                "status": "ok",
                "result_bytes": result_bytes,
                "segments": segments,
            })
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
    """Receive one JSON message, dispatch, send response, close."""
    try:
        header = _recv_exact(conn, 4)
        msg_len = struct.unpack("!I", header)[0]
        payload = _recv_exact(conn, msg_len)
        params = json.loads(payload.decode("utf-8"))
    except (EOFError, ConnectionResetError, json.JSONDecodeError) as e:
        log_fn(f"[pool_server] Bad request from {addr}: {e}")
        return

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
