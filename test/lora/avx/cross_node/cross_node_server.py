#!/usr/bin/env python3
"""
Cross-Node LoRA Miss-Recovery Server (UM251 side)

Runs on UM251 (CPU-only node). Receives requests from UM253 and:
- Strategy 1: sends LoRA weights to UM253 via GLOO
- Strategy 2: receives activation, computes merge on CPU, sends result back
- Strategy 3: receives activation, bundles act+weights into contiguous buffer,
             single RDMA write back to UM253 GPU, GPU computes merge

Uses PyTorch Distributed GLOO backend for all communication.
TCP socket used for JSON control messages between the two nodes.
"""

import argparse
import json
import socket
import struct
import threading
import time
from typing import Any

import torch
import torch.distributed as dist

BYTES_PER_PARAM = 2  # bf16 / fp16


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Node LoRA Miss-Recovery Server (UM251)"
    )
    parser.add_argument("--rank", type=int, required=True,
                        help="Distributed rank of this node (1 for UM251)")
    parser.add_argument("--master-addr", type=str, required=True,
                        help="IP address of rank 0 (UM253)")
    parser.add_argument("--master-port", type=int, required=True,
                        help="Port for torch.distributed init (29500)")
    parser.add_argument("--listen-port", type=int, required=True,
                        help="TCP port for JSON control messages on UM251")
    parser.add_argument("--hidden-dim", type=int, default=2048,
                        help="Model hidden dimension H (default: 2048)")
    parser.add_argument("--intermediate-dim", type=int, default=1536,
                        help="Model intermediate dimension I (default: 1536)")
    parser.add_argument(
        "--ranks", type=str, default="16,32,64,128",
        help="Comma-separated LoRA ranks (default: 16,32,64,128)"
    )
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of benchmark iterations (default: 100)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Weight generation
# ---------------------------------------------------------------------------

def generate_lora_weights(
    rank: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_adapters: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Generate LoRA A [R, H] and B [R, I] tensors for num_adapters.
    Returns (A_tensors, B_tensors), each a list of bf16 pinned-memory tensors.
    """
    a_tensors: list[torch.Tensor] = []
    b_tensors: list[torch.Tensor] = []

    for _ in range(num_adapters):
        a = torch.empty(
            rank, hidden_dim,
            dtype=torch.bfloat16,
            pin_memory=True,
        ).normal_(mean=0.0, std=0.02)
        b = torch.empty(
            rank, intermediate_dim,
            dtype=torch.bfloat16,
            pin_memory=True,
        ).normal_(mean=0.0, std=0.02)
        a_tensors.append(a)
        b_tensors.append(b)

    return a_tensors, b_tensors


# ---------------------------------------------------------------------------
# Strategy 1: weight-transfer (CPU → UM253 via GLOO RDMA)
# ---------------------------------------------------------------------------

def run_strategy1(
    rank: int,
    num_miss: int,
    hidden_dim: int,
    intermediate_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """
    Generate LoRA A and B weights for num_miss adapters on UM251,
    then send them to UM253 (rank 0) via torch.distributed GLOO send.

    The client side times the full transfer + GPU upload pipeline.
    UM251 just sends; timing is reported back by the client.
    """
    # Generate weights for all miss adapters
    a_tensors, b_tensors = generate_lora_weights(
        rank=rank,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_adapters=num_miss,
    )

    # Send A tensors one by one (rank 0 is the destination)
    for a in a_tensors:
        dist.send(tensor=a, dst=0)

    # Send B tensors one by one
    for b in b_tensors:
        dist.send(tensor=b, dst=0)

    # The client sends back timing data after its benchmark loop.
    # We receive a small dict tensor (float32) with the timing fields.
    timing_tensor = torch.empty(6, dtype=torch.float32)
    dist.recv(timing_tensor, src=0)

    (
        rdma_weight_ms,
        total_ms,
        ep_bw_pct,
        gpu_compute_ms,
        gpu_upstream_ms,
        gpu_stream_ms,
    ) = timing_tensor.tolist()

    return {
        "type": "timing_result",
        "rank": rank,
        "num_miss": num_miss,
        "strategy": 1,
        "ep_bw_pct": int(ep_bw_pct),
        "rdma_weight_ms": rdma_weight_ms,
        "total_ms": total_ms,
        "gpu_compute_ms": gpu_compute_ms,
        "gpu_upstream_ms": gpu_upstream_ms,
        "gpu_stream_ms": gpu_stream_ms,
    }


# ---------------------------------------------------------------------------
# Strategy 2: activation-transfer + CPU compute
# ---------------------------------------------------------------------------

def run_strategy2(
    rank: int,
    num_miss: int,
    hidden_dim: int,
    intermediate_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """
    Receive activation tensor from UM253, compute the LoRA merge
    result = act @ A^T @ B^T on CPU (float32 BLAS), send result back.

    Timing breakdown is measured on UM251 (this server).
    """
    total_weights_bytes = 0
    for r in [16, 32, 64, 128]:
        total_weights_bytes += r * hidden_dim * 2  # A
        total_weights_bytes += r * intermediate_dim * 2  # B

    # ---- Warmup: receive + compute + send (warm the caches / interconnects) ----
    for _ in range(warmup):
        # Receive activation [seq_len, intermediate_dim] as bf16
        act = torch.empty(num_miss, intermediate_dim, dtype=torch.bfloat16)
        dist.recv(tensor=act, src=0)

        # Receive A and B for each miss adapter
        a_list: list[torch.Tensor] = []
        b_list: list[torch.Tensor] = []
        for _ in range(num_miss):
            a = torch.empty(rank, hidden_dim, dtype=torch.bfloat16)
            b = torch.empty(rank, intermediate_dim, dtype=torch.bfloat16)
            dist.recv(tensor=a, src=0)
            dist.recv(tensor=b, src=0)
            a_list.append(a)
            b_list.append(b)

        # CPU merge: result = act @ A^T @ B^T
        # act:      [num_miss, intermediate_dim]
        # A^T:      [hidden_dim, rank]
        # inter:    [num_miss, rank]
        # B^T:      [intermediate_dim, rank]
        # result:   [num_miss, rank]
        act_f32 = act.to(torch.float32)
        results: list[torch.Tensor] = []
        for a, b in zip(a_list, b_list):
            inter = act_f32 @ a.to(torch.float32).t()      # [num_miss, rank]
            result = inter @ b.to(torch.float32).t()         # [num_miss, rank]
            results.append(result)

        # Send result back to UM253
        for result in results:
            dist.send(tensor=result, dst=0)

    # ---- Timed iterations ----
    d2h_times: list[float] = []
    cpu_compute_times: list[float] = []
    rdma_result_times: list[float] = []
    total_times: list[float] = []

    for _ in range(iters):
        iter_start = time.perf_counter()

        # Step 1: receive activation from UM253
        act = torch.empty(num_miss, intermediate_dim, dtype=torch.bfloat16)
        dist.recv(tensor=act, src=0)
        recv_end = time.perf_counter()

        # Step 2: receive LoRA weights from UM253 (A and B per adapter)
        a_list: list[torch.Tensor] = []
        b_list: list[torch.Tensor] = []
        for _ in range(num_miss):
            a = torch.empty(rank, hidden_dim, dtype=torch.bfloat16)
            b = torch.empty(rank, intermediate_dim, dtype=torch.bfloat16)
            dist.recv(tensor=a, src=0)
            dist.recv(tensor=b, src=0)
            a_list.append(a)
            b_list.append(b)
        weights_recv_end = time.perf_counter()

        # Step 3: CPU merge compute (float32 BLAS)
        act_f32 = act.to(torch.float32)
        results: list[torch.Tensor] = []
        for a, b in zip(a_list, b_list):
            inter = act_f32 @ a.to(torch.float32).t()
            result = inter @ b.to(torch.float32).t()
            results.append(result)
        compute_end = time.perf_counter()

        # Step 4: send result back to UM253
        for result in results:
            dist.send(tensor=result, dst=0)
        send_end = time.perf_counter()

        d2h_ms = (weights_recv_end - recv_end) * 1000.0
        cpu_compute_ms = (compute_end - weights_recv_end) * 1000.0
        rdma_result_ms = (send_end - compute_end) * 1000.0
        iter_ms = (send_end - iter_start) * 1000.0

        d2h_times.append(d2h_ms)
        cpu_compute_times.append(cpu_compute_ms)
        rdma_result_times.append(rdma_result_ms)
        total_times.append(iter_ms)

    d2h_avg = sum(d2h_times) / len(d2h_times)
    cpu_compute_avg = sum(cpu_compute_times) / len(cpu_compute_times)
    rdma_result_avg = sum(rdma_result_times) / len(rdma_result_times)
    total_avg = sum(total_times) / len(total_times)

    return {
        "type": "timing_result",
        "rank": rank,
        "num_miss": num_miss,
        "strategy": 2,
        "ep_bw_pct": 0,
        "d2h_ms": d2h_avg,
        "rdma_activation_ms": 0.0,
        "cpu_compute_ms": cpu_compute_avg,
        "rdma_result_ms": rdma_result_avg,
        "h2d_ms": 0.0,
        "total_ms": total_avg,
    }


# ---------------------------------------------------------------------------
# Strategy 3: activation relay → UM253 GPU compute (pre-cached weights)
# ---------------------------------------------------------------------------

def run_strategy3(
    rank: int,
    num_miss: int,
    hidden_dim: int,
    intermediate_dim: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """
    Strategy 3 — weights are pre-cached on UM253 GPU from prior S1 transfer.

    Flow:
      1. UM251 receives activation from UM253 (via dist.recv)
      2. UM251 immediately sends activation back to UM253 GPU (via dist.send)
         — UM253 GPU pulls activation and computes with cached weights
      3. UM251 receives result from UM253 (via dist.recv)
      4. (No weights transferred in this step)

    Timing: rdma_activation_ms (recv+send activation), rdma_result_ms (recv result).
    GPU compute time is measured by UM253 client.
    """
    act_elem = hidden_dim  # [1, hidden_dim] flattened
    act_tensor = torch.empty(act_elem, dtype=torch.bfloat16, pin_memory=True)

    # Result size: [num_miss, 1, intermediate_dim] — one delta per adapter
    result_elem = num_miss * intermediate_dim
    result_tensor = torch.empty(result_elem, dtype=torch.bfloat16, pin_memory=True)

    # ---- Warmup ----
    for _ in range(warmup):
        # Step 1: receive activation from UM253
        dist.recv(tensor=act_tensor, src=0)
        # Step 2: relay activation back to UM253 GPU
        dist.send(tensor=act_tensor, dst=0)
        # Step 3: receive result from UM253
        dist.recv(tensor=result_tensor, src=0)

    # ---- Timed iterations ----
    rdma_act_times: list[float] = []
    rdma_res_times: list[float] = []

    for _ in range(iters):
        # Step 1: recv activation
        t0 = time.perf_counter()
        dist.recv(tensor=act_tensor, src=0)
        recv_end = time.perf_counter()

        # Step 2: send activation back to UM253 GPU (relay)
        dist.send(tensor=act_tensor, dst=0)
        relay_end = time.perf_counter()

        # Step 3: recv result from UM253 GPU
        dist.recv(tensor=result_tensor, src=0)
        t1 = time.perf_counter()

        rdma_act_times.append((relay_end - t0) * 1000.0)  # recv + send activation
        rdma_res_times.append((t1 - relay_end) * 1000.0)  # recv result

    act_avg = sum(rdma_act_times) / len(rdma_act_times) if rdma_act_times else 0.0
    res_avg = sum(rdma_res_times) / len(rdma_res_times) if rdma_res_times else 0.0

    return {
        "type": "timing_result",
        "rank": rank,
        "num_miss": num_miss,
        "strategy": 3,
        "ep_bw_pct": 0,
        "rdma_activation_ms": act_avg,   # recv + send activation
        "rdma_result_ms": res_avg,     # recv result
        "bundle_ms": 0.0,
        "bundle_bytes": 0,
        "total_ms": 0.0,  # GPU compute filled by client
    }


# ---------------------------------------------------------------------------
# Handle a single request from UM253
# ---------------------------------------------------------------------------

def handle_request(
    conn: socket.socket,
    addr: tuple[str, int],
    hidden_dim: int,
    intermediate_dim: int,
    warmup: int,
    iters: int,
    log_fn=print,
) -> None:
    """Receive JSON control message, run the requested strategy, send back result."""
    try:
        # Receive 4-byte length header (network byte order / big-endian)
        header = b""
        while len(header) < 4:
            chunk = conn.recv(4 - len(header))
            if not chunk:
                log_fn(f"[UM251] Connection closed by {addr} before sending header")
                return
            header += chunk

        msg_len = struct.unpack("!I", header)[0]

        # Receive JSON payload
        payload = b""
        while len(payload) < msg_len:
            chunk = conn.recv(msg_len - len(payload))
            if not chunk:
                log_fn(f"[UM251] Connection closed by {addr} during payload")
                return
            payload += chunk

        request = json.loads(payload.decode("utf-8"))
        log_fn(f"[UM251] Received request: {request}")

        req_type = request.get("type", "")
        rank = request.get("rank", 64)
        num_miss = request.get("num_miss", 1)
        req_warmup = request.get("warmup", warmup)
        req_iters = request.get("iters", iters)

        if req_type == "strategy1_transfer":
            result = run_strategy1(
                rank=rank,
                num_miss=num_miss,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                warmup=req_warmup,
                iters=req_iters,
            )
        elif req_type == "strategy2_compute":
            result = run_strategy2(
                rank=rank,
                num_miss=num_miss,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                warmup=req_warmup,
                iters=req_iters,
            )
        elif req_type == "strategy3_bundled":
            result = run_strategy3(
                rank=rank,
                num_miss=num_miss,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                warmup=req_warmup,
                iters=req_iters,
            )
        else:
            result = {
                "type": "error",
                "message": f"Unknown request type: {req_type}",
            }

        log_fn(f"[UM251] Sending response: {result}")

        response_bytes = json.dumps(result).encode("utf-8")
        response_header = struct.pack("!I", len(response_bytes))
        conn.sendall(response_header + response_bytes)

    except json.JSONDecodeError as e:
        log_fn(f"[UM251] JSON decode error from {addr}: {e}")
    except Exception as e:
        log_fn(f"[UM251] Error handling request from {addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main: initialise distributed + TCP socket server loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse LoRA ranks (stored as a set for fast lookup, though we generate per-request)
    ranks_list = [int(r.strip()) for r in args.ranks.split(",")]

    # ---- Initialise PyTorch Distributed (GLOO backend) ----
    # world_size=2: rank 0 = UM253 (GPU), rank 1 = UM251 (CPU)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=2,
        rank=args.rank,
    )
    print(f"[UM251] Distributed init done. rank={args.rank}, world_size=2, "
          f"backend=gloo, init=tcp://{args.master_addr}:{args.master_port}")

    # Verify the other rank is ready (barrier with rank 0)
    dist.barrier()
    print("[UM251] Barrier passed — both ranks initialised.")

    # ---- TCP socket server for JSON control messages ----
    listen_host = "0.0.0.0"
    listen_port = args.listen_port

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((listen_host, listen_port))
    server_sock.listen(8)
    print(f"[UM251] TCP control server listening on {listen_host}:{listen_port}")

    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(msg)

    try:
        while True:
            conn, addr = server_sock.accept()
            log(f"[UM251] Accepted connection from {addr}")
            thread = threading.Thread(
                target=handle_request,
                args=(
                    conn,
                    addr,
                    args.hidden_dim,
                    args.intermediate_dim,
                    args.warmup,
                    args.iters,
                    log,
                ),
                daemon=True,
            )
            thread.start()
    except KeyboardInterrupt:
        print("[UM251] Shutdown requested.")
    finally:
        server_sock.close()
        dist.destroy_process_group()
        print("[UM251] Process group destroyed. Server exit.")


if __name__ == "__main__":
    main()
