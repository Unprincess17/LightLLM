#!/usr/bin/env python3
"""
Cross-Node LoRA Miss-Recovery Benchmark (UM253 side — GPU node)

Measures three recovery strategies for cross-node LoRA misses:

  Strategy 1 (weight-transfer):
    UM251 CPU → GLOO → UM253 GPU → GPU compute

  Strategy 2 (activation-transfer):
    UM253 GPU → CPU → GLOO → UM251 CPU compute → GLOO → UM253 CPU → GPU

  Strategy 3 (activation-transfer + GPU compute):
    UM253 GPU → D2H → pinned CPU → GLOO send act → UM251
    UM251 bundles [act | A_0 | B_0 | ...] → single GLOO send back
    → UM253 GPU extracts weights, computes merge, result stays on GPU
    (UM253 is both the inference node and the GPU compute node)

Variables: rank (16/32/64/128), num_miss (1/2/4),
           EP background traffic (0/25/50/75/90%)

Output: CSV with per-stage timing + metadata JSON
Figures: 3 plots (strategy comparison, latency breakdown, degradation ratio)

Usage (UM253 — this node):
  python cross_node_benchmark.py \
      --master-addr 10.10.1.3 \
      --master-port 29500 \
      --server-port 29501 \
      --listen-port 29502 \
      --ranks 16,32,64,128 \
      --num-miss-list 1,2,4 \
      --ep-bw-pct-list 0,25,50,75,90 \
      --warmup 10 \
      --iters 100 \
      --output-dir results/cross_node_benchmark \
      --mlx-device mlx5_0

Prerequisites on both nodes:
  - torch.distributed with GLOO backend
  - numpy, matplotlib
  - ib_write_bw (perftest) for EP traffic generation
  - UM251 must be running: python cross_node_server.py --port 29501
"""

import argparse
import csv as csv_lib
import json
import os
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

# ------------------------------------------------------------------
# Model dimensions (Qwen3-30B-A3B)
# ------------------------------------------------------------------
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 1536  # From spec (not 768 like single-machine)
BYTES_PER_PARAM = 2  # bf16


# ------------------------------------------------------------------
# Protocol constants
# ------------------------------------------------------------------

# Argument parsing
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Node LoRA Miss-Recovery Benchmark — UM253 (GPU) side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Network")
    g.add_argument(
        "--master-addr",
        default="10.10.1.1",
        help="IP address of rank 0 for torch.distributed rendezvous (UM253 IB IP)",
    )
    g.add_argument(
        "--server-addr",
        default="10.10.1.3",
        help="IP address of UM251 cross_node_server.py (TCP control)",
    )
    g.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Port for torch.distributed rendezvous on UM253",
    )
    g.add_argument(
        "--server-port",
        type=int,
        default=29501,
        help="TCP port where UM251 cross_node_server.py is listening",
    )
    g = parser.add_argument_group("Benchmark")
    g.add_argument(
        "--ranks",
        default="16,32,64,128",
        help="Comma-separated LoRA rank values",
    )
    g.add_argument(
        "--num-miss-list",
        default="1,2,4",
        help="Comma-separated number of missed LoRA adapters",
    )
    g.add_argument(
        "--ep-bw-pct-list",
        default="0,25,50,75,90",
        help="Comma-separated expert-parallel background traffic percentages",
    )
    g.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per config",
    )
    g.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Measurement iterations per config",
    )
    g.add_argument(
        "--output-dir",
        default="results/cross_node_benchmark",
        help="Directory for CSV, JSON, and figure output",
    )

    g = parser.add_argument_group("Model")
    g.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
        help="Model hidden dimension",
    )
    g.add_argument(
        "--intermediate-dim",
        type=int,
        default=INTERMEDIATE_DIM,
        help="Model intermediate (FFN) dimension",
    )

    g = parser.add_argument_group("EP Traffic")
    g.add_argument(
        "--mlx-device",
        default="mlx5_0",
        help="Mellanox RDMA device for ib_write_bw",
    )
    g.add_argument(
        "--local-ip",
        default="10.10.1.1",
        help="Local IP of this node (UM253) for EP traffic",
    )
    g.add_argument(
        "--remote-ip",
        default="10.10.1.3",
        help="Remote IP of UM251 for EP traffic",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Distributed setup
# ------------------------------------------------------------------

def setup_distributed(args: argparse.Namespace) -> int:
    """
    Initialise torch.distributed (GLOO backend) for this client.

    UM253 = rank 0 (GPU node)
    UM251 = rank 1 (CPU node)

    Returns the local rank (always 0 on UM253).
    """
    if dist.is_initialized():
        dist.destroy_process_group()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    # Ensure GLOO uses the IB interface, not loopback
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        # Auto-detect IB interface
        import subprocess as _sp
        try:
            result = _sp.run(
                ["ip", "-o", "link", "show"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "ibs" in line or "ibp" in line:
                    iface = line.split(": ")[1]
                    os.environ["GLOO_SOCKET_IFNAME"] = iface
                    print(f"[dist] Auto-detected GLOO_SOCKET_IFNAME={iface}", flush=True)
                    break
        except Exception:
            pass

    dist.init_process_group(
        backend="gloo",
        rank=0,          # always 0 on UM253
        world_size=2,
    )
    dist.barrier()
    return 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------
# Tensor generation
# ------------------------------------------------------------------

def generate_weights(
    rank: int,
    num_miss: int,
    intermediate_dim: int = INTERMEDIATE_DIM,
    hidden_dim: int = HIDDEN_DIM,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate LoRA A and B weight tensors.

    Returns:
        a_tensors: list of num_miss tensors of shape (rank, hidden_dim)
        b_tensors: list of num_miss tensors of shape (rank, intermediate_dim)

    So that: activation @ A^T @ B^T = [1, H] @ [H, R] @ [R, I] = [1, I]
    """
    a_tensors: List[torch.Tensor] = []
    b_tensors: List[torch.Tensor] = []

    for _ in range(num_miss):
        a = torch.empty((rank, hidden_dim), dtype=dtype)
        torch.nn.init.xavier_uniform_(a)
        a = a.pin_memory()
        a_tensors.append(a)

        b = torch.empty((rank, intermediate_dim), dtype=dtype)
        torch.nn.init.xavier_uniform_(b)
        b = b.pin_memory()
        b_tensors.append(b)

    return a_tensors, b_tensors


def generate_activation(
    hidden_dim: int = HIDDEN_DIM,
    seq_len: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Generate a single-token activation vector (row vector).

    Returns a pinned bf16 tensor of shape (seq_len, hidden_dim).
    """
    act = torch.empty((seq_len, hidden_dim), dtype=dtype)
    torch.nn.init.xavier_uniform_(act)
    act = act.pin_memory()
    return act


# ------------------------------------------------------------------
# Socket protocol helpers
# ------------------------------------------------------------------

def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    """Send a JSON-serialisable object with 4-byte big-endian length prefix."""
    payload = json.dumps(obj).encode("utf-8")
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def recv_json(sock: socket.socket) -> Dict[str, Any]:
    """Receive a JSON object with 4-byte big-endian length prefix."""
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise EOFError("Socket closed before receiving header")
        header += chunk
    msg_len = struct.unpack("!I", header)[0]
    payload = b""
    while len(payload) < msg_len:
        chunk = sock.recv(msg_len - len(payload))
        if not chunk:
            raise EOFError("Socket closed during payload")
        payload += chunk
    return json.loads(payload.decode("utf-8"))


# ------------------------------------------------------------------
# TCP client helpers
# ------------------------------------------------------------------

def connect_to_server(args: argparse.Namespace) -> socket.socket:
    """Connect TCP socket to UM251's cross_node_server.py."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120.0)
    sock.connect((args.server_addr, args.server_port))
    return sock


# ------------------------------------------------------------------
# Timing: Strategy 1 (weight-transfer)
#
#   UM253 sends LoRA A,B tensors → GLOO RDMA → UM251 (CPU)
#   UM253 also computes locally: activation @ A^T @ B^T on GPU
#
#   Measurement points:
#     rdma_weight_ms   — GLOO dist.send of A and B tensors (UM253 → UM251)
#     gpu_compute_ms   — activation @ A^T @ B^T on GPU (UM253 local)
#     total_ms         — wall-clock: send-start to GPU result ready
# ------------------------------------------------------------------

def run_strategy1_timing(
    rank: int,
    num_miss: int,
    warmup: int,
    iters: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    S1 (weight-transfer): UM251 generates LoRA weights, sends via GLOO.
    UM253 receives weights, uploads to GPU, computes LoRA merge.
    """
    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Pre-allocate receive buffers on pinned CPU
    a_bufs = [torch.empty(rank, args.hidden_dim, dtype=torch.bfloat16, pin_memory=True)
              for _ in range(num_miss)]
    b_bufs = [torch.empty(rank, args.intermediate_dim, dtype=torch.bfloat16, pin_memory=True)
              for _ in range(num_miss)]

    # GPU activation (reused across iterations)
    activation_gpu = generate_activation(args.hidden_dim).to(device_gpu)

    rdma_weight_times: List[float] = []
    gpu_compute_times: List[float] = []

    sock = connect_to_server(args)
    try:
        send_json(sock, {
            "type": "strategy1_transfer",
            "rank": rank,
            "num_miss": num_miss,
            "warmup": warmup,
            "iters": iters,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
        })

        for phase_name, n_iters in [("warmup", warmup), ("measure", iters)]:
            for _ in range(n_iters):
                # Receive weights from UM251 via GLOO
                t0 = time.perf_counter()
                for a_buf, b_buf in zip(a_bufs, b_bufs):
                    dist.recv(tensor=a_buf, src=1)
                    dist.recv(tensor=b_buf, src=1)
                t1 = time.perf_counter()
                rdma_weight_ms = (t1 - t0) * 1000.0

                # Upload to GPU and compute
                gpu_t0 = time.perf_counter()
                a_gpu = [a.to(device_gpu, non_blocking=True) for a in a_bufs]
                b_gpu = [b.to(device_gpu, non_blocking=True) for b in b_bufs]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                with torch.no_grad():
                    for a_g, b_g in zip(a_gpu, b_gpu):
                        # a_g: [R, H], b_g: [R, I]
                        # act [1, H] @ A^T [H, R] @ B [R, I] = [1, I] (LoRA delta)
                        tmp = activation_gpu @ a_g.t()        # [1, H] @ [H, R] = [1, R]
                        delta = tmp @ b_g                     # [1, R] @ [R, I] = [1, I]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_t1 = time.perf_counter()
                gpu_compute_ms = (gpu_t1 - gpu_t0) * 1000.0

                if phase_name == "measure":
                    rdma_weight_times.append(rdma_weight_ms)
                    gpu_compute_times.append(gpu_compute_ms)

        # Send timing summary to server
        timing = torch.tensor(
            [statistics.median(rdma_weight_times) if rdma_weight_times else 0.0,
             statistics.median(gpu_compute_times) if gpu_compute_times else 0.0],
            dtype=torch.float32,
        )
        dist.send(tensor=timing, dst=1)

        # Receive server's result bundle
        result_bundle = recv_json(sock)

    finally:
        sock.close()

    median_rdma = statistics.median(rdma_weight_times) if rdma_weight_times else 0.0
    median_gpu = statistics.median(gpu_compute_times) if gpu_compute_times else 0.0

    return {
        "rank": rank,
        "num_miss": num_miss,
        "strategy": 1,
        "rdma_weight_ms": round(median_rdma, 4),
        "gpu_compute_ms": round(median_gpu, 4),
        "total_ms": round(median_rdma + median_gpu, 4),
    }


# ------------------------------------------------------------------
# Timing: Strategy 2 (activation-transfer)
#
#   UM253 GPU activation → D2H → pinned CPU
#   Pinned CPU → GLOO RDMA → UM251 CPU
#   UM251 CPU: activation @ A^T @ B^T (pre-loaded on server)
#   UM251 result → GLOO RDMA → UM253 pinned CPU → H2D → UM253 GPU
#
#   Measurement points:
#     d2h_ms              — GPU → pinned CPU copy (UM253)
#     rdma_activation_ms  — dist.send of activation (UM253 → UM251)
#     rdma_result_ms      — dist.recv of result (UM251 → UM253)
#     h2d_ms              — pinned CPU → GPU copy (UM253)
#     total_ms            — D2H to result-on-GPU wall-clock
# ------------------------------------------------------------------

def run_strategy2_timing(
    rank: int,
    num_miss: int,
    warmup: int,
    iters: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    S2 (activation-transfer + CPU compute):
    UM253 sends activation [1, H] to UM251 via GLOO.
    UM251 computes act @ A^T @ B^T on CPU, sends result [num_miss, I] back.
    """
    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    activation_gpu = generate_activation(args.hidden_dim).to(device_gpu)
    act_pinned = torch.empty(1, args.hidden_dim, dtype=torch.bfloat16, pin_memory=True)
    result_pinned = torch.empty(num_miss, args.intermediate_dim, dtype=torch.bfloat16, pin_memory=True)

    d2h_times: List[float] = []
    rdma_act_times: List[float] = []
    rdma_res_times: List[float] = []
    h2d_times: List[float] = []

    sock = connect_to_server(args)
    try:
        send_json(sock, {
            "type": "strategy2_compute",
            "rank": rank,
            "num_miss": num_miss,
            "warmup": warmup,
            "iters": iters,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
        })

        for phase_name, n_iters in [("warmup", warmup), ("measure", iters)]:
            for _ in range(n_iters):
                # D2H: GPU → pinned CPU
                d2h_t0 = time.perf_counter()
                act_cpu = activation_gpu.to("cpu", non_blocking=True)
                act_pinned.copy_(act_cpu, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                d2h_t1 = time.perf_counter()

                # Send activation to UM251
                act_t0 = time.perf_counter()
                dist.send(tensor=act_pinned, dst=1)
                act_t1 = time.perf_counter()

                # Receive result from UM251: [num_miss, I]
                res_t0 = time.perf_counter()
                dist.recv(tensor=result_pinned, src=1)
                res_t1 = time.perf_counter()

                # H2D: pinned CPU → GPU
                h2d_t0 = time.perf_counter()
                result_gpu = result_pinned.to(device_gpu, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                h2d_t1 = time.perf_counter()

                if phase_name == "measure":
                    d2h_times.append((d2h_t1 - d2h_t0) * 1000.0)
                    rdma_act_times.append((act_t1 - act_t0) * 1000.0)
                    rdma_res_times.append((res_t1 - res_t0) * 1000.0)
                    h2d_times.append((h2d_t1 - h2d_t0) * 1000.0)

        # Receive server's result bundle
        result_bundle = recv_json(sock)

    finally:
        sock.close()

    def _med(lst: List[float]) -> float:
        return round(statistics.median(lst), 4) if lst else 0.0

    total_times = [
        a + b + c + d
        for a, b, c, d in zip(d2h_times, rdma_act_times, rdma_res_times, h2d_times)
    ]

    return {
        "rank": rank,
        "num_miss": num_miss,
        "strategy": 2,
        "d2h_ms": _med(d2h_times),
        "rdma_activation_ms": _med(rdma_act_times),
        "rdma_result_ms": _med(rdma_res_times),
        "h2d_ms": _med(h2d_times),
        "cpu_compute_ms": result_bundle.get("cpu_compute_ms", 0.0),
        "total_ms": round(statistics.median(total_times), 4) if total_times else 0.0,
    }


# ------------------------------------------------------------------
# Timing: Strategy 3 (activation relay → GPU compute, pre-cached weights)
#
#   Weights are PRE-CACHED on UM253 GPU from a prior S1 transfer.
#   This step only transfers activation + result — zero weight transfer.
#
#   Flow:
#     1. UM253 GPU → D2H → pinned CPU: activation
#     2. Pinned CPU → GLOO → UM251: activation (small, ~H bytes)
#     3. UM251: relay activation back to UM253 GPU via GLOO
#     4. UM253 GPU: LoRA compute with PRE-CACHED weights
#     5. UM253 GPU → GLOO → UM251: result (~num_miss*I bytes)
#
#   Measurement points:
#     d2h_ms             — GPU → pinned CPU copy (activation)
#     rdma_activation_ms  — RDMA round-trip: UM253→UM251→UM253 (activation relay)
#     gpu_compute_ms     — activation @ A^T @ B^T on GPU (pre-cached weights)
#     rdma_result_ms     — dist.send of result (UM253 → UM251)
#     total_ms           — D2H to result-sent wall-clock
# ------------------------------------------------------------------

def run_strategy3_timing(
    rank: int,
    num_miss: int,
    warmup: int,
    iters: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    S3 (activation relay → GPU compute, pre-cached weights):
    Weights are PRE-CACHED on UM253 GPU. No weight transfer.

    Flow:
      1. UM253 GPU → D2H → pinned CPU: activation [1, H]
      2. Pinned CPU → GLOO → UM251: activation
      3. UM251 relays activation back to UM253 via GLOO
      4. UM253 GPU: LoRA compute with pre-cached weights
      5. UM253 → GLOO → UM251: result [num_miss, I]
    """
    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    activation_gpu = generate_activation(args.hidden_dim).to(device_gpu)

    # Pre-cached weights on GPU
    a_gpu_list: List[torch.Tensor] = []
    b_gpu_list: List[torch.Tensor] = []
    for _ in range(num_miss):
        a_g = torch.randn(rank, args.hidden_dim, device=device_gpu, dtype=torch.bfloat16) / (rank ** 0.5)
        b_g = torch.randn(rank, args.intermediate_dim, device=device_gpu, dtype=torch.bfloat16) / (rank ** 0.5)
        a_gpu_list.append(a_g)
        b_gpu_list.append(b_g)

    # Pinned CPU buffers
    act_pinned = torch.empty(args.hidden_dim, dtype=torch.bfloat16, pin_memory=True)
    result_pinned = torch.empty(num_miss * args.intermediate_dim, dtype=torch.bfloat16, pin_memory=True)

    d2h_times: List[float] = []
    rdma_act_times: List[float] = []
    gpu_compute_times: List[float] = []
    rdma_result_times: List[float] = []

    sock = connect_to_server(args)
    try:
        send_json(sock, {
            "type": "strategy3_bundled",
            "rank": rank,
            "num_miss": num_miss,
            "warmup": warmup,
            "iters": iters,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
        })

        for phase_name, n_iters in [("warmup", warmup), ("measure", iters)]:
            for _ in range(n_iters):
                # D2H: GPU → pinned CPU
                d2h_t0 = time.perf_counter()
                act_cpu = activation_gpu.view(-1).to("cpu", non_blocking=True)
                act_pinned.copy_(act_cpu, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                d2h_t1 = time.perf_counter()

                # RDMA activation relay: send to UM251, recv back
                rdma_act_t0 = time.perf_counter()
                dist.send(tensor=act_pinned, dst=1)
                dist.recv(tensor=act_pinned, src=1)
                rdma_act_t1 = time.perf_counter()

                # GPU compute with pre-cached weights
                act_gpu = act_pinned.view(1, args.hidden_dim).to(device_gpu)
                gpu_t0 = time.perf_counter()
                with torch.no_grad():
                    for a_g, b_g in zip(a_gpu_list, b_gpu_list):
                        # a_g: [R, H], b_g: [R, I]
                        tmp = act_gpu @ a_g.t()     # [1, H] @ [H, R] = [1, R]
                        delta = tmp @ b_g            # [1, R] @ [R, I] = [1, I]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_t1 = time.perf_counter()

                # Send result to UM251
                rdma_res_t0 = time.perf_counter()
                dist.send(tensor=result_pinned, dst=1)
                rdma_res_t1 = time.perf_counter()

                if phase_name == "measure":
                    d2h_times.append((d2h_t1 - d2h_t0) * 1000.0)
                    rdma_act_times.append((rdma_act_t1 - rdma_act_t0) * 1000.0)
                    gpu_compute_times.append((gpu_t1 - gpu_t0) * 1000.0)
                    rdma_result_times.append((rdma_res_t1 - rdma_res_t0) * 1000.0)

        result_bundle = recv_json(sock)

    finally:
        sock.close()

    def _med(lst: List[float]) -> float:
        return round(statistics.median(lst), 4) if lst else 0.0

    total_times = [
        a + b + c + d
        for a, b, c, d in zip(d2h_times, rdma_act_times, gpu_compute_times, rdma_result_times)
    ]

    return {
        "rank": rank,
        "num_miss": num_miss,
        "strategy": 3,
        "d2h_ms": _med(d2h_times),
        "rdma_activation_ms": _med(rdma_act_times),
        "gpu_compute_ms": _med(gpu_compute_times),
        "rdma_result_ms": _med(rdma_result_times),
        "bundle_ms": 0.0,
        "bundle_bytes": 0,
        "total_ms": round(statistics.median(total_times), 4) if total_times else 0.0,
    }


# ------------------------------------------------------------------
# Benchmark orchestration
# ------------------------------------------------------------------

def run_benchmark_for_config(
    rank: int,
    num_miss: int,
    warmup: int,
    iters: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run all three strategies for a single (rank, num_miss) config.

    Returns (timing1, timing2, timing3).
    """
    print(
        f"  [config] rank={rank}, num_miss={num_miss}, "
        f"warmup={warmup}, iters={iters}",
        flush=True,
    )

    t1 = run_strategy1_timing(
        rank=rank, num_miss=num_miss,
        warmup=warmup, iters=iters,
        args=args,
    )
    t2 = run_strategy2_timing(
        rank=rank, num_miss=num_miss,
        warmup=warmup, iters=iters,
        args=args,
    )
    t3 = run_strategy3_timing(
        rank=rank, num_miss=num_miss,
        warmup=warmup, iters=iters,
        args=args,
    )

    print(
        f"  [result] rank={rank}, num_miss={num_miss} → "
        f"S1_total={t1['total_ms']:.3f}ms "
        f"(rdma_weight={t1['rdma_weight_ms']:.3f}ms, "
        f"gpu={t1['gpu_compute_ms']:.3f}ms) | "
        f"S2_total={t2['total_ms']:.3f}ms "
        f"(d2h={t2['d2h_ms']:.3f}ms, "
        f"rdma_act={t2['rdma_activation_ms']:.3f}ms, "
        f"rdma_res={t2['rdma_result_ms']:.3f}ms, "
        f"h2d={t2['h2d_ms']:.3f}ms) | "
        f"S3_total={t3['total_ms']:.3f}ms "
        f"(d2h={t3['d2h_ms']:.3f}ms, "
        f"rdma_act={t3['rdma_activation_ms']:.3f}ms, "
        f"gpu={t3['gpu_compute_ms']:.3f}ms)",
        flush=True,
    )

    return t1, t2, t3


def collect_system_metadata() -> Dict[str, Any]:
    """Gather system-level metadata for the result JSON."""
    meta: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "gpu_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        meta["cuda_memory_allocated_gb"] = (
            torch.cuda.memory_allocated(0) / 1024**3
        )
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=pcie.link.width.current",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            meta["pcie_width"] = result.stdout.strip()
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["ibstat"],
            capture_output=True, text=True, timeout=5,
        )
        meta["ib_devices"] = result.stdout[:500]
    except Exception:
        pass

    return meta


# ------------------------------------------------------------------
# CSV output
# ------------------------------------------------------------------

CSV_COLUMNS = [
    "rank", "num_miss", "strategy", "ep_bw_pct",
    "rdma_weight_ms", "gpu_compute_ms",
    "d2h_ms", "rdma_activation_ms", "rdma_result_ms", "h2d_ms", "cpu_compute_ms",
    "bundle_ms", "bundle_bytes",
    "total_ms",
]


def write_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv_lib.DictWriter(
            f, fieldnames=CSV_COLUMNS, extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"[output] CSV written → {output_path}", flush=True)


def write_metadata(
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[output] metadata JSON → {output_path}", flush=True)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(csv_path: Path, output_dir: Path) -> None:
    """
    Generate three publication-quality figures from benchmark CSV:

      Fig 3a: Strategy comparison — total latency vs EP BW%
               Multiple lines for (rank, num_miss), solid=S1, dashed=S2

      Fig 3b: Latency breakdown — stacked bar for rank=32, num_miss=2,
              EP=0%% vs EP=75%%

      Fig 3c: Degradation ratio — latency(EP=X%) / latency(EP=0%)
              comparing Strategy 1 vs Strategy 2
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        print(f"[plot] matplotlib not available: {exc}")
        return

    rows: List[Dict[str, Any]] = []
    with csv_path.open() as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            # Convert numeric columns
            for col in CSV_COLUMNS:
                if col in row and row[col]:
                    row[col] = float(row[col])
            rows.append(row)

    if not rows:
        print("[plot] No data in CSV to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Colour / marker map for (rank, num_miss) combinations
    color_map = {
        (16, 1):  "#1f77b4",
        (16, 2):  "#aec7e8",
        (16, 4):  "#ff7f0e",
        (32, 1):  "#2ca02c",
        (32, 2):  "#98df8a",
        (32, 4):  "#d62728",
        (64, 1):  "#9467bd",
        (64, 2):  "#c5b0d5",
        (64, 4):  "#8c564b",
        (128, 1): "#e377c2",
        (128, 2): "#f7b6d2",
        (128, 4): "#7f7f7f",
    }
    marker_cycle = ["o", "s", "^", "D", "v", "p", "h", "*"]

    data_s1 = [r for r in rows if r["strategy"] == 1]
    data_s2 = [r for r in rows if r["strategy"] == 2]
    data_s3 = [r for r in rows if r["strategy"] == 3]

    def groupby(
        data: List[Dict[str, Any]],
        keys: List[str],
    ) -> Dict[Tuple, List[Dict[str, Any]]]:
        groups: Dict[Tuple, List[Dict[str, Any]]] = {}
        for row in data:
            key = tuple(row[k] for k in keys)
            groups.setdefault(key, []).append(row)
        return groups

    # ------------------------------------------------------------------
    # Fig 3a — Strategy comparison: total_ms vs ep_bw_pct
    # ------------------------------------------------------------------
    fig3a, ax3a = plt.subplots(figsize=(12, 7))

    def _plot_group(
        data: List[Dict[str, Any]],
        strategy_label: str,
        linestyle: str,
    ) -> None:
        grouped = groupby(data, ["rank", "num_miss"])
        m_idx = 0
        for (rank, num_miss), config_rows in sorted(grouped.items()):
            color = color_map.get((rank, num_miss), "#333333")
            marker = marker_cycle[m_idx % len(marker_cycle)]
            m_idx += 1

            config_rows = sorted(config_rows, key=lambda r: r["ep_bw_pct"])
            x = [r["ep_bw_pct"] for r in config_rows]
            y = [r["total_ms"] for r in config_rows]

            label = f"Rank={rank} N={num_miss} [{strategy_label}]"
            ax3a.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=7,
                linewidth=2,
                label=label,
            )

    _plot_group(data_s1, "S1", "solid")
    _plot_group(data_s2, "S2", "dashed")
    _plot_group(data_s3, "S3", "dashdot")

    ax3a.set_xlabel("EP Background Traffic (%)", fontsize=13)
    ax3a.set_ylabel("Total Recovery Latency (ms)", fontsize=13)
    ax3a.set_title(
        "Fig 3a — Cross-Node LoRA Recovery: Strategy 1 vs Strategy 2 vs Strategy 3",
        fontsize=13,
    )
    ax3a.set_xlim(-2, 95)
    ax3a.grid(True, alpha=0.3)
    ax3a.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    fig3a.tight_layout()
    out3a = output_dir / "fig3a_strategy_comparison.png"
    fig3a.savefig(out3a, dpi=150)
    plt.close(fig3a)
    print(f"[plot] Fig 3a saved → {out3a}", flush=True)

    # ------------------------------------------------------------------
    # Fig 3b — Stacked bar: rank=32, num_miss=2, EP=0 vs EP=75, all 3 strategies
    # ------------------------------------------------------------------
    TARGET_RANK = 32
    TARGET_NUM_MISS = 2
    TARGET_EP_VALS = [0, 75]

    # S1: [RDMA weight transfer, GPU compute]
    # S2: [D2H, RDMA act, CPU compute, RDMA result, H2D]
    # S3: [D2H, RDMA activation (round-trip relay), GPU compute, RDMA result]
    breakdown_keys_s1 = ["rdma_weight_ms", "gpu_compute_ms"]
    breakdown_labels_s1 = ["RDMA Weight Transfer", "GPU Compute"]
    breakdown_keys_s2 = [
        "d2h_ms", "rdma_activation_ms", "cpu_compute_ms", "rdma_result_ms", "h2d_ms",
    ]
    breakdown_labels_s2 = ["D2H Copy", "RDMA Activation", "CPU Compute", "RDMA Result", "H2D Copy"]
    breakdown_keys_s3 = [
        "d2h_ms", "rdma_activation_ms", "gpu_compute_ms", "rdma_result_ms",
    ]
    breakdown_labels_s3 = ["D2H Copy", "RDMA Act Relay", "GPU Compute", "RDMA Result"]

    # Colors per segment type (shared across strategies)
    seg_colors = {
        "RDMA Weight Transfer": "#4c72b0",
        "GPU Compute":          "#c44e52",
        "D2H Copy":            "#81D4FA",
        "RDMA Activation":     "#1976D2",
        "RDMA Act Relay":       "#1976D2",
        "CPU Compute":          "#388E3C",
        "RDMA Result":         "#7B1FA2",
        "H2D Copy":            "#F57C00",
    }

    fig3b, axes3b = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ep_val in zip(axes3b, TARGET_EP_VALS):
        rows = {
            1: next((r for r in data_s1 if r["rank"] == TARGET_RANK
                      and r["num_miss"] == TARGET_NUM_MISS and r["ep_bw_pct"] == ep_val), None),
            2: next((r for r in data_s2 if r["rank"] == TARGET_RANK
                      and r["num_miss"] == TARGET_NUM_MISS and r["ep_bw_pct"] == ep_val), None),
            3: next((r for r in data_s3 if r["rank"] == TARGET_RANK
                      and r["num_miss"] == TARGET_NUM_MISS and r["ep_bw_pct"] == ep_val), None),
        }

        n_strategies = 3
        bar_width = 0.22
        x_base = [0, 1, 2]  # positions for S1, S2, S3

        all_labels: List[str] = []
        all_colors: List[str] = []

        # Collect all unique labels and assign x-positions
        # Strategy 1 bar
        for key, label in zip(breakdown_keys_s1, breakdown_labels_s1):
            color = seg_colors.get(label, "#999999")
            vals = [0.0, 0.0, 0.0]
            if rows[1]:
                vals[0] = rows[1].get(key, 0.0) or 0.0
            bottom = [0.0, 0.0, 0.0]
            ax.bar([x_base[0]], vals, bar_width, bottom=bottom,
                   color=color, edgecolor="white", label=label if label not in all_labels else "")
            if label not in all_labels:
                all_labels.append(label)
                all_colors.append(color)
            # Annotate
            if vals[0] >= 0.01:
                ax.text(x_base[0], vals[0] / 2, f"{vals[0]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

        # Strategy 2 bar
        for key, label in zip(breakdown_keys_s2, breakdown_labels_s2):
            color = seg_colors.get(label, "#999999")
            vals = [0.0, 0.0, 0.0]
            if rows[2]:
                vals[1] = rows[2].get(key, 0.0) or 0.0
            # Stack on previous segments for S2
            bottom_s2 = sum(
                rows[2].get(k, 0.0) or 0.0
                for k in breakdown_keys_s2[:breakdown_keys_s2.index(key)]
            ) if rows[2] else 0.0
            ax.bar([x_base[1]], [vals[1]], bar_width, bottom=[bottom_s2],
                   color=color, edgecolor="white", label=label if label not in all_labels else "")
            if label not in all_labels:
                all_labels.append(label)
                all_colors.append(color)
            if vals[1] >= 0.01:
                ax.text(x_base[1], bottom_s2 + vals[1] / 2, f"{vals[1]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

        # Strategy 3 bar
        for key, label in zip(breakdown_keys_s3, breakdown_labels_s3):
            color = seg_colors.get(label, "#999999")
            vals = [0.0, 0.0, 0.0]
            if rows[3]:
                vals[2] = rows[3].get(key, 0.0) or 0.0
            bottom_s3 = sum(
                rows[3].get(k, 0.0) or 0.0
                for k in breakdown_keys_s3[:breakdown_keys_s3.index(key)]
            ) if rows[3] else 0.0
            ax.bar([x_base[2]], [vals[2]], bar_width, bottom=[bottom_s3],
                   color=color, edgecolor="white", label=label if label not in all_labels else "")
            if label not in all_labels:
                all_labels.append(label)
                all_colors.append(color)
            if vals[2] >= 0.01:
                ax.text(x_base[2], bottom_s3 + vals[2] / 2, f"{vals[2]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

        # Total latency labels on top
        for strat, xb in enumerate(x_base, 1):
            row = rows[strat]
            if row:
                total = row.get("total_ms", 0.0) or 0.0
                if total > 0:
                    ax.text(xb, total + 0.3, f"S{strat}\n{total:.2f}ms",
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold")

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"Strategy {i}" for i in range(1, n_strategies + 1)], fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title(f"Rank={TARGET_RANK}, N={TARGET_NUM_MISS}, EP={ep_val}%", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(handles=[
            mpatches.Patch(color=all_colors[i], label=all_labels[i])
            for i in range(len(all_labels))
        ], fontsize=7, loc="upper right", ncol=1)

    fig3b.suptitle(
        f"Fig 3b — Latency Breakdown (Rank={TARGET_RANK}, NumMiss={TARGET_NUM_MISS})",
        fontsize=13,
    )
    fig3b.tight_layout()
    out3b = output_dir / "fig3b_latency_breakdown.png"
    fig3b.savefig(out3b, dpi=150)
    plt.close(fig3b)
    print(f"[plot] Fig 3b saved → {out3b}", flush=True)

    # ------------------------------------------------------------------
    # Fig 3c — Degradation ratio: latency(EP=X%) / latency(EP=0%)
    # ------------------------------------------------------------------
    fig3c, ax3c = plt.subplots(figsize=(12, 7))

    def _compute_ratios(
        data: List[Dict[str, Any]],
        strategy_num: int,
    ) -> List[Tuple]:
        """Return sorted list of (rank, num_miss, ep_pct, ratio)."""
        # Build baseline map: (rank, num_miss) -> total_ms at EP=0
        baseline: Dict[Tuple, float] = {}
        for row in data:
            if row["strategy"] != strategy_num:
                continue
            key = (row["rank"], row["num_miss"])
            if row["ep_bw_pct"] == 0:
                baseline[key] = row["total_ms"]

        ratios: List[Tuple] = []
        for row in data:
            if row["strategy"] != strategy_num:
                continue
            key = (row["rank"], row["num_miss"])
            if row["ep_bw_pct"] > 0:
                bl = baseline.get(key)
                if bl and bl > 0:
                    ratios.append(
                        (row["rank"], row["num_miss"], row["ep_bw_pct"],
                         row["total_ms"] / bl)
                    )
        return sorted(ratios, key=lambda r: (r[0], r[1], r[2]))

    for data, strat_label, linestyle, marker in [
        (data_s1, "S1", "solid", "o"),
        (data_s2, "S2", "dashed", "s"),
    ]:
        strat_num = 1 if strat_label == "S1" else 2
        ratios = _compute_ratios(data, strat_num)
        grouped = groupby(
            [{"r": r[0], "n": r[1], "ep": r[2], "ratio": r[3]} for r in ratios],
            ["r", "n"],
        )
        for (rank, num_miss), items in sorted(grouped.items()):
            color = color_map.get((rank, num_miss), "#333333")
            x = [it["ep"] for it in items]
            y = [it["ratio"] for it in items]
            ax3c.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=7,
                linewidth=2,
                label=f"Rank={rank} N={num_miss} [{strat_label}]",
            )

    ax3c.axhline(
        y=1.0,
        color="black",
        linestyle=":",
        linewidth=1,
        label="Baseline (EP=0%)",
    )
    ax3c.set_xlabel("EP Background Traffic (%)", fontsize=13)
    ax3c.set_ylabel("Degradation Ratio (vs EP=0%)", fontsize=13)
    ax3c.set_title(
        "Fig 3c — Latency Degradation Under EP Background Traffic",
        fontsize=13,
    )
    ax3c.set_xlim(-2, 95)
    ax3c.grid(True, alpha=0.3)
    ax3c.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    fig3c.tight_layout()
    out3c = output_dir / "fig3c_degradation_ratio.png"
    fig3c.savefig(out3c, dpi=150)
    plt.close(fig3c)
    print(f"[plot] Fig 3c saved → {out3c}", flush=True)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve list arguments
    ranks = [int(x) for x in args.ranks.split(",")]
    num_miss_list = [int(x) for x in args.num_miss_list.split(",")]
    ep_bw_pct_list = [int(x) for x in args.ep_bw_pct_list.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Cross-Node LoRA Miss-Recovery Benchmark (UM253 side)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Master addr    : {args.master_addr}:{args.master_port}", flush=True)
    print(f"  Server port    : {args.server_port}", flush=True)
    print(f"  Ranks          : {ranks}", flush=True)
    print(f"  NumMiss list   : {num_miss_list}", flush=True)
    print(f"  EP BW pct list : {ep_bw_pct_list}", flush=True)
    print(f"  Warmup/Iters   : {args.warmup}/{args.iters}", flush=True)
    print(f"  Hidden/Intermed: {args.hidden_dim}/{args.intermediate_dim}", flush=True)
    print(f"  Output dir     : {output_dir}", flush=True)
    print(f"  GPU available  : {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU device     : {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 60, flush=True)

    # ----------------------------------------------------------------
    # Import EP traffic generator (companion script in same directory)
    # ----------------------------------------------------------------
    _script_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(_script_dir))
    try:
        from ep_traffic_generator import EPTrafficGenerator, EPTrafficConfig
    except ImportError as exc:
        print(f"[fatal] Could not import ep_traffic_generator: {exc}", flush=True)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Setup distributed (GLOO backend — UM253 = rank 0, UM251 = rank 1)
    # ----------------------------------------------------------------
    setup_distributed(args)

    all_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    ep_gen: Optional[EPTrafficGenerator] = None
    ep_config = EPTrafficConfig(
        local_ip=args.local_ip,
        remote_ip=args.remote_ip,
        mlx_device=args.mlx_device,
    )

    try:
        for ep_pct in ep_bw_pct_list:
            print(f"\n>>> EP background traffic: {ep_pct}%", flush=True)

            # Start EP traffic for this group
            ep_gen = EPTrafficGenerator(ep_config, mode="continuous")
            try:
                ep_gen.start(bw_pct=ep_pct)
            except Exception as exc:
                print(f"[warning] EP traffic start failed: {exc}", flush=True)
                ep_gen = None

            for rank in ranks:
                for num_miss in num_miss_list:
                    try:
                        t1, t2, t3 = run_benchmark_for_config(
                            rank=rank,
                            num_miss=num_miss,
                            warmup=args.warmup,
                            iters=args.iters,
                            args=args,
                        )
                        t1["ep_bw_pct"] = ep_pct
                        t2["ep_bw_pct"] = ep_pct
                        t3["ep_bw_pct"] = ep_pct
                        all_results.extend([t1, t2, t3])
                    except Exception as exc:
                        print(
                            f"[error] rank={rank} num_miss={num_miss}: {exc}",
                            flush=True,
                        )
                        errors.append({
                            "rank": rank,
                            "num_miss": num_miss,
                            "ep_bw_pct": ep_pct,
                            "error": str(exc),
                        })

            # Stop EP traffic before next group
            if ep_gen is not None:
                ep_gen.stop()
                ep_gen = None

            # EP group complete

    except KeyboardInterrupt:
        print("\n[interrupt] Received Ctrl+C — saving partial results", flush=True)

    finally:
        if ep_gen is not None:
            ep_gen.stop()
        cleanup_distributed()

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    csv_path = output_dir / "cross_node_benchmark.csv"
    write_csv(all_results, csv_path)

    metadata = {
        "args": vars(args),
        "ranks": ranks,
        "num_miss_list": num_miss_list,
        "ep_bw_pct_list": ep_bw_pct_list,
        "system": collect_system_metadata(),
        "errors": errors,
        "total_configs": (
            len(ranks) * len(num_miss_list) * len(ep_bw_pct_list)
        ),
        "completed_configs": len(all_results) // 3,
    }
    meta_path = output_dir / "cross_node_benchmark_meta.json"
    write_metadata(metadata, meta_path)

    # Generate plots
    if all_results:
        plot_results(csv_path, output_dir)

    print(
        f"\n[DONE] Results: {len(all_results)} rows | "
        f"Errors: {len(errors)} | "
        f"CSV: {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
