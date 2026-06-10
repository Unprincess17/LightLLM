#!/usr/bin/env python3
"""EP-like bidirectional traffic generator with IB port counter validation."""

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Dict, Iterable


_COUNTER_SCALE_BYTES = {
    "port_xmit_data": 4,
    "port_rcv_data": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EP-like all-to-all traffic generator")
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--peer-ip")
    parser.add_argument("--port", type=int, default=18515)
    parser.add_argument("--target-gbps", type=float, default=0.0)
    parser.add_argument("--message-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--burst-ms", type=float, default=8.0)
    parser.add_argument("--gap-ms", type=float, default=2.0)
    parser.add_argument("--mlx-device", default="mlx5_0")
    parser.add_argument("--ib-port", type=int, default=1)
    parser.add_argument("--counters", default="port_xmit_data,port_rcv_data")
    parser.add_argument("--validate-counters", action="store_true")
    return parser.parse_args()


def counter_paths(mlx_device: str, ib_port: int, counters: Iterable[str]) -> Dict[str, Path]:
    base = Path("/sys/class/infiniband") / mlx_device / "ports" / str(ib_port) / "counters"
    return {name: base / name for name in counters}


def read_counters(mlx_device: str, ib_port: int, counters: Iterable[str]) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for name, path in counter_paths(mlx_device, ib_port, counters).items():
        raw = int(path.read_text().strip())
        values[name] = raw * _COUNTER_SCALE_BYTES.get(name, 1)
    return values


def send_json(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_json(sock: socket.socket) -> dict:
    header = recv_exact(sock, 4)
    size = struct.unpack("!I", header)[0]
    return json.loads(recv_exact(sock, size).decode("utf-8"))


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def traffic_loop(
    sock: socket.socket,
    stop_event: threading.Event,
    message: memoryview,
    burst_s: float,
    gap_s: float,
    target_gbps: float,
) -> None:
    bytes_per_second = target_gbps * 1e9 / 8.0
    while not stop_event.is_set():
        burst_start = time.monotonic()
        deadline = burst_start + burst_s
        sent = 0
        while time.monotonic() < deadline and not stop_event.is_set():
            sock.sendall(message)
            sent += len(message)
            if bytes_per_second > 0:
                expected_elapsed = sent / bytes_per_second
                actual_elapsed = time.monotonic() - burst_start
                if expected_elapsed > actual_elapsed:
                    stop_event.wait(expected_elapsed - actual_elapsed)
        if gap_s > 0:
            stop_event.wait(gap_s)


def recv_loop(sock: socket.socket, stop_event: threading.Event, message_bytes: int) -> None:
    while not stop_event.is_set():
        try:
            recv_exact(sock, message_bytes)
        except EOFError:
            stop_event.set()
            return


def run_connection(sock: socket.socket, args: argparse.Namespace, counters: list[str], target_gbps: float) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    if args.role == "client":
        send_json(sock, {
            "message_bytes": args.message_bytes,
            "burst_ms": args.burst_ms,
            "gap_ms": args.gap_ms,
            "target_gbps": target_gbps,
        })
    else:
        config = recv_json(sock)
        args.message_bytes = int(config["message_bytes"])
        args.burst_ms = float(config["burst_ms"])
        args.gap_ms = float(config["gap_ms"])
        target_gbps = float(config["target_gbps"])

    before = read_counters(args.mlx_device, args.ib_port, counters)
    stop_event = threading.Event()
    payload = memoryview(bytearray(args.message_bytes))
    burst_s = args.burst_ms / 1000.0
    gap_s = args.gap_ms / 1000.0

    sender = threading.Thread(
        target=traffic_loop,
        args=(sock, stop_event, payload, burst_s, gap_s, target_gbps),
        daemon=True,
    )
    receiver = threading.Thread(target=recv_loop, args=(sock, stop_event, args.message_bytes), daemon=True)
    sender.start()
    receiver.start()

    try:
        while not stop_event.wait(0.5):
            after = read_counters(args.mlx_device, args.ib_port, counters)
            deltas = {name: after[name] - before[name] for name in counters}
            print(f"[ep_alltoall] counter_deltas={deltas}", flush=True)
            if args.validate_counters and not any(value > 0 for value in deltas.values()):
                raise RuntimeError(f"IB counters did not increase: {deltas}")
    finally:
        stop_event.set()
        sock.close()


def server(args: argparse.Namespace, counters: list[str]) -> None:
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((args.bind_ip, args.port))
    listen_sock.listen(args.streams)
    print(f"[ep_alltoall] listening on {args.bind_ip}:{args.port} streams={args.streams}", flush=True)

    threads = []
    for _ in range(args.streams):
        conn, addr = listen_sock.accept()
        print(f"[ep_alltoall] accepted {addr}", flush=True)
        thread = threading.Thread(target=run_connection, args=(conn, args, counters, 0.0), daemon=False)
        thread.start()
        threads.append(thread)

    listen_sock.close()
    for thread in threads:
        thread.join()


def client(args: argparse.Namespace, counters: list[str]) -> None:
    if not args.peer_ip:
        raise ValueError("--peer-ip is required for client role")

    target_per_stream = args.target_gbps / max(args.streams, 1)
    threads = []
    for _ in range(args.streams):
        sock = socket.create_connection((args.peer_ip, args.port), timeout=10)
        print(f"[ep_alltoall] connected to {args.peer_ip}:{args.port}", flush=True)
        thread = threading.Thread(target=run_connection, args=(sock, args, counters, target_per_stream), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def main() -> None:
    args = parse_args()
    counters = [name.strip() for name in args.counters.split(",") if name.strip()]
    if args.role == "server":
        server(args, counters)
    else:
        client(args, counters)


if __name__ == "__main__":
    main()
