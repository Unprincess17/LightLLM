#!/usr/bin/env python3
"""
Python CFFI wrapper for librdmaqpgen.so — RDMA RC-QP pressure generator.

Provides RDMATrafficGenerator with the same start/stop/is_running interface
as EPTrafficGenerator, so it plugs in as mode="verbs_qp".
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SO_PATH = str(_HERE / "librdmaqpgen.so")

# Build the .so if missing
if not os.path.exists(_SO_PATH):
    subprocess.run(["make", "-C", str(_HERE)], check=True)

from cffi import FFI  # noqa: E402

ffi = FFI()
ffi.cdef("""
typedef struct rdmaqp_ctx rdmaqp_ctx;

typedef struct {
    int      qpn_base;
    int      lid;
    char     gid[40];
    uint64_t mr_addr;
    uint32_t mr_rkey;
} rdmaqp_peer_info;

rdmaqp_ctx* rdmaqp_create(
    const char *mlx_device,
    int         ib_port,
    int         num_qps,
    int         qp_depth,
    int         msg_bytes,
    char        errbuf[256]);

int rdmaqp_get_local_info(rdmaqp_ctx *ctx, rdmaqp_peer_info *info);
int rdmaqp_connect(rdmaqp_ctx *ctx, const rdmaqp_peer_info *remote,
                   char errbuf[256]);
int rdmaqp_start_burst_loop(rdmaqp_ctx *ctx, long burst_us, long gap_us,
                            int target_gbps);
int rdmaqp_stop(rdmaqp_ctx *ctx);
uint64_t rdmaqp_bytes_sent(rdmaqp_ctx *ctx);
void rdmaqp_destroy(rdmaqp_ctx *ctx);
""")

_lib = ffi.dlopen(_SO_PATH)

# Link capacity constants
LINK_CAPACITY_GBPS = 200
DEFAULT_QP_DEPTH = 128
DEFAULT_MSG_BYTES = 65536   # 64 KB per WR
DEFAULT_NUM_QPS = 16
DEFAULT_BURST_US = 8000     # 8 ms
DEFAULT_GAP_US = 2000       # 2 ms


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed during handshake")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_json(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_json(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    size = struct.unpack("!I", header)[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


class RDMATrafficGenerator:
    """RDMA RC-QP pressure generator wrapping librdmaqpgen.so."""

    def __init__(
        self,
        local_ip: str,
        remote_ip: str,
        mlx_device: str = "mlx5_0",
        ib_port: int = 1,
        num_qps: int = DEFAULT_NUM_QPS,
        qp_depth: int = DEFAULT_QP_DEPTH,
        msg_bytes: int = DEFAULT_MSG_BYTES,
        control_port: int = 18515,
        remote_ssh_host: Optional[str] = None,
        burst_us: int = DEFAULT_BURST_US,
        gap_us: int = DEFAULT_GAP_US,
    ) -> None:
        self._local_ip = local_ip
        self._remote_ip = remote_ip
        self._mlx_device = mlx_device
        self._ib_port = ib_port
        self._num_qps = num_qps
        self._qp_depth = qp_depth
        self._msg_bytes = msg_bytes
        self._control_port = control_port
        self._remote_ssh_host = remote_ssh_host
        self._burst_us = burst_us
        self._gap_us = gap_us

        self._ctx = ffi.NULL
        self._running = False
        self._bw_pct = 0
        self._remote_server_started = False

    # ------------------------------------------------------------------
    # Public API (compatible with EPTrafficGenerator)
    # ------------------------------------------------------------------

    def start(self, bw_pct: int) -> None:
        if self._running:
            self.stop()

        self._bw_pct = bw_pct

        if bw_pct == 0:
            print(
                f"[RDMATrafficGenerator] disabled for 0% BW baseline "
                f"(local={self._local_ip}, remote={self._remote_ip})",
                flush=True,
            )
            return

        target_gbps = int(LINK_CAPACITY_GBPS * bw_pct / 100.0)

        # Create local context
        errbuf = ffi.new("char[256]")
        self._ctx = _lib.rdmaqp_create(
            self._mlx_device.encode(),
            self._ib_port,
            self._num_qps,
            self._qp_depth,
            self._msg_bytes,
            errbuf,
        )
        if self._ctx == ffi.NULL:
            msg = ffi.string(errbuf).decode()
            self._ctx = ffi.NULL
            raise RuntimeError(f"rdmaqp_create failed: {msg}")

        # Start remote server and handshake
        self._start_remote_server()
        remote_dict = self._handshake()

        # Connect QPs
        remote_info = ffi.new("rdmaqp_peer_info *")
        remote_info.qpn_base = remote_dict["qpn_base"]
        remote_info.lid = remote_dict["lid"]
        gid_bytes = remote_dict["gid"].encode()
        ffi.memmove(remote_info.gid, gid_bytes, min(len(gid_bytes), 39))
        remote_info.gid[39] = 0
        remote_info.mr_addr = remote_dict["mr_addr"]
        remote_info.mr_rkey = remote_dict["mr_rkey"]

        ret = _lib.rdmaqp_connect(self._ctx, remote_info, errbuf)
        if ret != 0:
            msg = ffi.string(errbuf).decode()
            raise RuntimeError(f"rdmaqp_connect failed: {msg}")

        # Start burst loop
        ret = _lib.rdmaqp_start_burst_loop(
            self._ctx, self._burst_us, self._gap_us, target_gbps,
        )
        if ret != 0:
            raise RuntimeError("rdmaqp_start_burst_loop failed")

        self._running = True
        print(
            f"[RDMATrafficGenerator] started ({self._num_qps} QPs, {bw_pct}% BW, "
            f"target={target_gbps}Gbps, local={self._local_ip}, "
            f"remote={self._remote_ip})",
            flush=True,
        )

    def stop(self) -> None:
        if self._ctx != ffi.NULL:
            if self._running:
                _lib.rdmaqp_stop(self._ctx)
                self._running = False
            _lib.rdmaqp_destroy(self._ctx)
            self._ctx = ffi.NULL
        self._stop_remote_server()
        print("[RDMATrafficGenerator] stopped", flush=True)

    def is_running(self) -> bool:
        return self._running

    @property
    def bw_pct(self) -> int:
        return self._bw_pct

    # ------------------------------------------------------------------
    # Handshake internals
    # ------------------------------------------------------------------

    def _handshake(self) -> dict:
        """TCP handshake: exchange QP/GID/MR info, return remote info dict."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Try connecting with retries (server may still be starting)
        deadline = time.monotonic() + 10.0
        last_err = None
        while time.monotonic() < deadline:
            try:
                sock.connect((self._remote_ip, self._control_port))
                break
            except (ConnectionRefusedError, OSError) as exc:
                last_err = exc
                time.sleep(0.5)
        else:
            raise RuntimeError(
                f"Handshake: could not connect to {self._remote_ip}:"
                f"{self._control_port}: {last_err}"
            )

        try:
            # Send local info
            local = ffi.new("rdmaqp_peer_info *")
            _lib.rdmaqp_get_local_info(self._ctx, local)
            local_dict = {
                "qpn_base": local.qpn_base,
                "lid": local.lid,
                "gid": ffi.string(local.gid).decode(),
                "mr_addr": local.mr_addr,
                "mr_rkey": local.mr_rkey,
            }
            _send_json(sock, local_dict)

            # Receive remote info
            remote_dict = _recv_json(sock)
            return remote_dict
        finally:
            sock.close()

    def _start_remote_server(self) -> None:
        if not self._remote_ssh_host:
            return
        # Kill any stale server first
        subprocess.run(
            ["ssh", self._remote_ssh_host,
             "pkill -x ib_write_bw 2>/dev/null || true; "
             "pkill -f 'rdma_qp_pressure.*_run_server' 2>/dev/null || true"],
            timeout=5, capture_output=True,
        )
        # Build server command
        server_script = (
            f"import sys; sys.path.insert(0, '{_HERE.parent}'); "
            f"from rdma_qp_generator.rdma_qp_pressure import _run_server; "
            f"_run_server('{self._remote_ip}', {self._control_port}, "
            f"'{self._mlx_device}', {self._ib_port}, {self._num_qps}, "
            f"{self._qp_depth}, {self._msg_bytes})"
        )
        self._remote_proc = subprocess.Popen(
            ["ssh", self._remote_ssh_host,
             "nohup python3 -c '" + server_script +
             "' > /tmp/rdma_qp_server.log 2>&1 < /dev/null &"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._remote_server_started = True
        time.sleep(2)  # give the remote server time to bind

    def _stop_remote_server(self) -> None:
        if not self._remote_server_started or not self._remote_ssh_host:
            return
        subprocess.run(
            ["ssh", self._remote_ssh_host,
             "pkill -f 'rdma_qp_pressure.*_run_server' 2>/dev/null || true"],
            timeout=5, capture_output=True,
        )
        self._remote_server_started = False


def _run_server(
    bind_ip: str,
    port: int,
    mlx_device: str,
    ib_port: int,
    num_qps: int,
    qp_depth: int,
    msg_bytes: int,
) -> None:
    """Run the RDMA QP server side (called via SSH from client)."""
    print(f"[rdma_qp_server] starting on {bind_ip}:{port}", flush=True)

    errbuf = ffi.new("char[256]")
    ctx = _lib.rdmaqp_create(
        mlx_device.encode(), ib_port, num_qps, qp_depth, msg_bytes, errbuf,
    )
    if ctx == ffi.NULL:
        msg = ffi.string(errbuf).decode()
        raise RuntimeError(f"server rdmaqp_create failed: {msg}")

    try:
        # Accept one TCP connection
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind((bind_ip, port))
        listen_sock.listen(1)
        print(f"[rdma_qp_server] listening on {bind_ip}:{port}", flush=True)

        conn, addr = listen_sock.accept()
        print(f"[rdma_qp_server] accepted from {addr}", flush=True)
        listen_sock.close()

        # Receive client info
        client_dict = _recv_json(conn)

        # Send local info
        local = ffi.new("rdmaqp_peer_info *")
        _lib.rdmaqp_get_local_info(ctx, local)
        local_dict = {
            "qpn_base": local.qpn_base,
            "lid": local.lid,
            "gid": ffi.string(local.gid).decode(),
            "mr_addr": local.mr_addr,
            "mr_rkey": local.mr_rkey,
        }
        _send_json(conn, local_dict)
        conn.close()

        # Connect QPs to client
        remote = ffi.new("rdmaqp_peer_info *")
        remote.qpn_base = client_dict["qpn_base"]
        remote.lid = client_dict["lid"]
        gid_bytes = client_dict["gid"].encode()
        ffi.memmove(remote.gid, gid_bytes, min(len(gid_bytes), 39))
        remote.gid[39] = 0
        remote.mr_addr = client_dict["mr_addr"]
        remote.mr_rkey = client_dict["mr_rkey"]

        ret = _lib.rdmaqp_connect(ctx, remote, errbuf)
        if ret != 0:
            msg = ffi.string(errbuf).decode()
            raise RuntimeError(f"server rdmaqp_connect failed: {msg}")

        # Start burst loop (full speed — rate limiting is client-side)
        ret = _lib.rdmaqp_start_burst_loop(ctx, 8000, 2000, 0)
        if ret != 0:
            raise RuntimeError("server rdmaqp_start_burst_loop failed")

        # Wait until killed
        print("[rdma_qp_server] burst loop running, waiting for SIGTERM",
              flush=True)
        while True:
            time.sleep(5)

    except KeyboardInterrupt:
        pass
    finally:
        _lib.rdmaqp_stop(ctx)
        _lib.rdmaqp_destroy(ctx)
        print("[rdma_qp_server] stopped", flush=True)
