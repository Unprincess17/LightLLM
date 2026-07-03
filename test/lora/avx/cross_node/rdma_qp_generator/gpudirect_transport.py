# test/lora/avx/cross_node/rdma_qp_generator/gpudirect_transport.py
"""Python CFFI wrapper for libgdrqpgen.so.

Provides GPUDirectTransport — point-to-point RDMA WRITE/READ between
GPU memory regions registered via nvidia_peermem.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import json
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SO_PATH = str(_HERE / "libgdrqpgen.so")

if not os.path.exists(_SO_PATH):
    subprocess.run(["make", "-C", str(_HERE), "libgdrqpgen.so"], check=True)

from cffi import FFI  # noqa: E402

ffi = FFI()
ffi.cdef("""
typedef struct gdrqp_ctx gdrqp_ctx;

typedef struct {
    int      qpn;
    int      lid;
    char     gid[40];
    uint64_t mr_addr;
    uint32_t mr_rkey;
    uint64_t mr_size;
} gdrqp_peer_info;

gdrqp_ctx* gdrqp_create(const char *mlx_device, int ib_port,
                        int qp_depth, size_t gpu_buffer_bytes,
                        char errbuf[256]);
int gdrqp_get_local_info(gdrqp_ctx *ctx, gdrqp_peer_info *info);
int gdrqp_connect(gdrqp_ctx *ctx, const gdrqp_peer_info *remote,
                  char errbuf[256]);
int gdrqp_write(gdrqp_ctx *ctx, size_t local_offset, size_t remote_offset,
                size_t nbytes, char errbuf[256]);
int gdrqp_read(gdrqp_ctx *ctx, size_t local_offset, size_t remote_offset,
               size_t nbytes, char errbuf[256]);
void* gdrqp_gpu_ptr(gdrqp_ctx *ctx);
size_t gdrqp_gpu_size(gdrqp_ctx *ctx);
void gdrqp_destroy(gdrqp_ctx *ctx);
""")

_lib = ffi.dlopen(_SO_PATH)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n > 0:
        c = sock.recv(n)
        if not c:
            raise EOFError("socket closed during handshake")
        chunks.append(c)
        n -= len(c)
    return b"".join(chunks)


def _send_json(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_json(sock: socket.socket) -> dict:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


class GPUDirectTransport:
    """Point-to-point GPUDirect RDMA transport between two nodes.

    Lifecycle:
      1. __init__   — store config, allocate nothing yet
      2. start_server / start_client — create ctx, register GPU MR
      3. connect    — TCP handshake then transition QP to RTS
      4. write / read — point-to-point GPU-to-GPU RDMA
      5. close      — destroy ctx, free GPU buffer
    """

    def __init__(
        self,
        local_ip: str,
        remote_ip: str,
        mlx_device: str = "mlx5_0",
        ib_port: int = 1,
        qp_depth: int = 128,
        gpu_buffer_bytes: int = 1 << 24,  # 16 MB default
        control_port: int = 18516,
    ) -> None:
        self._local_ip = local_ip
        self._remote_ip = remote_ip
        self._mlx_device = mlx_device
        self._ib_port = ib_port
        self._qp_depth = qp_depth
        self._gpu_buffer_bytes = gpu_buffer_bytes
        self._control_port = control_port
        self._ctx = ffi.NULL
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _create_ctx(self) -> None:
        errbuf = ffi.new("char[256]")
        self._ctx = _lib.gdrqp_create(
            self._mlx_device.encode(),
            self._ib_port,
            self._qp_depth,
            self._gpu_buffer_bytes,
            errbuf,
        )
        if self._ctx == ffi.NULL:
            msg = ffi.string(errbuf).decode()
            raise RuntimeError(f"gdrqp_create failed: {msg}")

    def _local_info_dict(self) -> dict:
        info = ffi.new("gdrqp_peer_info *")
        if _lib.gdrqp_get_local_info(self._ctx, info) != 0:
            raise RuntimeError("gdrqp_get_local_info failed")
        return {
            "qpn": info.qpn,
            "lid": info.lid,
            "gid": ffi.string(info.gid).decode(),
            "mr_addr": info.mr_addr,
            "mr_rkey": info.mr_rkey,
            "mr_size": info.mr_size,
        }

    def _do_connect(self, remote_dict: dict) -> None:
        remote = ffi.new("gdrqp_peer_info *")
        remote.qpn = remote_dict["qpn"]
        remote.lid = remote_dict["lid"]
        gid_bytes = remote_dict["gid"].encode()
        ffi.memmove(remote.gid, gid_bytes, min(len(gid_bytes), 39))
        remote.gid[39] = b"\0"
        remote.mr_addr = remote_dict["mr_addr"]
        remote.mr_rkey = remote_dict["mr_rkey"]
        remote.mr_size = remote_dict["mr_size"]
        errbuf = ffi.new("char[256]")
        if _lib.gdrqp_connect(self._ctx, remote, errbuf) != 0:
            raise RuntimeError(
                f"gdrqp_connect failed: {ffi.string(errbuf).decode()}")

    def start_client(self) -> None:
        """Create local ctx, TCP-connect to server, exchange peer info,
        then transition QP to RTS."""
        self._create_ctx()
        try:
            deadline = time.monotonic() + 10.0
            last_err = None
            while time.monotonic() < deadline:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.connect((self._remote_ip, self._control_port))
                    break
                except (ConnectionRefusedError, OSError) as exc:
                    last_err = exc
                    sock.close()
                    time.sleep(0.01)
            else:
                raise RuntimeError(f"connect to {self._remote_ip} failed: "
                                   f"{last_err}")
            try:
                _send_json(sock, self._local_info_dict())
                remote_dict = _recv_json(sock)
            finally:
                sock.close()
            self._do_connect(remote_dict)
            self._connected = True
        except BaseException:
            self.close()
            raise

    def start_server(self) -> None:
        """Create local ctx, listen for one client, exchange peer info,
        then transition QP to RTS.

        Binds the listening socket *before* creating the QP/MR so the peer's
        start_client() never hits ConnectionRefused while we're still in
        _create_ctx().  Previously _create_ctx() ran first (10-100 ms); if
        the peer finished its own _create_ctx() faster, it retried with a
        500 ms sleep, producing ~520 ms outliers in the sweep.
        """
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listen.bind((self._local_ip, self._control_port))
            listen.listen(1)
        except BaseException:
            listen.close()
            raise
        try:
            self._create_ctx()
            conn, _ = listen.accept()
        except BaseException:
            self.close()
            raise
        finally:
            listen.close()
        try:
            remote_dict = _recv_json(conn)
            _send_json(conn, self._local_info_dict())
        finally:
            conn.close()
        self._do_connect(remote_dict)
        self._connected = True

    def gpu_ptr(self) -> int:
        """Raw GPU pointer (as int) — caller can wrap into torch tensor
        via cuda.IPC or cudaMemcpy."""
        if self._ctx == ffi.NULL:
            raise RuntimeError("transport not initialized")
        return int(ffi.cast("uintptr_t", _lib.gdrqp_gpu_ptr(self._ctx)))

    def write_to_remote(self, local_offset: int, remote_offset: int,
                        nbytes: int) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        errbuf = ffi.new("char[256]")
        ret = _lib.gdrqp_write(self._ctx, local_offset, remote_offset,
                                nbytes, errbuf)
        if ret != 0:
            raise RuntimeError(
                f"gdrqp_write failed: {ffi.string(errbuf).decode()}")

    def read_from_remote(self, local_offset: int, remote_offset: int,
                         nbytes: int) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        errbuf = ffi.new("char[256]")
        ret = _lib.gdrqp_read(self._ctx, local_offset, remote_offset,
                               nbytes, errbuf)
        if ret != 0:
            raise RuntimeError(
                f"gdrqp_read failed: {ffi.string(errbuf).decode()}")

    def close(self) -> None:
        if self._ctx != ffi.NULL:
            _lib.gdrqp_destroy(self._ctx)
            self._ctx = ffi.NULL
        self._connected = False
