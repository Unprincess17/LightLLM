#!/usr/bin/env python3
"""QP Pool for GPUDirectTransport — pre-connected and lazy modes.

Pre-connected mode: all N QP pairs are created, handshaked, and transitioned
to RTS state at setup time. Borrow/return is just semaphore acquire/release
with zero connection latency on the hot path.

Lazy mode: QPs and MRs are pre-created (amortizing MR registration), but the
TCP handshake + QP INIT→RTR→RTS transition happens on each borrow.
"""

from __future__ import annotations

import socket
import struct
import json
import threading
import time
from typing import Optional, Any

try:
    from rdma_qp_generator.gpudirect_transport import GPUDirectTransport
except ImportError:
    GPUDirectTransport = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n > 0:
        c = sock.recv(n)
        if not c:
            raise EOFError("socket closed")
        chunks.append(c)
        n -= len(c)
    return b"".join(chunks)


def _send_json(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_json(sock: socket.socket) -> dict:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    return json.loads(_recv_exact(sock, size).decode("utf-8"))


# ---------------------------------------------------------------------------
# QPPoolClient — runs on the inference node (UM253).
# Each transport acts as a GPUDirect "server" (listens, accepts).
# ---------------------------------------------------------------------------

class QPPoolClient:
    """Pre-created pool of GPUDirect transports on the client (inference) side.

    Each transport is a GPUDirect *server* — it binds a TCP socket, waits for
    the server node to connect, then exchanges peer info.
    """

    def __init__(
        self,
        size: int,
        local_ip: str,
        remote_ip: str,
        base_control_port: int,
        gpu_buffer_bytes: int,
        mode: str = "preconnected",
        mlx_device: str = "mlx5_0",
        ib_port: int = 1,
        qp_depth: int = 128,
        active_cap: int = None,
    ):
        self.size = size
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.base_control_port = base_control_port
        self.gpu_buffer_bytes = gpu_buffer_bytes
        self.mode = mode
        self.mlx_device = mlx_device
        self.ib_port = ib_port
        self.qp_depth = qp_depth

        self._transports: list[GPUDirectTransport] = []
        self._server_socks: list[socket.socket] = []  # listening sockets
        self._accept_threads: dict[int, threading.Thread] = {}  # lazy mode accept threads by port
        self._sem = threading.Semaphore(size)
        self._lock = threading.Lock()
        self._connected = False

        # Active concurrency cap (independent of physical pool size).
        # None means active_cap == size (backward compatible).
        self.active_cap = active_cap if active_cap is not None else size
        self._active_sem = threading.Semaphore(self.active_cap)
        self._qp_wait_us: list[float] = []

    # ------------------------------------------------------------------
    # setup / teardown
    # ------------------------------------------------------------------

    def setup(self, server_sock: socket.socket) -> None:
        """Create N transports, start each as GPUDirect server.

        Sends ``setup_pool`` over *server_sock* (the main TCP control
        connection to the remote server process) with the local QP info
        for every transport so the server can connect back.
        """
        local_infos = []
        control_ports = []

        for i in range(self.size):
            port = self.base_control_port + i
            transport = GPUDirectTransport(
                local_ip=self.local_ip,
                remote_ip=self.remote_ip,
                gpu_buffer_bytes=self.gpu_buffer_bytes,
                control_port=port,
                mlx_device=self.mlx_device,
                ib_port=self.ib_port,
                qp_depth=self.qp_depth,
            )
            # Create QP + MR, get local info
            transport._create_ctx()
            info = transport._local_info_dict()
            local_infos.append(info)
            control_ports.append(port)
            self._transports.append(transport)

        # Send pool setup to server
        _send_json(server_sock, {
            "type": "setup_pool",
            "pool_size": self.size,
            "control_ports": control_ports,
            "local_infos": local_infos,
            "mode": self.mode,
            "gpu_buffer_bytes": self.gpu_buffer_bytes,
        })

        if self.mode == "preconnected":
            self._accept_all()
        elif self.mode == "lazy":
            # Store local infos for later per-request handshake
            pass

        self._connected = True

    def _accept_all(self) -> None:
        """Accept connections from server for all N transports.

        Each transport's control port must receive one connection from the
        server side.  We accept them in order (server connects sequentially).
        """
        # Start listening on all control ports
        listeners = []
        for i, transport in enumerate(self._transports):
            port = self.base_control_port + i
            listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen.bind((self.local_ip, port))
            listen.listen(1)
            listeners.append(listen)
            self._server_socks.append(listen)

        # Accept one connection per transport (server connects in order)
        for i, (transport, listen) in enumerate(zip(self._transports, listeners)):
            conn, _ = listen.accept()
            try:
                remote_dict = _recv_json(conn)
                _send_json(conn, transport._local_info_dict())
            finally:
                conn.close()
            transport._do_connect(remote_dict)
            transport._connected = True

    def teardown(self) -> None:
        """Close all transports and listening sockets."""
        for transport in self._transports:
            try:
                transport.close()
            except Exception:
                pass
        for sock in self._server_socks:
            try:
                sock.close()
            except Exception:
                pass
        self._transports.clear()
        self._server_socks.clear()
        self._connected = False

    # ------------------------------------------------------------------
    # hot path
    # ------------------------------------------------------------------

    def borrow(self) -> tuple[int, GPUDirectTransport, Optional[threading.Event]]:
        """Acquire a free transport.  Returns (pool_id, transport, ready_event).

        In lazy mode, *ready_event* is set when the server has connected and
        the QP is in RTS state.  The caller MUST wait on this event before
        using the transport for RDMA operations.
        In preconnected mode, *ready_event* is None (transport is already RTS).
        """
        t_wait_start = time.perf_counter()
        self._active_sem.acquire()        # active concurrency gate
        self._sem.acquire()               # physical pool gate
        t_wait_end = time.perf_counter()
        self._qp_wait_us.append((t_wait_end - t_wait_start) * 1e6)
        with self._lock:
            pool_id = len(self._transports) - 1
            transport = self._transports.pop()

        ready_event: Optional[threading.Event] = None

        if self.mode == "lazy":
            ready_event = threading.Event()
            port = self.base_control_port + pool_id

            # Bind and listen synchronously in the main thread so the port
            # is guaranteed to be accepting connections before we return.
            # Only the blocking accept() + handshake is done in a thread.
            listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen.bind((self.local_ip, port))
            listen.listen(1)
            print(f"[qppool] Listening on {self.local_ip}:{port}", flush=True)

            def _accept_one():
                try:
                    conn, _ = listen.accept()
                    try:
                        remote_dict = _recv_json(conn)
                        _send_json(conn, transport._local_info_dict())
                    finally:
                        conn.close()
                    transport._do_connect(remote_dict)
                    transport._connected = True
                finally:
                    listen.close()
                ready_event.set()

            t = threading.Thread(target=_accept_one, daemon=True)
            t.start()
            self._accept_threads[port] = t

        return pool_id, transport, ready_event

    def return_transport(self, pool_id: int, transport: GPUDirectTransport) -> None:
        """Return a transport to the pool."""
        if self.mode == "lazy":
            # Wait for the accept thread to finish and close the listen socket,
            # otherwise the next borrow on the same port may get EADDRINUSE.
            port = self.base_control_port + pool_id
            t = self._accept_threads.pop(port, None)
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
            # Disconnect: close QP, recreate in INIT
            transport.close()
            transport._create_ctx()

        with self._lock:
            self._transports.append(transport)
        self._sem.release()
        self._active_sem.release()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def gpu_ptr(self, pool_id: int) -> int:
        """Raw GPU pointer for transport at index."""
        return self._transports[pool_id].gpu_ptr()

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# QPPoolServer — runs on the weight node (UM251).
# Each transport acts as a GPUDirect "client" (connects to UM253).
# ---------------------------------------------------------------------------

class QPPoolServer:
    """Pre-created pool of GPUDirect transports on the server (weight) side.

    Each transport connects as a GPUDirect *client* to the corresponding
    client-side transport.
    """

    def __init__(
        self,
        local_ip: str,
        mlx_device: str = "mlx5_0",
        ib_port: int = 1,
        qp_depth: int = 128,
        active_cap: int = None,
    ):
        self.local_ip = local_ip
        self.mlx_device = mlx_device
        self.ib_port = ib_port
        self.qp_depth = qp_depth
        self._transports: list[GPUDirectTransport] = []
        self._sem = threading.Semaphore(1)  # updated in setup
        self._lock = threading.Lock()
        self._connected = False
        self._mode = "preconnected"

        # Active concurrency cap (independent of physical pool size).
        # None means active_cap == size (backward compatible, resolved in setup).
        self.active_cap = active_cap
        self._active_sem = threading.Semaphore(1)  # updated in setup
        self._qp_wait_us: list[float] = []

    def setup(self, params: dict[str, Any]) -> None:
        """Create N transports and connect to client.

        *params* is the ``setup_pool`` dict from the client containing
        ``pool_size``, ``control_ports``, ``local_infos``, ``mode``,
        ``gpu_buffer_bytes``.
        """
        size = params["pool_size"]
        control_ports = params["control_ports"]
        client_infos = params["local_infos"]
        self._mode = params.get("mode", "preconnected")
        gpu_buffer_bytes = params["gpu_buffer_bytes"]
        remote_ip = params.get("remote_ip", "")

        self._transports = []
        for i in range(size):
            transport = GPUDirectTransport(
                local_ip=self.local_ip,
                remote_ip=remote_ip,
                gpu_buffer_bytes=gpu_buffer_bytes,
                control_port=control_ports[i],
                mlx_device=self.mlx_device,
                ib_port=self.ib_port,
                qp_depth=self.qp_depth,
            )
            transport._create_ctx()
            self._transports.append(transport)

        if self._mode == "preconnected":
            self._connect_all(control_ports, client_infos, remote_ip)

        self._sem = threading.Semaphore(size)
        self.active_cap = self.active_cap if self.active_cap is not None else size
        self._active_sem = threading.Semaphore(self.active_cap)
        self._connected = True

    def _connect_all(self, control_ports, client_infos, remote_ip):
        """Connect to all client-side transports."""
        for i, transport in enumerate(self._transports):
            port = control_ports[i]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((remote_ip, port))
            try:
                _send_json(sock, transport._local_info_dict())
                client_dict = _recv_json(sock)
            finally:
                sock.close()
            transport._do_connect(client_dict)
            transport._connected = True

    def teardown(self) -> None:
        for transport in self._transports:
            try:
                transport.close()
            except Exception:
                pass
        self._transports.clear()
        self._connected = False

    # ------------------------------------------------------------------
    # hot path
    # ------------------------------------------------------------------

    def borrow(self, handshake_port: Optional[int] = None) -> tuple[int, GPUDirectTransport]:
        """Acquire a free transport.

        In lazy mode, *handshake_port* is the port the client is listening on
        for the QP handshake.  If not given, falls back to ``transport._control_port``
        (used by pre-connected mode which doesn't call borrow directly).
        """
        t_wait_start = time.perf_counter()
        self._active_sem.acquire()        # active concurrency gate
        self._sem.acquire()               # physical pool gate
        t_wait_end = time.perf_counter()
        self._qp_wait_us.append((t_wait_end - t_wait_start) * 1e6)
        with self._lock:
            pool_id = len(self._transports) - 1
            transport = self._transports.pop()

        if self._mode == "lazy":
            # Connect to the port the client is listening on.
            # If handshake_port is given (from client's s4a_pooled message),
            # use it directly; otherwise fall back to transport._control_port.
            port = handshake_port if handshake_port is not None else transport._control_port
            remote_ip = transport._remote_ip
            print(f"[qppool] Server connecting to {remote_ip}:{port}", flush=True)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((remote_ip, port))
            try:
                _send_json(sock, transport._local_info_dict())
                client_dict = _recv_json(sock)
            finally:
                sock.close()
            transport._do_connect(client_dict)
            transport._connected = True

        return pool_id, transport

    def return_transport(self, pool_id: int, transport: GPUDirectTransport) -> None:
        if self._mode == "lazy":
            transport.close()
            transport._create_ctx()
        with self._lock:
            self._transports.append(transport)
        self._sem.release()
        self._active_sem.release()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def size(self) -> int:
        return len(self._transports)
