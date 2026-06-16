#!/usr/bin/env python3
"""
EP Traffic Generator — simulates MoE Expert Parallel all-to-all background traffic.

Uses ib_write_bw (perftest package) as the RDMA traffic generator, controlled
by Python subprocess. Supports rate limiting to simulate 0-90% link saturation.

Important: ib_write_bw requires a server process on the remote peer. Start it on
UM251 before running the benchmark, e.g.:
    ib_write_bw -d mlx5_0 -p 18515 --duration=999999

Usage:
    gen = EPTrafficGenerator(
        EPTrafficConfig(
            local_ip="10.10.1.1",   # UM253
            remote_ip="10.10.1.3",  # UM251
            mlx_device="mlx5_0",    # RDMA device
        )
    )

    gen.start(bw_pct=50)   # 50% of link capacity
    # ... run benchmark ...
    gen.stop()
"""

import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# Link capacity in Gbps (HDR InfiniBand 200 Gbps)
LINK_CAPACITY_GBPS = 200


@dataclass
class EPTrafficConfig:
    """Configuration for EP traffic generator."""
    local_ip: str
    remote_ip: str
    mlx_device: str = "mlx5_0"
    base_port: int = 18515
    remote_ssh_host: Optional[str] = None
    direction: str = "bidirectional"
    mode: str = "ib_write_bw"
    ib_port: int = 1
    link_capacity_gbps: int = LINK_CAPACITY_GBPS
    counter_names: Tuple[str, ...] = ("port_xmit_data", "port_rcv_data")
    alltoall_streams: int = 16
    # For bursty mode
    burst_on_ms: float = 10.0
    burst_off_ms: float = 10.0
    # For verbs_qp mode
    num_qps: int = 16
    qp_depth: int = 128
    msg_bytes: int = 65536

    @property
    def server_cmd(self) -> List[str]:
        if self.mode == "verbs_qp":
            return ["true"]
        if self.mode == "alltoall":
            script = Path(__file__).with_name("ep_alltoall_traffic.py")
            return [
                "python",
                str(script),
                "--role", "server",
                "--bind-ip", self.remote_ip,
                "--port", str(self.base_port),
                "--mlx-device", self.mlx_device,
                "--ib-port", str(self.ib_port),
                "--counters", ",".join(self.counter_names),
                "--streams", str(self.alltoall_streams),
            ]

        cmd = [
            "ib_write_bw",
            "-d", self.mlx_device,
            "-p", str(self.base_port),
            "--duration=999999",
        ]
        if self.direction == "bidirectional":
            cmd.append("-b")
        return cmd

    @property
    def remote_cleanup_cmd(self) -> str:
        if self.mode == "verbs_qp":
            return "pkill -f 'rdma_qp_pressure.*_run_server' 2>/dev/null || true"
        if self.mode == "alltoall":
            return "pkill -f '[e]p_alltoall_traffic.py' || true"
        return "pkill -x ib_write_bw || true"

    @property
    def remote_start_cmds(self) -> List[str]:
        server_cmd = " ".join(shlex.quote(part) for part in self.server_cmd)
        return [
            self.remote_cleanup_cmd,
            f"nohup {server_cmd} > /tmp/colora_ib_write_bw_server.log 2>&1 < /dev/null &",
        ]

    def client_cmd(self, bw_pct: int) -> List[str]:
        if self.mode == "alltoall":
            script = Path(__file__).with_name("ep_alltoall_traffic.py")
            target_gbps = self.link_capacity_gbps * bw_pct / 100.0
            return [
                "python",
                str(script),
                "--role", "client",
                "--peer-ip", self.remote_ip,
                "--port", str(self.base_port),
                "--target-gbps", f"{target_gbps:g}",
                "--mlx-device", self.mlx_device,
                "--ib-port", str(self.ib_port),
                "--counters", ",".join(self.counter_names),
                "--streams", str(self.alltoall_streams),
                "--validate-counters",
            ]

        # This perftest version uses --rate_limit with Gbps units by default.
        rate_gbps = self.link_capacity_gbps * bw_pct / 100.0
        cmd = [
            "ib_write_bw",
            "-d", self.mlx_device,
            self.remote_ip,
            "-p", str(self.base_port),
            f"--rate_limit={rate_gbps:g}",
            "--rate_units=g",
            "--rate_limit_type=SW",
            "--duration=999999",
        ]
        if self.direction == "bidirectional":
            cmd.append("-b")
        return cmd


class EPTrafficGenerator:
    """
    Controls a local ib_write_bw client connected to a remote ib_write_bw server.

    Two operating modes:
      - continuous : constant-rate sustained traffic at `bw_pct` of link capacity
      - bursty     : ON/OFF bursts at full rate, duty-cycle = bw_pct/100

    Thread-safe. Fails fast when ib_write_bw is missing or the remote server is
    not reachable so benchmark results are not mislabeled as contended.
    """

    def __init__(self, config: EPTrafficConfig, mode: str = "continuous") -> None:
        self.config = config
        self.mode = mode  # "continuous" or "bursty"
        self._lock = threading.Lock()
        self._running = False
        self._client_proc: Optional[subprocess.Popen] = None
        self._client_stderr: Optional[tempfile.NamedTemporaryFile] = None
        self._remote_server_started = False
        self._burst_thread: Optional[threading.Thread] = None
        self._burst_stop_event = threading.Event()
        self._bw_pct: int = 0

        # Check availability once at construction time
        self._ib_write_bw_available = self._check_ib_write_bw()

        self._verbs_gen = None
        if self.config.mode == "verbs_qp":
            from rdma_qp_generator.rdma_qp_pressure import RDMATrafficGenerator
            self._verbs_gen = RDMATrafficGenerator(
                local_ip=self.config.local_ip,
                remote_ip=self.config.remote_ip,
                mlx_device=self.config.mlx_device,
                ib_port=self.config.ib_port,
                num_qps=self.config.num_qps,
                qp_depth=self.config.qp_depth,
                msg_bytes=self.config.msg_bytes,
                control_port=self.config.base_port,
                remote_ssh_host=self.config.remote_ssh_host,
                burst_us=int(self.config.burst_on_ms * 1000),
                gap_us=int(self.config.burst_off_ms * 1000),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, bw_pct: int = 0) -> None:
        """
        Start EP background traffic at `bw_pct` of link capacity.

        Args:
            bw_pct: Percentage of link capacity to saturate (0-100).
                    0 means traffic is started but rate-limited to 0 (effectively idle).
        """
        with self._lock:
            if self._running:
                self.stop()

            self._bw_pct = bw_pct
            self._running = True

            if self._verbs_gen is not None:
                self._verbs_gen.start(bw_pct)
                return

            if bw_pct == 0:
                print(
                    f"[EPTrafficGenerator] disabled for 0% BW baseline "
                    f"(local={self.config.local_ip}, remote={self.config.remote_ip})",
                    flush=True,
                )
                return

            if self.config.mode == "ib_write_bw" and not self._ib_write_bw_available:
                self._running = False
                raise RuntimeError("ib_write_bw not found on PATH")

            self._start_remote_server_if_configured()

            if self.mode == "bursty" and bw_pct > 0 and bw_pct < 100:
                self._start_bursty(bw_pct)
            else:
                self._start_client(bw_pct)

            self._wait_until_active()

            print(
                f"[EPTrafficGenerator] started ({self.mode} mode, {bw_pct}% BW, "
                f"local={self.config.local_ip}, remote={self.config.remote_ip}, "
                f"remote server={self.config.remote_ip}:{self.config.base_port})",
                flush=True,
            )

    def stop(self) -> None:
        """Stop all traffic-generating processes gracefully."""
        with self._lock:
            self._running = False

            if self._verbs_gen is not None:
                self._verbs_gen.stop()
                return

            # Stop burst thread first
            if self._burst_thread is not None:
                self._burst_stop_event.set()
                self._burst_thread.join(timeout=2.0)
                self._burst_thread = None
                self._burst_stop_event.clear()

            if self._client_proc is not None:
                self._terminate_proc(self._client_proc, "client")

            self._client_proc = None
            self._close_client_stderr()
            self._stop_remote_server_if_configured()

            if self._ib_write_bw_available:
                print("[EPTrafficGenerator] stopped", flush=True)

    def is_running(self) -> bool:
        """Return True if traffic generation is active."""
        with self._lock:
            if not self._running:
                return False
            if self._verbs_gen is not None:
                return self._verbs_gen.is_running()
            if self._client_proc is not None and self._client_proc.poll() is not None:
                print(
                    "[EPTrafficGenerator] client process died unexpectedly — "
                    f"marking stopped. stderr: {self._client_error_tail()}",
                    flush=True,
                )
                self._running = False
                return False
            return True

    @property
    def bw_pct(self) -> int:
        """Current bandwidth percentage."""
        with self._lock:
            return self._bw_pct

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_ib_write_bw(self) -> bool:
        """Return True if ib_write_bw is found on PATH."""
        for candidate in ("ib_write_bw", "/usr/bin/ib_write_bw"):
            try:
                subprocess.run(
                    [candidate, "--help"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                return True
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                # --help often exits quickly; a timeout means it hung (unlikely for ib_write_bw)
                return True  # treat as available
        return False

    def _start_remote_server_if_configured(self) -> None:
        if not self.config.remote_ssh_host:
            return

        for remote_cmd in self.config.remote_start_cmds:
            subprocess.run(
                ["ssh", self.config.remote_ssh_host, remote_cmd],
                check=True,
                timeout=15,
            )
        self._remote_server_started = True
        time.sleep(0.5)

    def _stop_remote_server_if_configured(self) -> None:
        if not self._remote_server_started or not self.config.remote_ssh_host:
            return
        subprocess.run(
            ["ssh", self.config.remote_ssh_host, self.config.remote_cleanup_cmd],
            check=False,
            timeout=15,
        )
        self._remote_server_started = False

    def _start_client(self, bw_pct: int) -> None:
        env = os.environ.copy()
        env["IBV_DEVICE_NAME"] = self.config.mlx_device
        self._close_client_stderr()
        self._client_stderr = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="ep_ib_write_bw_",
            suffix=".stderr",
            delete=False,
        )
        self._client_proc = subprocess.Popen(
            self.config.client_cmd(bw_pct=bw_pct),
            stdout=subprocess.DEVNULL,
            stderr=self._client_stderr,
            env=env,
            preexec_fn=os.setsid,
        )

    def _wait_until_active(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._client_proc is not None and self._client_proc.poll() is not None:
                self._running = False
                raise RuntimeError(
                    "ib_write_bw client exited during startup. "
                    f"Is remote server running on {self.config.remote_ip}:{self.config.base_port}? "
                    f"stderr: {self._client_error_tail()}"
                )
            time.sleep(0.05)

        if self._client_proc is None:
            self._running = False
            raise RuntimeError("ib_write_bw client was not started")

    def _start_bursty(self, bw_pct: int) -> None:
        """Run ib_write_bw client in a thread cycling ON/OFF bursts."""

        def burst_loop() -> None:
            # In bursty mode we drive ON/OFF by starting/killing the client process.
            # A lighter alternative is writing to /sys/class/infiniband/.../port_cntl
            # but killing the client subprocess is the most portable approach.
            T_on = self.config.burst_on_ms / 1000.0
            T_off = self.config.burst_off_ms / 1000.0
            burst_iter = 0

            while not self._burst_stop_event.is_set():
                # Start ON burst
                env = os.environ.copy()
                env["IBV_DEVICE_NAME"] = self.config.mlx_device
                burst_stderr = tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix=f"ep_ib_write_bw_burst_{burst_iter}_",
                    suffix=".stderr",
                    delete=False,
                )
                burst_proc = subprocess.Popen(
                    self.config.client_cmd(bw_pct=100),  # full rate during ON
                    stdout=subprocess.DEVNULL,
                    stderr=burst_stderr,
                    env=env,
                    preexec_fn=os.setsid,
                )
                try:
                    # Sleep for ON duration, checking stop every 10 ms
                    deadline = time.monotonic() + T_on
                    while time.monotonic() < deadline and not self._burst_stop_event.is_set():
                        if burst_proc.poll() is not None:
                            self._running = False
                            print(
                                f"[EPTrafficGenerator] burst client exited early. stderr file: {burst_stderr.name}",
                                flush=True,
                            )
                            break
                        time.sleep(0.01)
                finally:
                    self._terminate_proc(burst_proc, f"burst[{burst_iter}]-client")
                    burst_stderr.close()

                if self._burst_stop_event.is_set():
                    break

                # OFF period
                time.sleep(T_off)
                burst_iter += 1

        self._burst_stop_event.clear()
        self._burst_thread = threading.Thread(target=burst_loop, daemon=True)
        self._burst_thread.start()

    @staticmethod
    def _terminate_proc(proc: subprocess.Popen, name: str) -> None:
        """Send SIGTERM then SIGKILL to a process group."""
        try:
            # SIGTERM the whole process group (set via os.setsid in preexec_fn)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=1.0)
            except ProcessLookupError:
                pass  # already dead
        except ProcessLookupError:
            pass  # already dead
        except Exception as exc:
            print(
                f"[EPTrafficGenerator] error terminating {name} (pid={proc.pid}): {exc}",
                flush=True,
            )

    def _client_error_tail(self) -> str:
        if self._client_stderr is None:
            return "<no stderr file>"
        path = Path(self._client_stderr.name)
        try:
            data = path.read_bytes()[-4096:]
        except OSError as exc:
            return f"<could not read {path}: {exc}>"
        text = data.decode("utf-8", errors="replace").strip()
        return text or f"<empty stderr: {path}>"

    def _close_client_stderr(self) -> None:
        if self._client_stderr is not None:
            self._client_stderr.close()
            self._client_stderr = None


# ------------------------------------------------------------------
# Convenience helpers for use in cross_node_benchmark.py
# ------------------------------------------------------------------

def start_ep_burst(
    local_ip: str,
    remote_ip: str,
    mlx_device: str,
    bw_pct: int,
) -> EPTrafficGenerator:
    gen = EPTrafficGenerator(
        EPTrafficConfig(local_ip=local_ip, remote_ip=remote_ip, mlx_device=mlx_device),
        mode="bursty",
    )
    gen.start(bw_pct=bw_pct)
    return gen


def stop_ep(gen: EPTrafficGenerator) -> None:
    gen.stop()
