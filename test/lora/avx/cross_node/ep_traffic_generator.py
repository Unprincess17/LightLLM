#!/usr/bin/env python3
"""
EP Traffic Generator — simulates MoE Expert Parallel all-to-all background traffic.

Uses ib_write_bw (perftest package) as the RDMA traffic generator, controlled
by Python subprocess. Supports rate limiting to simulate 0-90% link saturation.

Usage:
    gen = EPTrafficGenerator(
        local_ip="10.10.1.1",   # UM253
        remote_ip="10.10.1.3",  # UM251
        mlx_device="mlx5_0",    # RDMA device
    )

    gen.start(bw_pct=50)   # 50% of link capacity
    # ... run benchmark ...
    gen.stop()
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


# Link capacity in Gbps (HDR InfiniBand 200 Gbps)
LINK_CAPACITY_GBPS = 200


@dataclass
class EPTrafficConfig:
    """Configuration for EP traffic generator."""
    local_ip: str
    remote_ip: str
    mlx_device: str = "mlx5_0"
    base_port: int = 18515
    # For bursty mode
    burst_on_ms: float = 10.0
    burst_off_ms: float = 10.0

    @property
    def server_cmd(self) -> List[str]:
        return [
            "ib_write_bw",
            "-d", self.mlx_device,
            "-p", str(self.base_port),
            "--duration=999999",
            # Suppress per-iteration output so we don't pollute stdout
            "-z",
        ]

    def client_cmd(self, bw_pct: int) -> List[str]:
        # ib_write_bw --rate uses Mbps units.
        # 200 Gbps = 200,000 Mbps.  bw_pct of that.
        rate_mbps = int(LINK_CAPACITY_GBPS * 1000 * bw_pct / 100.0)
        cmd = [
            "ib_write_bw",
            "-d", self.mlx_device,
            self.remote_ip,
            "-p", str(self.base_port),
            f"--rate={rate_mbps}",
            "--duration=999999",
            "-z",  # suppress output
        ]
        return cmd


class EPTrafficGenerator:
    """
    Controls ib_write_bw server/client pairs to generate background RDMA traffic.

    Two operating modes:
      - continuous : constant-rate sustained traffic at `bw_pct` of link capacity
      - bursty     : ON/OFF bursts at full rate, duty-cycle = bw_pct/100

    Thread-safe. Catches missing ib_write_bw gracefully (warns and no-ops).
    """

    def __init__(self, config: EPTrafficConfig, mode: str = "continuous") -> None:
        self.config = config
        self.mode = mode  # "continuous" or "bursty"
        self._lock = threading.Lock()
        self._running = False
        self._server_proc: Optional[subprocess.Popen] = None
        self._client_proc: Optional[subprocess.Popen] = None
        self._burst_thread: Optional[threading.Thread] = None
        self._burst_stop_event = threading.Event()
        self._bw_pct: int = 0

        # Check availability once at construction time
        self._ib_write_bw_available = self._check_ib_write_bw()

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

            if not self._ib_write_bw_available:
                print(
                    "[EPTrafficGenerator] ib_write_bw not found — continuing without "
                    "EP background traffic.",
                    flush=True,
                )
                return

            self._start_server()
            time.sleep(0.5)  # let server bind the port

            if self.mode == "bursty" and bw_pct > 0 and bw_pct < 100:
                self._start_bursty(bw_pct)
            else:
                self._start_client(bw_pct)

            print(
                f"[EPTrafficGenerator] started ({self.mode} mode, {bw_pct}% BW, "
                f"local={self.config.local_ip}, remote={self.config.remote_ip})",
                flush=True,
            )

    def stop(self) -> None:
        """Stop all traffic-generating processes gracefully."""
        with self._lock:
            self._running = False

            # Stop burst thread first
            if self._burst_thread is not None:
                self._burst_stop_event.set()
                self._burst_thread.join(timeout=2.0)
                self._burst_thread = None
                self._burst_stop_event.clear()

            # Kill client then server
            for proc, name in [
                (self._client_proc, "client"),
                (self._server_proc, "server"),
            ]:
                if proc is not None:
                    self._terminate_proc(proc, name)

            self._server_proc = None
            self._client_proc = None

            if self._ib_write_bw_available:
                print("[EPTrafficGenerator] stopped", flush=True)

    def is_running(self) -> bool:
        """Return True if traffic generation is active."""
        with self._lock:
            if not self._running:
                return False
            # Re-check process liveness
            for proc, name in [
                (self._client_proc, "client"),
                (self._server_proc, "server"),
            ]:
                if proc is not None and proc.poll() is not None:
                    print(
                        f"[EPTrafficGenerator] {name} process died unexpectedly — "
                        f"marking stopped",
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

    def _start_server(self) -> None:
        env = os.environ.copy()
        env["IBV_DEVICE_NAME"] = self.config.mlx_device
        self._server_proc = subprocess.Popen(
            self.config.server_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,  # new process group for clean kill
        )

    def _start_client(self, bw_pct: int) -> None:
        env = os.environ.copy()
        env["IBV_DEVICE_NAME"] = self.config.mlx_device
        self._client_proc = subprocess.Popen(
            self.config.client_cmd(bw_pct=bw_pct),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )

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
                burst_proc = subprocess.Popen(
                    self.config.client_cmd(bw_pct=100),  # full rate during ON
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    preexec_fn=os.setsid,
                )
                try:
                    # Sleep for ON duration, checking stop every 10 ms
                    deadline = time.monotonic() + T_on
                    while time.monotonic() < deadline and not self._burst_stop_event.is_set():
                        time.sleep(0.01)
                finally:
                    self._terminate_proc(burst_proc, f"burst[{burst_iter}]-client")

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
