"""19-timestamp request timeline, accounting model, and immutable run metadata.

See spec section "Common instrumentation". Timestamps are stored as float
milliseconds (perf_counter*1e3 or similar). Null means "not applicable for
this cell" (e.g., t2/t3 for persistent TCP), NOT zero.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

# All timestamp names in canonical order
TIMESTAMPS = [
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9",
    "t10", "t11", "t12", "t13", "t14", "t15", "t16", "t17", "t18",
]
CLIENT_ADMISSION = ["c0", "c1", "c2"]
SCHEDULER = ["s0", "s1", "s2", "s3", "s4", "s5"]


@dataclass
class RequestTimeline:
    """Per-request timestamps. Null = not applicable (e.g. t2/t3 for persistent TCP)."""
    req_id: int
    cell: str = ""
    _ts: dict = field(default_factory=dict)

    def set(self, name: str, value: Optional[float]) -> None:
        if value is None:
            self._ts[name] = None
        else:
            self._ts[name] = float(value)

    def get(self, name: str) -> Optional[float]:
        return self._ts.get(name)

    def e2e_us(self) -> float:
        t0, t18 = self._ts.get("t0"), self._ts.get("t18")
        if t0 is None or t18 is None:
            return float("nan")
        return (t18 - t0) * 1e3

    def client_request_span_us(self) -> float:
        t0, t5 = self._ts.get("t0"), self._ts.get("t5")
        if t0 is None or t5 is None:
            return float("nan")
        return (t5 - t0) * 1e3

    def server_span_us(self) -> float:
        t6, t17 = self._ts.get("t6"), self._ts.get("t17")
        if t6 is None or t17 is None:
            return float("nan")
        return (t17 - t6) * 1e3


@dataclass
class RunMetadata:
    """Immutable per-run metadata, attached to every result row."""
    schema_version: int = 1
    git_commit: str = ""
    config_hash: str = ""
    hostnames: str = ""
    hardware_ids: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    pytorch_version: str = ""
    rdma_core_version: str = ""
    gpu_clocks: str = ""
    cpu_affinity: str = ""
    numa_node: str = ""
    random_seed: int = 0
    trace_id: str = ""
    cell_id: str = ""
    policy_id: str = ""
    trial_id: int = 0
    start_timestamp: str = ""
    success_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def account_request(tl: RequestTimeline,
                    instrumented_client_intervals_us: list = None,
                    instrumented_server_intervals_us: list = None) -> dict:
    """Compute the accounting decomposition per spec section "Accounting model".

    Returns a dict with client_request_span_us, server_span_us, e2e_us,
    client_local_gap_us, server_local_gap_us, instrumentation_gap_us,
    instrumentation_gap_fraction, cross_domain_residual_us,
    cross_domain_residual_fraction.
    """
    instrumented_client_intervals_us = instrumented_client_intervals_us or []
    instrumented_server_intervals_us = instrumented_server_intervals_us or []

    e2e = tl.e2e_us()
    client_span = tl.client_request_span_us()
    server_span = tl.server_span_us()

    client_local_gap = client_span - sum(instrumented_client_intervals_us)
    server_local_gap = server_span - sum(instrumented_server_intervals_us)
    instrumentation_gap = client_local_gap + server_local_gap

    cross_domain_residual = e2e - client_span - server_span

    def frac(part, whole):
        return (part / whole) if whole and whole == whole and whole > 0 else 0.0  # NaN-safe

    return {
        "e2e_us": e2e,
        "client_request_span_us": client_span,
        "server_span_us": server_span,
        "client_local_gap_us": client_local_gap,
        "server_local_gap_us": server_local_gap,
        "instrumentation_gap_us": instrumentation_gap,
        "instrumentation_gap_fraction": frac(instrumentation_gap, e2e),
        "cross_domain_residual_us": cross_domain_residual,
        "cross_domain_residual_fraction": frac(cross_domain_residual, e2e),
    }


def should_flag_instrumentation_gap(acc: dict, abs_threshold_us: float = 50.0,
                                    frac_threshold: float = 0.05) -> bool:
    """Per spec: flag if BOTH fraction > 5% AND absolute > 50us."""
    return (acc["instrumentation_gap_fraction"] > frac_threshold
            and acc["instrumentation_gap_us"] > abs_threshold_us)
