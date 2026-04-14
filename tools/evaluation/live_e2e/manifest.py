from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class LiveE2ERun:
    """Single run definition for live E2E evaluation."""
    run_label: str
    suite_kind: str  # "paper" or "diagnostic"
    mode_label: str  # "baseline", "execution_first", "load_then_run", etc.
    compute_device: str
    miss_handling_mode: Optional[str] = None
    overlap_policy: Optional[str] = None
    overlap_mode: Optional[str] = None
    async_fallback: Optional[bool] = None
    cpu_workers: Optional[int] = None
    cpu_queue_depth: Optional[int] = None
    cpu_batch_timeout_us: Optional[int] = None
    max_continuations: Optional[int] = None
    cpu_kernel_mode: Optional[str] = None
    coalescing_packer: Optional[bool] = None
    speculative_dispatch: Optional[bool] = None
    spec_layer_whitelist: Optional[str] = None
    temporal_prefetch: Optional[bool] = None
    temporal_prefetch_layer_whitelist: Optional[str] = None
    temporal_hot_cache_slots: Optional[int] = None
    cache_budget_mb: Optional[int] = None
    promote_min_hits: Optional[int] = None
    promote_window: Optional[int] = None
    max_promote_per_step: Optional[int] = None
    decay: Optional[float] = None
    deferred_promotion_delta_steps: Optional[int] = None
    promotion_ema_alpha: Optional[float] = None
    requests_path: Optional[str] = None
    adapter_trace_path: Optional[str] = None
    warmup_adapter_trace_path: Optional[str] = None
    measurement_adapter_trace_path: Optional[str] = None
    server_host: Optional[str] = None
    server_port: Optional[int] = None
    adapter_ids: Optional[str] = None
    lora_dirs: Optional[str] = None
    output_root: Path = Path("artifacts/evaluation/live_e2e")
    nsys_enabled: bool = False
    nsys_output_prefix: Optional[str] = None
    # Passed to nsys profile (optional). Defaults applied in runner.build_nsys_command.
    nsys_trace: Optional[str] = None  # e.g. "cuda,nvtx" or "cuda,nvtx,osrt"; default if omitted
    nsys_force_overwrite: bool = True
    nsys_delay_seconds: Optional[int] = None  # nsys --delay=N (seconds before capture)
    warmup_requests: int = 0
    measurement_requests: Optional[int] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for snapshotting."""
        return {
            "run_label": self.run_label,
            "suite_kind": self.suite_kind,
            "mode_label": self.mode_label,
            "compute_device": self.compute_device,
            "miss_handling_mode": self.miss_handling_mode,
            "overlap_policy": self.overlap_policy,
            "overlap_mode": self.overlap_mode,
            "async_fallback": self.async_fallback,
            "cpu_workers": self.cpu_workers,
            "cpu_queue_depth": self.cpu_queue_depth,
            "cpu_batch_timeout_us": self.cpu_batch_timeout_us,
            "max_continuations": self.max_continuations,
            "cpu_kernel_mode": self.cpu_kernel_mode,
            "coalescing_packer": self.coalescing_packer,
            "speculative_dispatch": self.speculative_dispatch,
            "spec_layer_whitelist": self.spec_layer_whitelist,
            "temporal_prefetch": self.temporal_prefetch,
            "temporal_prefetch_layer_whitelist": self.temporal_prefetch_layer_whitelist,
            "temporal_hot_cache_slots": self.temporal_hot_cache_slots,
            "cache_budget_mb": self.cache_budget_mb,
            "promote_min_hits": self.promote_min_hits,
            "promote_window": self.promote_window,
            "max_promote_per_step": self.max_promote_per_step,
            "decay": self.decay,
            "deferred_promotion_delta_steps": self.deferred_promotion_delta_steps,
            "promotion_ema_alpha": self.promotion_ema_alpha,
            "requests_path": self.requests_path,
            "adapter_trace_path": self.adapter_trace_path,
            "warmup_adapter_trace_path": self.warmup_adapter_trace_path,
            "measurement_adapter_trace_path": self.measurement_adapter_trace_path,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "adapter_ids": self.adapter_ids,
            "lora_dirs": self.lora_dirs,
            "output_root": str(self.output_root),
            "nsys_enabled": self.nsys_enabled,
            "nsys_output_prefix": self.nsys_output_prefix,
            "nsys_trace": self.nsys_trace,
            "nsys_force_overwrite": self.nsys_force_overwrite,
            "nsys_delay_seconds": self.nsys_delay_seconds,
            "warmup_requests": self.warmup_requests,
            "measurement_requests": self.measurement_requests,
        }


@dataclass
class LiveE2EManifest:
    """Top-level manifest for a collection of live E2E runs."""
    run_id: str
    runs: List[LiveE2ERun] = field(default_factory=list)
    description: Optional[str] = None
    benchmark_script: str = "test/lora/benchmark_lora.sh"


def load_manifest(manifest_path: Path) -> LiveE2EManifest:
    """Load and validate a manifest from YAML."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    runs = []
    for run_data in data.get("runs", []):
        if "output_root" in run_data:
            run_data["output_root"] = Path(run_data["output_root"])
        try:
            run = LiveE2ERun(**run_data)
        except TypeError as e:
            raise ValueError(f"Invalid run definition in manifest: {e}") from e

        # Backward-compatibility alias:
        # legacy adapter_trace_path acts as measurement trace when explicit field is absent.
        if run.measurement_adapter_trace_path is None and run.adapter_trace_path is not None:
            run.measurement_adapter_trace_path = run.adapter_trace_path

        if run.suite_kind not in ("paper", "diagnostic"):
            raise ValueError(f"Invalid suite_kind: {run.suite_kind}, must be 'paper' or 'diagnostic'")

        runs.append(run)

    return LiveE2EManifest(
        run_id=data["run_id"],
        runs=runs,
        description=data.get("description"),
        benchmark_script=data.get("benchmark_script", "test/lora/benchmark_lora.sh"),
    )
