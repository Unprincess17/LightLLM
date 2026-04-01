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
    async_fallback: Optional[bool] = None
    cpu_workers: Optional[int] = None
    cpu_queue_depth: Optional[int] = None
    cpu_batch_timeout_us: Optional[int] = None
    speculative_dispatch: Optional[bool] = None
    requests_path: str = "fixed_requests.jsonl"
    adapter_trace_path: str = "adapter_trace.jsonl"
    output_root: Path = Path("artifacts/evaluation/live_e2e")
    nsys_enabled: bool = False
    nsys_output_prefix: Optional[str] = None
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
            "async_fallback": self.async_fallback,
            "cpu_workers": self.cpu_workers,
            "cpu_queue_depth": self.cpu_queue_depth,
            "cpu_batch_timeout_us": self.cpu_batch_timeout_us,
            "speculative_dispatch": self.speculative_dispatch,
            "requests_path": self.requests_path,
            "adapter_trace_path": self.adapter_trace_path,
            "output_root": str(self.output_root),
            "nsys_enabled": self.nsys_enabled,
            "nsys_output_prefix": self.nsys_output_prefix,
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

        if run.suite_kind not in ("paper", "diagnostic"):
            raise ValueError(f"Invalid suite_kind: {run.suite_kind}, must be 'paper' or 'diagnostic'")

        runs.append(run)

    return LiveE2EManifest(
        run_id=data["run_id"],
        runs=runs,
        description=data.get("description"),
        benchmark_script=data.get("benchmark_script", "test/lora/benchmark_lora.sh"),
    )
