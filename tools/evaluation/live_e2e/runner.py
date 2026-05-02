import shlex
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
import json
import threading
from typing import Any, Dict, List, Optional
import yaml

from tools.case_study.common import ensure_dir, write_json
from tools.evaluation.live_e2e.manifest import LiveE2EManifest, LiveE2ERun
from tools.evaluation.live_e2e.manifest import load_manifest


def _save_git_info(output_dir: Path) -> None:
    """Save git commit hash and dirty-status to the run output directory."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        if status:
            (output_dir / "git_status.txt").write_text(status + "\n", encoding="utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def _save_manifest_yaml(output_dir: Path, run: LiveE2ERun) -> None:
    """Save the resolved run config as manifest.yaml."""
    meta = run.to_metadata()
    with (output_dir / "manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


_COLORA_LINE_RE = re.compile(r"\[COLoRA\]\s+layer=.*")
_KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


def _extract_colora_stats_from_log(stdout_log_path: Path) -> Dict[str, Any]:
    """Parse [COLoRA] lines from server stdout log and aggregate stats.

    Returns a dict with aggregated (sum or last) values across all layers.
    If no [COLoRA] lines found, returns an empty dict.
    """
    if not stdout_log_path.exists():
        return {}

    # Collect per-key values across all layers
    per_key: Dict[str, List[float]] = {}
    with stdout_log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not _COLORA_LINE_RE.search(line):
                continue
            for k, v in _KV_RE.findall(line):
                try:
                    fv = float(v.rstrip(","))
                except ValueError:
                    continue
                per_key.setdefault(k, []).append(fv)

    if not per_key:
        return {}

    # Aggregate: sum for counters, last for gauges/config
    counter_keys = {
        "colora_hit_tokens", "colora_miss_tokens",
        "cpu_compute_time", "gpu_compute_time",
        "cpu_queue_wait_time", "cpu_join_stall_time",
        "d2h_bytes", "h2d_bytes",
        "weight_h2d_bytes", "weight_h2d_time",
        "d2h_activation_time", "h2d_residual_time",
        "pack_time", "merge_time", "admit_time",
        "fallback_degrade_count",
        "promotion_drop_total", "promotion_admitted",
        "promotion_reject_delta", "promotion_reject_no_ema",
        "tracker_queue_drop",
        "prefetch_submitted", "prefetch_ready_hits",
        "prefetch_not_ready", "prefetch_stale",
        "prefetch_false_positives", "prefetch_slot_overwrite",
        "moe_kernel_calls", "moe_kernel_tokens",
        "attempted_bind", "successful_bind",
        "cpu_async_submitted", "cpu_inline_executed",
        "blocking_promotion_count",
        "cache_evictions_total",
    }
    gauge_keys = {
        "cpu_queue_depth", "promotion_drop_queue_high_watermark",
        "cache_capacity_slots", "cache_resident_slots", "cache_free_slots",
        "overlap_ratio",
    }

    result: Dict[str, Any] = {}
    for k, vs in per_key.items():
        if k in counter_keys:
            result[k] = sum(vs)
        elif k in gauge_keys:
            result[k] = vs[-1] if vs else 0
        else:
            # Default: take the last value for unknown keys
            result[k] = vs[-1] if vs else 0

    # Compute derived fields
    hit = result.get("colora_hit_tokens", 0)
    miss = result.get("colora_miss_tokens", 0)
    total = hit + miss
    if total > 0:
        result["cache_hit_rate"] = hit / total
        result["cache_miss_rate"] = miss / total

    return result


def build_benchmark_command(run: LiveE2ERun, benchmark_script: str) -> str:
    """Build the benchmark_lora.sh command string from a run definition."""
    parts = []
    if run.coalescing_packer is not None:
        parts.append(f"MOE_COALESCING_PACKER={1 if run.coalescing_packer else 0}")
    if run.cpu_kernel_mode is not None:
        parts.append(f"COLORA_CPU_KERNEL_MODE={str(run.cpu_kernel_mode).strip().lower()}")
    parts.append(benchmark_script)
    parts.append(f"--compute_device {run.compute_device}")
    if run.adapter_ids is not None:
        parts.append(f"--adapter_ids {run.adapter_ids}")
    if run.lora_dirs is not None:
        parts.append(f"--lora_dirs {run.lora_dirs}")
    if run.lora_clone_count is not None:
        parts.append(f"--lora_clone_count {run.lora_clone_count}")
    if run.server_host is not None:
        parts.append(f"--server_host {run.server_host}")
    if run.server_port is not None:
        parts.append(f"--server_port {run.server_port}")

    if run.mode_label is not None:
        parts.append(f"--mode_label {run.mode_label}")
    if run.miss_handling_mode is not None:
        parts.append(f"--colora_miss_policy {run.miss_handling_mode}")
    if run.overlap_mode is not None:
        parts.append(f"--colora_overlap_mode {run.overlap_mode}")
    if run.overlap_policy is not None:
        # Translate overlap policy into request-skip control flags.
        # no_overlap => disable request-level skip-and-reinsert.
        if str(run.overlap_policy).strip().lower() == "no_overlap":
            parts.append("--colora_request_skip 0")
        else:
            # Any non-no_overlap value (e.g. request_skip, skip_reinsert_stability)
            parts.append("--colora_request_skip 1")
    if run.async_fallback is not None:
        parts.append(f"--colora_async_fallback {1 if run.async_fallback else 0}")
    if run.cpu_workers is not None:
        parts.append(f"--colora_cpu_workers {run.cpu_workers}")
    if run.cpu_queue_depth is not None:
        parts.append(f"--colora_cpu_queue_depth {run.cpu_queue_depth}")
    if run.cpu_batch_timeout_us is not None:
        parts.append(f"--colora_cpu_batch_timeout_us {run.cpu_batch_timeout_us}")
    if run.max_continuations is not None:
        parts.append(f"--colora_max_continuations {run.max_continuations}")
    if run.colora_hit_indexing is not None:
        parts.append(f"--colora_hit_indexing {str(run.colora_hit_indexing).strip().lower()}")
    if run.temporal_prefetch is not None:
        if run.temporal_prefetch:
            parts.append("--colora_temporal_prefetch")
        else:
            parts.append("--no_colora_temporal_prefetch")
    if run.temporal_prefetch_layer_whitelist is not None:
        parts.append(f"--colora_temporal_prefetch_layer_whitelist {run.temporal_prefetch_layer_whitelist}")
    if run.temporal_hot_cache_slots is not None:
        parts.append(f"--colora_temporal_hot_cache_slots {run.temporal_hot_cache_slots}")
    if run.cache_budget_mb is not None:
        parts.append(f"--colora_cache_budget_mb {run.cache_budget_mb}")
    if run.promote_min_hits is not None:
        parts.append(f"--colora_promote_min_hits {run.promote_min_hits}")
    if run.promote_window is not None:
        parts.append(f"--colora_promote_window {run.promote_window}")
    if run.max_promote_per_step is not None:
        parts.append(f"--colora_max_promote_per_step {run.max_promote_per_step}")
    if run.decay is not None:
        parts.append(f"--colora_decay {run.decay}")
    if run.deferred_promotion_delta_steps is not None:
        parts.append(f"--colora_deferred_promotion_delta_steps {run.deferred_promotion_delta_steps}")
    if run.promotion_ema_alpha is not None:
        parts.append(f"--colora_promotion_ema_alpha {run.promotion_ema_alpha}")
    if run.speculative_dispatch is not None:
        if run.speculative_dispatch:
            parts.append("--colora_speculative_dispatch")
        else:
            parts.append("--no_colora_speculative_dispatch")
    if run.spec_layer_whitelist is not None:
        parts.append(f"--colora_spec_layer_whitelist {run.spec_layer_whitelist}")
    if run.warmup_requests is not None:
        parts.append(f"--warmup_num_requests {run.warmup_requests}")
    if run.measurement_requests is not None:
        parts.append(f"--measure_num_requests {run.measurement_requests}")
    if run.warmup_adapter_trace_path is not None:
        parts.append(f"--warmup_adapter_trace_path {run.warmup_adapter_trace_path}")
    if run.measurement_adapter_trace_path is not None:
        parts.append(f"--measure_adapter_trace_path {run.measurement_adapter_trace_path}")
    if run.max_concurrent_requests is not None:
        parts.append(f"--max_concurrent_requests {run.max_concurrent_requests}")
    if run.rps is not None:
        parts.append(f"--rps {run.rps}")

    # The per-request log path will be determined at runtime in the run directory
    return " ".join(parts)


def build_nsys_command(base_cmd: str, output_path: Path, run: LiveE2ERun) -> str:
    """Wrap the benchmark command with nsys profile.

    nsys expects an *executable* after its own flags. A line like
    ``VAR=1 ./script.sh`` is wrong: ``VAR=1`` is treated as the app name and
    fails with "Executable not found". We therefore run the full shell line via
    ``bash -c <quoted>``.

    Do **not** insert a standalone ``--`` before ``bash``: this nsys CLI treats
    ``--`` as an ambiguous option prefix (not POSIX end-of-options) and errors.

    Default trace is ``cuda,nvtx`` (no ``osrt``): with multi-process Python + CUDA,
    ``osrt`` often produces very large streams and Nsight importers can fail with
    "Wrong event order". Override with ``run.nsys_trace`` (e.g. ``cuda,nvtx,osrt``)
    when you need libc waits.

    Mirrors common manual usage::
      nsys profile --trace=cuda,nvtx --output=... bash -c '...'
    """
    prefix = run.nsys_output_prefix or f"{run.run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_file = output_path / prefix
    trace = run.nsys_trace or "cuda,nvtx"
    parts = [
        "nsys",
        "profile",
        "--stats=false",
        f"--trace={trace}",
        f"--output={shlex.quote(str(out_file))}",
    ]
    if run.nsys_force_overwrite:
        parts.append("--force-overwrite=true")
    delay = run.nsys_delay_seconds
    if delay is not None:
        parts.append(f"--delay={int(delay)}")
    parts.append("bash")
    parts.append("-c")
    parts.append(shlex.quote(base_cmd))
    return " ".join(parts)


def get_run_output_dir(manifest: LiveE2EManifest, run: LiveE2ERun) -> Path:
    """Get the output directory for a run, following the artifact layout."""
    if run.suite_kind == "paper":
        return run.output_root / manifest.run_id / "paper_runs" / run.run_label
    elif run.suite_kind == "diagnostic":
        return run.output_root / manifest.run_id / "diagnostic_runs" / run.run_label
    else:
        raise ValueError(f"Unknown suite_kind: {run.suite_kind}")


def get_summaries_dir(manifest: LiveE2EManifest) -> Path:
    """Get the summaries directory for an evaluation."""
    return manifest.runs[0].output_root / manifest.run_id / "summaries"


def read_existing_run_result(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``run_result.json`` if present and parseable; otherwise None."""
    path = output_dir / "run_result.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _stream_pipe(pipe, log_file, output_stream):
    """Stream a subprocess pipe to terminal and log file (tee-style)."""
    for line in iter(pipe.readline, ""):
        log_file.write(line)
        log_file.flush()
        output_stream.write(line)
        output_stream.flush()
    pipe.close()


def run_single(manifest: LiveE2EManifest, run: LiveE2ERun, capture_stdout: bool = True) -> dict:
    """Execute a single run, capture outputs, return run result metadata."""
    output_dir = get_run_output_dir(manifest, run)
    ensure_dir(output_dir)

    # Write config snapshot before running
    config_snapshot_path = output_dir / "config_snapshot.json"
    write_json(config_snapshot_path, run.to_metadata())

    # Save resolved manifest, git info
    _save_manifest_yaml(output_dir, run)
    _save_git_info(output_dir)

    # Build the command
    base_cmd = build_benchmark_command(run, manifest.benchmark_script)
    # Append the per-request log path to our output directory
    per_request_log_path = output_dir / "per_request_metrics.jsonl"
    base_cmd += f" --per_request_log_path {per_request_log_path}"
    # Append colora stats dump path
    colora_stats_path = output_dir / "colora_stats.json"
    base_cmd += f" --colora_stats_path {colora_stats_path}"

    # Save the full benchmark command for reproducibility
    (output_dir / "benchmark_command.txt").write_text(base_cmd + "\n", encoding="utf-8")

    if run.nsys_enabled:
        cmd = build_nsys_command(base_cmd, output_dir, run)
    else:
        cmd = base_cmd

    print(f"\n=== Running {run.run_label} ===")
    print(f"Command: {cmd}")
    print(f"Output directory: {output_dir}")

    # Execute and capture output
    stdout_path = output_dir / "benchmark_stdout.log"
    stderr_path = output_dir / "benchmark_stderr.log"

    start_time = datetime.now()
    proc = None
    try:
        if capture_stdout:
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                t_out = threading.Thread(
                    target=_stream_pipe, args=(proc.stdout, stdout_file, sys.stdout), daemon=True
                )
                t_err = threading.Thread(
                    target=_stream_pipe, args=(proc.stderr, stderr_file, sys.stderr), daemon=True
                )
                t_out.start()
                t_err.start()
                proc.wait()
                t_out.join()
                t_err.join()
        else:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=False,
                text=True,
            )
    except Exception as e:
        end_time = datetime.now()
        result = {
            "run_label": run.run_label,
            "suite_kind": run.suite_kind,
            "valid": False,
            "failure_stage": "execution",
            "error": str(e),
            "start_time": start_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "command": cmd,
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "run_result.json", result)
        return result

    end_time = datetime.now()

    # Extract colora_stats: prefer the file created by the benchmark's curl of /colora_stats;
    # if missing/empty, fall back to parsing [COLoRA] lines from the server stdout log.
    if not colora_stats_path.exists() or colora_stats_path.stat().st_size <= 2:
        log_stats = _extract_colora_stats_from_log(stdout_path)
        if log_stats:
            write_json(colora_stats_path, log_stats)
            print(f"  colora_stats.json: extracted from server log ({len(log_stats)} fields)")

    # colora_stats.json is optional for load_then_run but expected for cpu_first
    valid = True
    failure_reason = None

    if proc.returncode != 0:
        valid = False
        failure_reason = f"benchmark exited with code {proc.returncode}"

    if not per_request_log_path.exists():
        valid = False
        failure_reason = "per-request log file not created"

    # colora_stats.json is optional for load_then_run but expected for cpu_first
    colora_stats_exists = colora_stats_path.exists()

    if run.nsys_enabled:
        # Check that at least one nsys file was created
        nsys_files = list(output_dir.glob("*.qdrep")) + list(output_dir.glob("*.nsys-rep"))
        if not nsys_files:
            valid = False
            failure_reason = "nsys enabled but no profiling artifact created"

    result = {
        "run_label": run.run_label,
        "suite_kind": run.suite_kind,
        "mode_label": run.mode_label,
        "valid": valid,
        "failure_reason": failure_reason,
        "exit_code": proc.returncode,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "command": cmd,
        "output_dir": str(output_dir),
        "per_request_log_path": str(per_request_log_path),
        "colora_stats_path": str(colora_stats_path),
        "colora_stats_exists": colora_stats_exists,
        "start_time": start_time.isoformat(),
        "nsys_enabled": run.nsys_enabled,
    }

    write_json(output_dir / "run_result.json", result)
    print(f"=== Completed {run.run_label}: valid={valid}")
    if failure_reason:
        print(f"Failure: {failure_reason}")

    return result


def run_manifest(manifest_path: Path, *, overwrite: bool = False) -> List[Dict[str, Any]]:
    """Run all runs defined in a manifest.

    By default (``overwrite=False``), a run is **not** re-executed if its output
    directory already contains ``run_result.json`` with ``"valid": true``. This
    supports resuming after flaky nsys or partial failures. Pass
    ``overwrite=True`` to always execute every run regardless of prior results.
    """
    manifest = load_manifest(manifest_path)
    print(f"Loaded manifest run_id={manifest.run_id}, {len(manifest.runs)} runs")
    if overwrite:
        print("Overwrite: re-running all runs (ignoring existing valid run_result.json).")
    else:
        print("Resume: runs with existing valid run_result.json will be skipped (use --overwrite to re-run).")

    results: List[Dict[str, Any]] = []
    for run in manifest.runs:
        output_dir = get_run_output_dir(manifest, run)
        if not overwrite:
            prev = read_existing_run_result(output_dir)
            if prev is not None and prev.get("valid") is True:
                merged = {**prev, "skipped": True}
                results.append(merged)
                print(f"\n=== Skipping {run.run_label} ===")
                print(f"Existing run_result.json has valid=true. Use --overwrite to re-run.")
                print(f"Output directory: {output_dir}")
                continue

        result = run_single(manifest, run)
        results.append(result)

    skipped_valid = sum(1 for r in results if r.get("skipped"))
    executed = len(results) - skipped_valid

    # Write overall results summary
    summaries_dir = get_summaries_dir(manifest)
    ensure_dir(summaries_dir)
    write_json(summaries_dir / "manifest_results.json", {
        "manifest_path": str(manifest_path),
        "run_id": manifest.run_id,
        "description": manifest.description,
        "total_runs": len(results),
        "valid_runs": sum(1 for r in results if r["valid"]),
        "executed_runs": executed,
        "skipped_valid_runs": skipped_valid,
        "overwrite": overwrite,
        "results": results,
    })

    print(f"\n=== All runs complete ===")
    print(
        f"Total: {len(results)}, Valid: {sum(1 for r in results if r['valid'])}, "
        f"Executed: {executed}, Skipped (prior valid): {skipped_valid}"
    )
    print(f"Results in: {summaries_dir}")

    return results
