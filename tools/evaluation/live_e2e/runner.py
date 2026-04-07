import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

from tools.case_study.common import ensure_dir, write_json
from tools.evaluation.live_e2e.manifest import LiveE2EManifest, LiveE2ERun
from tools.evaluation.live_e2e.manifest import load_manifest


def build_benchmark_command(run: LiveE2ERun, benchmark_script: str) -> str:
    """Build the benchmark_lora.sh command string from a run definition."""
    parts = [benchmark_script]
    parts.append(f"--compute_device {run.compute_device}")

    if run.miss_handling_mode is not None:
        parts.append(f"--colora_miss_policy {run.miss_handling_mode}")
    if run.overlap_policy is not None:
        parts.append(f"--colora_overlap_mode {run.overlap_policy}")
    if run.async_fallback is not None:
        parts.append(f"--colora_async_fallback {str(run.async_fallback).lower()}")
    if run.cpu_workers is not None:
        parts.append(f"--colora_cpu_workers {run.cpu_workers}")
    if run.cpu_queue_depth is not None:
        parts.append(f"--colora_cpu_queue_depth {run.cpu_queue_depth}")
    if run.cpu_batch_timeout_us is not None:
        parts.append(f"--colora_cpu_batch_timeout_us {run.cpu_batch_timeout_us}")
    if run.speculative_dispatch is not None:
        if run.speculative_dispatch:
            parts.append("--colora_speculative_dispatch")
        else:
            parts.append("--no_colora_speculative_dispatch")
    if run.warmup_requests is not None:
        parts.append(f"--warmup_num_requests {run.warmup_requests}")
    if run.measurement_requests is not None:
        parts.append(f"--measure_num_requests {run.measurement_requests}")

    # The per-request log path will be determined at runtime in the run directory
    return " ".join(parts)


def build_nsys_command(base_cmd: str, output_path: Path, run: LiveE2ERun) -> str:
    """Wrap the benchmark command with nsys profile."""
    prefix = run.nsys_output_prefix or f"{run.run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    nsys_cmd = (
        f"nsys profile --stats=true --output {output_path / prefix} {base_cmd}"
    )
    return nsys_cmd


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


def run_single(manifest: LiveE2EManifest, run: LiveE2ERun, capture_stdout: bool = True) -> dict:
    """Execute a single run, capture outputs, return run result metadata."""
    output_dir = get_run_output_dir(manifest, run)
    ensure_dir(output_dir)

    # Write config snapshot before running
    config_snapshot_path = output_dir / "config_snapshot.json"
    write_json(config_snapshot_path, run.to_metadata())

    # Build the command
    base_cmd = build_benchmark_command(run, manifest.benchmark_script)
    # Append the per-request log path to our output directory
    per_request_log_path = output_dir / "per_request_metrics.jsonl"
    base_cmd += f" --per_request_log_path {per_request_log_path}"

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
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_stdout,
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

    if capture_stdout:
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

    # Validate result
    valid = True
    failure_reason = None

    if proc.returncode != 0:
        valid = False
        failure_reason = f"benchmark exited with code {proc.returncode}"

    if not per_request_log_path.exists():
        valid = False
        failure_reason = "per-request log file not created"

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
        "start_time": start_time.isoformat(),
        "nsys_enabled": run.nsys_enabled,
    }

    write_json(output_dir / "run_result.json", result)
    print(f"=== Completed {run.run_label}: valid={valid}")
    if failure_reason:
        print(f"Failure: {failure_reason}")

    return result


def run_manifest(manifest_path: Path) -> list[dict]:
    """Run all runs defined in a manifest."""
    manifest = load_manifest(manifest_path)
    print(f"Loaded manifest run_id={manifest.run_id}, {len(manifest.runs)} runs")

    results = []
    for run in manifest.runs:
        result = run_single(manifest, run)
        results.append(result)

    # Write overall results summary
    summaries_dir = get_summaries_dir(manifest)
    ensure_dir(summaries_dir)
    write_json(summaries_dir / "manifest_results.json", {
        "manifest_path": str(manifest_path),
        "run_id": manifest.run_id,
        "description": manifest.description,
        "total_runs": len(results),
        "valid_runs": sum(1 for r in results if r["valid"]),
        "results": results,
    })

    print(f"\n=== All runs complete ===")
    print(f"Total: {len(results)}, Valid: {sum(1 for r in results if r['valid'])}")
    print(f"Results in: {summaries_dir}")

    return results
