#!/usr/bin/env python3
"""Calibrate transfer, launch, compute, and overlap terms for the TPOT baseline."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import ensure_parent_dir, load_global_config, stage_output_dir, write_json
from system_tpot_core import (
    DEFAULT_SLICE_BYTES,
    TRANSFER_MODE_DIRECT_PAGEABLE_FRAGMENTED,
    TRANSFER_MODE_DIRECT_PINNED_PACKED,
    TRANSFER_MODE_STAGED_PAGEABLE_PACKED,
    compute_component_sizes,
    dtype_name_to_torch_dtype,
    load_model_text_config,
    parse_batch_grid,
    parse_batch_value_map,
)


DEFAULT_BATCH_GRID = (1, 2, 4)
DEFAULT_PAYLOAD_GRID = (64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024)
DEFAULT_HOST_PROFILES = ("idle", "stressed")
DEFAULT_WARMUP_ITERS = 20
DEFAULT_MEASURE_ITERS = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the micro-benchmark inputs for system TPOT simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_path", type=str, default=None, help="Override output JSON path")
    parser.add_argument("--model_dir", type=str, default=None, help="Override HF model directory used for dimension lookup")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to benchmark")
    parser.add_argument("--dtype", type=str, default="bf16", help="LoRA dtype used for byte accounting")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank used for byte accounting")
    parser.add_argument("--slice_bytes", type=int, default=DEFAULT_SLICE_BYTES, help="Audit slice size in bytes")
    parser.add_argument("--batch_grid", type=str, default=",".join(str(value) for value in DEFAULT_BATCH_GRID), help="Comma-separated batch grid for overlap and compute benchmarks")
    parser.add_argument("--base_tpot_ms", type=str, required=True, help="Batch:value map for hot-cache base TPOT, e.g. 1:1.2,2:1.4,4:1.9")
    parser.add_argument("--payload_grid_bytes", type=str, default=",".join(str(value) for value in DEFAULT_PAYLOAD_GRID), help="Comma-separated packed-payload byte grid")
    parser.add_argument("--fragmented_slice_grid", type=str, default="1,2,4,8,16,32", help="Comma-separated slice-count grid for direct pageable fragmented copies")
    parser.add_argument("--host_profiles", type=str, default=",".join(DEFAULT_HOST_PROFILES), help="Comma-separated host profiles to benchmark")
    parser.add_argument("--warmup_iters", type=int, default=DEFAULT_WARMUP_ITERS, help="Warmup iterations per benchmark point")
    parser.add_argument("--measure_iters", type=int, default=DEFAULT_MEASURE_ITERS, help="Measured iterations per benchmark point")
    parser.add_argument("--stress_workers", type=int, default=max(1, (os.cpu_count() or 4) // 4), help="Background host-stress workers used for the stressed profile")
    parser.add_argument("--stress_array_mb", type=int, default=256, help="Per-worker array footprint in MiB for the stressed host profile")
    return parser.parse_args()


def summarize_ms(values_ms: Sequence[float]) -> Dict[str, float]:
    ordered = [float(value) for value in values_ms]
    if not ordered:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(sum(ordered) / len(ordered)),
        "p50_ms": float(np.percentile(ordered, 50)),
        "p90_ms": float(np.percentile(ordered, 90)),
        "max_ms": float(max(ordered)),
    }


def _stress_worker(stop_event: mp.synchronize.Event, array_bytes: int) -> None:
    src = np.random.randint(0, 256, size=array_bytes, dtype=np.uint8)
    dst = np.empty_like(src)
    checksum = 0
    while not stop_event.is_set():
        np.copyto(dst, src)
        checksum ^= int(dst[0])
        src, dst = dst, src
    if checksum == 257:
        print("unreachable checksum", flush=True)


class HostStressContext:
    def __init__(self, enabled: bool, worker_count: int, array_mb: int):
        self.enabled = bool(enabled)
        self.worker_count = int(worker_count)
        self.array_mb = int(array_mb)
        self.processes: List[mp.Process] = []
        self.stop_event: mp.synchronize.Event | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        ctx = mp.get_context("spawn")
        self.stop_event = ctx.Event()
        array_bytes = int(self.array_mb * 1024 * 1024)
        for _ in range(self.worker_count):
            process = ctx.Process(target=_stress_worker, args=(self.stop_event, array_bytes), daemon=True)
            process.start()
            self.processes.append(process)
        time.sleep(0.5)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stop_event is not None:
            self.stop_event.set()
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self.processes.clear()
        self.stop_event = None
        return False


def _measure_cuda_op(fn, warmup_iters: int, measure_iters: int) -> List[float]:
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    values_ms: List[float] = []
    for _ in range(measure_iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values_ms.append((time.perf_counter() - start) * 1000.0)
    return values_ms


def benchmark_packed_h2d(device: torch.device, payload_bytes: int, warmup_iters: int, measure_iters: int) -> Dict[str, float]:
    src = torch.empty(payload_bytes, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
    values_ms = _measure_cuda_op(lambda: dst.copy_(src, non_blocking=True), warmup_iters, measure_iters)
    row = {"payload_bytes": int(payload_bytes)}
    row.update(summarize_ms(values_ms))
    return row


def benchmark_launch_overhead(device: torch.device, payload_bytes: int, warmup_iters: int, measure_iters: int) -> Dict[str, float]:
    probe = torch.zeros(1, device=device, dtype=torch.float32)
    values_ms = _measure_cuda_op(lambda: probe.add_(1.0), warmup_iters, measure_iters)
    row = {"payload_bytes": int(payload_bytes)}
    row.update(summarize_ms(values_ms))
    return row


def _fragment_sizes(total_bytes: int, object_bytes: int) -> List[int]:
    remaining = int(total_bytes)
    sizes: List[int] = []
    while remaining > 0:
        chunk = min(int(object_bytes), remaining)
        sizes.append(int(chunk))
        remaining -= int(chunk)
    return sizes


def benchmark_pageable_gather(payload_bytes: int, object_bytes: int, warmup_iters: int, measure_iters: int) -> Dict[str, float]:
    chunk_sizes = _fragment_sizes(payload_bytes, object_bytes)
    pageable_chunks = [torch.randint(0, 255, (size,), dtype=torch.uint8) for size in chunk_sizes]
    staging = torch.empty(payload_bytes, dtype=torch.uint8, pin_memory=True)

    def run_once() -> None:
        offset = 0
        for chunk in pageable_chunks:
            next_offset = offset + int(chunk.numel())
            staging[offset:next_offset].copy_(chunk)
            offset = next_offset

    for _ in range(warmup_iters):
        run_once()
    values_ms: List[float] = []
    for _ in range(measure_iters):
        start = time.perf_counter()
        run_once()
        values_ms.append((time.perf_counter() - start) * 1000.0)
    row = {"payload_bytes": int(payload_bytes)}
    row.update(summarize_ms(values_ms))
    return row


def benchmark_direct_pageable_fragmented(
    device: torch.device,
    slice_bytes: int,
    slice_count: int,
    warmup_iters: int,
    measure_iters: int,
) -> Dict[str, float]:
    payload_bytes = int(slice_bytes) * int(slice_count)
    chunks = [torch.randint(0, 255, (int(slice_bytes),), dtype=torch.uint8) for _ in range(int(slice_count))]
    dst = torch.empty(payload_bytes, dtype=torch.uint8, device=device)

    def run_once() -> None:
        offset = 0
        for chunk in chunks:
            next_offset = offset + int(chunk.numel())
            dst[offset:next_offset].copy_(chunk, non_blocking=False)
            offset = next_offset

    values_ms = _measure_cuda_op(run_once, warmup_iters, measure_iters)
    row = {"slice_count": int(slice_count), "payload_bytes": int(payload_bytes)}
    row.update(summarize_ms(values_ms))
    return row


def benchmark_lora_compute_rows(
    device: torch.device,
    hidden_size: int,
    moe_intermediate_size: int,
    lora_rank: int,
    num_experts_per_tok: int,
    dtype: torch.dtype,
    batch_grid: Sequence[int],
    warmup_iters: int,
    measure_iters: int,
) -> Dict[str, dict]:
    rows_payload: Dict[str, dict] = {}
    for batch_size in batch_grid:
        rows = max(1, int(batch_size) * int(num_experts_per_tok))
        hidden = torch.randn((rows, hidden_size), dtype=dtype, device=device)
        mid = torch.randn((rows, moe_intermediate_size), dtype=dtype, device=device)
        w1_a = torch.randn((hidden_size, lora_rank), dtype=dtype, device=device)
        w1_b = torch.randn((lora_rank, moe_intermediate_size), dtype=dtype, device=device)
        w2_a = torch.randn((moe_intermediate_size, lora_rank), dtype=dtype, device=device)
        w2_b = torch.randn((lora_rank, hidden_size), dtype=dtype, device=device)

        early_values = _measure_cuda_op(lambda: (hidden @ w1_a) @ w1_b, warmup_iters, measure_iters)
        late_values = _measure_cuda_op(lambda: (mid @ w2_a) @ w2_b, warmup_iters, measure_iters)

        def total_fn() -> None:
            early = (hidden @ w1_a) @ w1_b
            late = (mid @ w2_a) @ w2_b
            _ = early + late[:, : early.shape[1]]

        total_values = _measure_cuda_op(total_fn, warmup_iters, measure_iters)
        rows_payload[str(batch_size)] = {
            "rows": int(rows),
            "early": summarize_ms(early_values),
            "late": summarize_ms(late_values),
            "total": summarize_ms(total_values),
        }
    return rows_payload


def benchmark_overlap_windows(
    device: torch.device,
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    dtype: torch.dtype,
    batch_grid: Sequence[int],
    warmup_iters: int,
    measure_iters: int,
) -> Dict[str, dict]:
    payload: Dict[str, dict] = {}
    for batch_size in batch_grid:
        hidden = torch.randn((int(batch_size), hidden_size), dtype=dtype, device=device)
        logits = torch.randn((int(batch_size), num_experts), dtype=torch.float32, device=device)
        gather_index = torch.arange(int(batch_size), device=device, dtype=torch.long).repeat_interleave(int(num_experts_per_tok))
        gate_up_weight = torch.randn((hidden_size, 2 * moe_intermediate_size), dtype=dtype, device=device)

        def early_window() -> None:
            topk_idx = torch.topk(logits, k=int(num_experts_per_tok), dim=-1).indices
            packed = hidden.index_select(0, gather_index)
            _ = packed + topk_idx.reshape(-1, 1).to(dtype=packed.dtype)

        def late_window() -> None:
            packed = hidden.index_select(0, gather_index)
            gate_up = packed @ gate_up_weight
            left, right = torch.chunk(gate_up, 2, dim=-1)
            _ = F.silu(left) * right

        early_values = _measure_cuda_op(early_window, warmup_iters, measure_iters)
        late_values = _measure_cuda_op(late_window, warmup_iters, measure_iters)
        payload[str(batch_size)] = {
            "early": summarize_ms(early_values),
            "late": summarize_ms(late_values),
        }
    return payload


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for calibrate_system_baseline.py")

    config = load_global_config(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(config.get("paths", {}).get("model_dir"))
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else stage_output_dir("calibration", config, args.run_id) / "system_baseline_calibration.json"
    )
    ensure_parent_dir(output_path)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    text_config = load_model_text_config(model_dir)
    dtype = dtype_name_to_torch_dtype(args.dtype)
    batch_grid = parse_batch_grid(args.batch_grid)
    base_tpot_ms = parse_batch_value_map(args.base_tpot_ms)
    payload_grid = [int(token) for token in str(args.payload_grid_bytes).split(",") if token.strip()]
    fragmented_slice_grid = [int(token) for token in str(args.fragmented_slice_grid).split(",") if token.strip()]
    host_profiles = [token.strip() for token in str(args.host_profiles).split(",") if token.strip()]

    object_sizes = compute_component_sizes(
        hidden_size=int(text_config["hidden_size"]),
        moe_intermediate_size=int(text_config["moe_intermediate_size"]),
        lora_rank=int(args.lora_rank),
        dtype=dtype,
        slice_bytes=int(args.slice_bytes),
    )

    transfer_modes: Dict[str, dict] = {
        TRANSFER_MODE_DIRECT_PINNED_PACKED: {"profiles": {}},
        TRANSFER_MODE_STAGED_PAGEABLE_PACKED: {"profiles": {}},
        TRANSFER_MODE_DIRECT_PAGEABLE_FRAGMENTED: {"profiles": {}},
    }

    for profile in host_profiles:
        stressed = profile.lower() == "stressed"
        print(f"benchmarking transfer curves for host profile={profile}")
        with HostStressContext(enabled=stressed, worker_count=int(args.stress_workers), array_mb=int(args.stress_array_mb)):
            pinned_h2d_rows = [
                benchmark_packed_h2d(device, int(payload_bytes), int(args.warmup_iters), int(args.measure_iters))
                for payload_bytes in payload_grid
            ]
            packed_launch_rows = [
                benchmark_launch_overhead(device, int(payload_bytes), int(args.warmup_iters), int(args.measure_iters))
                for payload_bytes in payload_grid
            ]
            packed_gather_rows = [
                benchmark_pageable_gather(int(payload_bytes), int(object_sizes.total_object_bytes), int(args.warmup_iters), int(args.measure_iters))
                for payload_bytes in payload_grid
            ]
            fragmented_total_rows = [
                benchmark_direct_pageable_fragmented(device, int(args.slice_bytes), int(slice_count), int(args.warmup_iters), int(args.measure_iters))
                for slice_count in fragmented_slice_grid
            ]

        transfer_modes[TRANSFER_MODE_DIRECT_PINNED_PACKED]["profiles"][profile] = {
            "packed_h2d_rows": pinned_h2d_rows,
            "packed_launch_rows": packed_launch_rows,
        }
        transfer_modes[TRANSFER_MODE_STAGED_PAGEABLE_PACKED]["profiles"][profile] = {
            "packed_gather_rows": packed_gather_rows,
            "packed_h2d_rows": pinned_h2d_rows,
            "packed_launch_rows": packed_launch_rows,
        }
        transfer_modes[TRANSFER_MODE_DIRECT_PAGEABLE_FRAGMENTED]["profiles"][profile] = {
            "fragmented_total_rows": fragmented_total_rows,
        }

    lora_compute_ms = benchmark_lora_compute_rows(
        device=device,
        hidden_size=int(text_config["hidden_size"]),
        moe_intermediate_size=int(text_config["moe_intermediate_size"]),
        lora_rank=int(args.lora_rank),
        num_experts_per_tok=int(text_config["num_experts_per_tok"]),
        dtype=dtype,
        batch_grid=batch_grid,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
    )
    overlap_windows_ms = benchmark_overlap_windows(
        device=device,
        hidden_size=int(text_config["hidden_size"]),
        moe_intermediate_size=int(text_config["moe_intermediate_size"]),
        num_experts=int(text_config["num_experts"]),
        num_experts_per_tok=int(text_config["num_experts_per_tok"]),
        dtype=dtype,
        batch_grid=batch_grid,
        warmup_iters=int(args.warmup_iters),
        measure_iters=int(args.measure_iters),
    )

    manifest = {
        "model_name": "case_study_system_baseline_calibration_v1",
        "device": {
            "requested_device": str(args.device),
            "resolved_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "measurement": {
            "warmup_iters": int(args.warmup_iters),
            "measure_iters": int(args.measure_iters),
            "host_profiles": list(host_profiles),
            "stress_workers": int(args.stress_workers),
            "stress_array_mb": int(args.stress_array_mb),
        },
        "object_sizes": {
            "hidden_size": int(text_config["hidden_size"]),
            "moe_intermediate_size": int(text_config["moe_intermediate_size"]),
            "num_hidden_layers": int(text_config["num_hidden_layers"]),
            "num_experts": int(text_config["num_experts"]),
            "num_experts_per_tok": int(text_config["num_experts_per_tok"]),
            "dtype": str(args.dtype),
            "lora_rank": int(args.lora_rank),
            "slice_bytes": int(args.slice_bytes),
            "early_component_label": object_sizes.early_component_label,
            "late_component_label": object_sizes.late_component_label,
            "early_object_bytes": int(object_sizes.early_object_bytes),
            "late_object_bytes": int(object_sizes.late_object_bytes),
            "total_object_bytes": int(object_sizes.total_object_bytes),
            "early_object_slices": int(object_sizes.early_object_slices),
            "late_object_slices": int(object_sizes.late_object_slices),
            "total_object_slices": int(object_sizes.total_object_slices),
            "gate_lora_excluded": bool(object_sizes.gate_lora_excluded),
        },
        "assumptions": {
            "host_pool_default": "pageable_with_pinned_staging",
            "pageable_and_pinned_benchmarked": True,
            "packed_scope": "per-layer causal packing only",
            "base_tpot_source": "user_supplied_hot_cache_measurement",
        },
        "base_tpot_ms": {str(key): float(value) for key, value in base_tpot_ms.items()},
        "transfer_modes": transfer_modes,
        "lora_compute_ms": lora_compute_ms,
        "overlap_windows_ms": overlap_windows_ms,
    }
    write_json(output_path, manifest)
    print(f"wrote calibration manifest: {output_path}")


if __name__ == "__main__":
    main()
