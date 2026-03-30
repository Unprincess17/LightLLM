#!/usr/bin/env python3
"""B11: small live validation for the case-study pipeline."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from collect_router_trace import print_log_tail, stream_subprocess_output, terminate_process, wait_for_server_ready
from common import (
    ensure_parent_dir,
    iter_jsonl,
    load_global_config,
    numeric_summary,
    parse_cardinalities,
    stage_output_dir,
    write_csv,
    write_json,
    write_jsonl,
)
from replay_fixed_requests import build_request_body, load_fixed_requests


DEFAULT_STAGE = "live_validation"
DEFAULT_SELECTION_STRATEGY = "auto_reuse_burst"
DEFAULT_MAPPING_MODE = "corr"
DEFAULT_METRIC_NAMES = (
    "lightllm_request_count_total",
    "lightllm_request_success_total",
    "lightllm_request_failure_total",
    "lightllm_request_duration_count",
    "lightllm_request_duration_sum",
    "lightllm_queue_size",
)

ADAPTER_ID_PATTERN = re.compile(r"^lora_(\d+)$")
PROM_METRIC_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|NaN|Inf|-Inf))\s*$"
)
COLORA_KV_PATTERN = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>[^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a narrow live LoRA replay sanity validation")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override live-validation stage directory")
    parser.add_argument("--requests_path", type=str, default=None, help="Override fixed_requests.jsonl path")
    parser.add_argument("--adapter_trace_path", type=str, default=None, help="Override mapped adapter trace path")
    parser.add_argument(
        "--mapping_mode",
        type=str,
        default=DEFAULT_MAPPING_MODE,
        choices=("corr", "indep"),
        help="Which mapped adapter stream to replay",
    )
    parser.add_argument(
        "--adapter_cardinality",
        type=int,
        default=None,
        help="Mapped adapter cardinality to use. Default: largest available cardinality not exceeding loaded live adapters",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default=DEFAULT_SELECTION_STRATEGY,
        choices=("auto_reuse_burst", "first_n", "explicit_indices"),
        help="How to derive the live request subset",
    )
    parser.add_argument(
        "--request_indices",
        type=str,
        default=None,
        help="Comma-separated aligned request indices when --selection_strategy=explicit_indices",
    )
    parser.add_argument("--request_count", type=int, default=4, help="Number of requests to replay live")
    parser.add_argument(
        "--search_limit",
        type=int,
        default=512,
        help="How many aligned requests to scan when auto-selecting a reuse-heavy window",
    )
    parser.add_argument(
        "--max_decode_tokens",
        type=int,
        default=16,
        help="Cap completion tokens per request to keep the live validation narrow",
    )
    parser.add_argument("--server_url", type=str, default=None, help="Base server URL")
    parser.add_argument("--health_path", type=str, default=None, help="Health endpoint path")
    parser.add_argument("--model_name", type=str, default=None, help="Model name passed to the API")
    parser.add_argument("--request_timeout_s", type=float, default=None, help="Per-request timeout")
    parser.add_argument("--startup_timeout_s", type=float, default=None, help="Server startup timeout")
    parser.add_argument("--server_log_path", type=str, default=None, help="Path to the live server log")
    parser.add_argument(
        "--server_log_settle_s",
        type=float,
        default=0.35,
        help="Sleep after each request before reading new server-log bytes",
    )
    parser.add_argument(
        "--server_log_tail_lines",
        type=int,
        default=80,
        help="How many server-log lines to print if the live run fails",
    )
    parser.add_argument("--reuse_server", action="store_true", help="Use an already-running server")
    parser.add_argument(
        "--tee_server_output",
        action="store_true",
        default=True,
        help="Mirror launched server logs to the terminal while still writing the log file",
    )
    parser.add_argument(
        "--no_tee_server_output",
        dest="tee_server_output",
        action="store_false",
        help="Do not mirror launched server logs to stdout; still write the log file",
    )
    return parser.parse_args()


def log_step(message: str) -> None:
    print(f"[live_validation] {message}", flush=True)


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_request_indices(raw_value: Optional[str]) -> List[int]:
    if raw_value is None:
        return []
    indices: List[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        indices.append(int(token))
    return indices


def read_jsonl_rows(path: Path) -> List[dict]:
    return list(iter_jsonl(path))


def load_adapter_trace_rows(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows = []
    for line_idx, record in enumerate(iter_jsonl(path)):
        row = dict(record)
        row["_line_idx"] = line_idx
        rows.append(row)
    rows.sort(key=lambda row: (int(row.get("arrival_idx", row["_line_idx"])), int(row["_line_idx"])))
    if limit is not None:
        return rows[:limit]
    return rows


def select_default_adapter_trace_path(
    config: Mapping[str, object],
    run_id: Optional[str],
    mapping_mode: str,
    adapter_cardinality: Optional[int],
    live_adapter_capacity: int,
) -> Tuple[Path, int]:
    adapter_trace_dir = stage_output_dir("adapter_trace", config, run_id)
    if adapter_cardinality is not None:
        candidate = adapter_trace_dir / f"adapter_trace_mapped_{mapping_mode}_c{adapter_cardinality:03d}.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"mapped adapter trace not found: {candidate}")
        return candidate, adapter_cardinality

    configured_cards = parse_cardinalities(config.get("case_study", {}).get("adapter_cardinalities"))
    candidate_cards = sorted({card for card in configured_cards if card <= live_adapter_capacity}, reverse=True)
    for card in candidate_cards:
        candidate = adapter_trace_dir / f"adapter_trace_mapped_{mapping_mode}_c{card:03d}.jsonl"
        if candidate.exists():
            return candidate, card

    raise FileNotFoundError(
        "could not infer a mapped adapter trace compatible with the live dummy-LoRA count; "
        f"checked cardinalities <= {live_adapter_capacity} under {adapter_trace_dir}"
    )


def build_live_adapter_alias_map(dummy_lora_dirs: Sequence[str]) -> Dict[str, dict]:
    alias_map: Dict[str, dict] = {}
    for adapter_slot, adapter_dir in enumerate(dummy_lora_dirs):
        resolved_dir = str(Path(adapter_dir).resolve())
        live_adapter_name = Path(resolved_dir).name
        alias_map[f"lora_{adapter_slot}"] = {
            "adapter_slot": adapter_slot,
            "live_adapter_name": live_adapter_name,
            "adapter_dir": resolved_dir,
            "live_adapter_id": str(adapter_slot + 1),
        }
    return alias_map


def _window_score(
    request_rows: Sequence[dict],
    adapter_rows: Sequence[dict],
    start_index: int,
    request_count: int,
    max_decode_tokens: int,
) -> dict:
    adapters = [str(adapter_rows[start_index + offset].get("adapter_id", "")) for offset in range(request_count)]
    counts = Counter(adapters)
    repeated_adapter_count = sum(1 for count in counts.values() if count > 1)
    repeat_request_count = sum(count for count in counts.values() if count > 1)
    immediate_reuse_pairs = sum(1 for offset in range(1, request_count) if adapters[offset] == adapters[offset - 1])
    token_cost = 0
    prompt_cost = 0
    for offset in range(request_count):
        request_row = request_rows[start_index + offset]
        prompt_len = int(request_row["prompt_len_tokens"])
        decode_len = int(request_row["target_decode_len"])
        capped_decode = max(1, min(decode_len, max_decode_tokens))
        prompt_cost += prompt_len
        token_cost += prompt_len + capped_decode
    return {
        "start_index": start_index,
        "token_cost": token_cost,
        "prompt_cost": prompt_cost,
        "repeated_adapter_count": repeated_adapter_count,
        "repeat_request_count": repeat_request_count,
        "immediate_reuse_pairs": immediate_reuse_pairs,
        "unique_adapter_count": len(counts),
        "adapter_counts": dict(counts),
    }


def select_auto_reuse_window(
    request_rows: Sequence[dict],
    adapter_rows: Sequence[dict],
    request_count: int,
    search_limit: int,
    max_decode_tokens: int,
) -> dict:
    aligned_count = min(len(request_rows), len(adapter_rows), search_limit)
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if aligned_count < request_count:
        raise ValueError(
            f"cannot select {request_count} requests from only {aligned_count} aligned request/adapter rows"
        )

    candidates = [
        _window_score(request_rows, adapter_rows, start_index, request_count, max_decode_tokens)
        for start_index in range(0, aligned_count - request_count + 1)
    ]

    def _pick(min_repeated_adapters: int, min_immediate_reuse: int) -> Optional[dict]:
        filtered = [
            candidate
            for candidate in candidates
            if candidate["repeated_adapter_count"] >= min_repeated_adapters
            and candidate["immediate_reuse_pairs"] >= min_immediate_reuse
        ]
        if not filtered:
            return None
        filtered.sort(
            key=lambda candidate: (
                -int(candidate["repeated_adapter_count"]),
                int(candidate["token_cost"]),
                -int(candidate["immediate_reuse_pairs"]),
                int(candidate["unique_adapter_count"]),
                int(candidate["start_index"]),
            )
        )
        return filtered[0]

    selected = _pick(min_repeated_adapters=2, min_immediate_reuse=1)
    if selected is None:
        selected = _pick(min_repeated_adapters=1, min_immediate_reuse=1)
    if selected is None:
        candidates.sort(
            key=lambda candidate: (
                int(candidate["token_cost"]),
                -int(candidate["repeated_adapter_count"]),
                -int(candidate["immediate_reuse_pairs"]),
                int(candidate["start_index"]),
            )
        )
        selected = candidates[0]
    return selected


def build_schedule_rows(
    request_rows: Sequence[dict],
    adapter_rows: Sequence[dict],
    selected_indices: Sequence[int],
    alias_map: Mapping[str, dict],
    requests_path: Path,
    adapter_trace_path: Path,
    adapter_cardinality: int,
    selection_strategy: str,
    selection_metadata: Mapping[str, object],
    max_decode_tokens: int,
) -> List[dict]:
    schedule_rows: List[dict] = []
    for submit_order, aligned_index in enumerate(selected_indices):
        if aligned_index < 0 or aligned_index >= len(request_rows) or aligned_index >= len(adapter_rows):
            raise IndexError(f"aligned request index out of range: {aligned_index}")
        request_row = dict(request_rows[aligned_index])
        adapter_row = dict(adapter_rows[aligned_index])
        offline_adapter_id = str(adapter_row.get("adapter_id", ""))
        alias_entry = alias_map.get(offline_adapter_id)
        if alias_entry is None:
            raise KeyError(f"no live dummy-LoRA mapping found for adapter {offline_adapter_id!r}")
        selected_decode_len = max(1, min(int(request_row["target_decode_len"]), max_decode_tokens))
        schedule_rows.append(
            {
                "submit_order": submit_order,
                "aligned_request_index": aligned_index,
                "req_idx": int(request_row["req_idx"]),
                "adapter_trace_arrival_idx": int(adapter_row.get("arrival_idx", aligned_index)),
                "adapter_id": offline_adapter_id,
                "live_adapter_name": str(alias_entry["live_adapter_name"]),
                "live_adapter_id": str(alias_entry["live_adapter_id"]),
                "live_adapter_dir": str(alias_entry["adapter_dir"]),
                "adapter_slot": int(alias_entry["adapter_slot"]),
                "selection_strategy": selection_strategy,
                "selection_metadata": dict(selection_metadata),
                "requests_path": str(requests_path),
                "adapter_trace_path": str(adapter_trace_path),
                "adapter_cardinality": int(adapter_cardinality),
                "prompt_len_tokens": int(request_row["prompt_len_tokens"]),
                "target_decode_len_source": int(request_row["target_decode_len"]),
                "max_tokens_live": selected_decode_len,
                "source_id": request_row.get("source_id"),
                "split_tag": request_row.get("split_tag"),
                "render_mode": request_row.get("render_mode"),
                "messages": request_row.get("messages") or [{"role": "user", "content": request_row["prompt_text"]}],
            }
        )
    return schedule_rows


def build_live_server_command(
    launcher_path: Path,
    model_dir: str,
    port: int,
    tp: int,
    dummy_lora_dirs: Sequence[str],
) -> List[str]:
    command = [
        "bash",
        str(launcher_path),
        "--model_dir",
        model_dir,
        "--port",
        str(port),
        "--tp",
        str(tp),
        "--lora_dirs",
        ",".join(str(Path(path).resolve()) for path in dummy_lora_dirs),
    ]
    return command


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def read_log_chunk(path: Path, start_offset: int) -> Tuple[int, str]:
    if not path.exists():
        return start_offset, ""
    with path.open("rb") as handle:
        handle.seek(start_offset)
        payload = handle.read()
        end_offset = handle.tell()
    return end_offset, payload.decode("utf-8", errors="replace")


def read_log_window(path: Path, start_offset: int, settle_s: float) -> Tuple[int, str]:
    current_offset = start_offset
    chunks: List[str] = []
    for _ in range(3):
        if settle_s > 0:
            time.sleep(settle_s)
        next_offset, chunk = read_log_chunk(path, current_offset)
        if not chunk:
            break
        chunks.append(chunk)
        current_offset = next_offset
        if "[COLoRA]" in chunk:
            continue
    return current_offset, "".join(chunks)


def parse_numeric_value(raw_value: str) -> object:
    lowered = raw_value.lower()
    if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
        return float(raw_value)
    try:
        if any(marker in raw_value for marker in (".", "e", "E")):
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def parse_colora_log_lines(log_text: str) -> List[dict]:
    rows: List[dict] = []
    for line in log_text.splitlines():
        if "[COLoRA]" not in line:
            continue
        suffix = line.split("[COLoRA]", 1)[1]
        parsed = {"raw_line": line}
        for match in COLORA_KV_PATTERN.finditer(suffix):
            parsed[match.group("key")] = parse_numeric_value(match.group("value"))
        rows.append(parsed)
    return rows


def summarize_colora_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    hit_tokens = sum(int(row.get("hit_tokens", 0)) for row in rows)
    miss_tokens = sum(int(row.get("miss_tokens", 0)) for row in rows)
    total_tokens = hit_tokens + miss_tokens
    observed_hit_rate = float(hit_tokens) / float(total_tokens) if total_tokens else 0.0
    cache_hit_rate_samples = [float(row.get("hit_rate", 0.0)) for row in rows]
    queue_depth_samples = [int(row.get("queue_depth", 0)) for row in rows]
    cpu_queue_depth_samples = [int(row.get("cpu_queue_depth", 0)) for row in rows]
    cache_capacity_samples = [int(row.get("cache_capacity_slots", 0)) for row in rows]
    cache_resident_samples = [int(row.get("cache_resident_slots", 0)) for row in rows]
    cache_free_samples = [int(row.get("cache_free_slots", 0)) for row in rows]
    cache_eviction_samples = [int(row.get("cache_evictions_total", 0)) for row in rows]
    overlap_modes = sorted({str(row.get("overlap_mode", "")) for row in rows if row.get("overlap_mode") not in (None, "")})
    miss_policies = sorted({str(row.get("miss_policy", "")) for row in rows if row.get("miss_policy") not in (None, "")})
    return {
        "colora_line_count": len(rows),
        "colora_hit_tokens": hit_tokens,
        "colora_miss_tokens": miss_tokens,
        "observed_hit_rate": observed_hit_rate,
        "cache_hit_rate_max": max(cache_hit_rate_samples) if cache_hit_rate_samples else 0.0,
        "cache_capacity_slots_max": max(cache_capacity_samples) if cache_capacity_samples else 0,
        "cache_resident_slots_max": max(cache_resident_samples) if cache_resident_samples else 0,
        "cache_free_slots_min": min(cache_free_samples) if cache_free_samples else 0,
        "cache_evictions_total_end": max(cache_eviction_samples) if cache_eviction_samples else 0,
        "promotion_queue_depth_max": max(queue_depth_samples) if queue_depth_samples else 0,
        "cpu_queue_depth_max": max(cpu_queue_depth_samples) if cpu_queue_depth_samples else 0,
        "promotion_drop_total_end": max((int(row.get("promotion_drop_total", 0)) for row in rows), default=0),
        "promotion_drop_queue_high_watermark_end": max(
            (int(row.get("promotion_drop_queue", 0)) for row in rows),
            default=0,
        ),
        "promotion_drop_cooldown_end": max(
            (int(row.get("promotion_drop_cooldown", 0)) for row in rows),
            default=0,
        ),
        "fallback_degrade_count_sum": sum(int(row.get("fallback_degrade_count", 0)) for row in rows),
        "cpu_compute_time_sum": sum(float(row.get("cpu_compute_time", 0.0)) for row in rows),
        "gpu_compute_time_sum": sum(float(row.get("gpu_compute_time", 0.0)) for row in rows),
        "cpu_queue_wait_time_sum": sum(float(row.get("cpu_queue_wait", 0.0)) for row in rows),
        "d2h_bytes_sum": sum(float(row.get("d2h_bytes", 0.0)) for row in rows),
        "h2d_bytes_sum": sum(float(row.get("h2d_bytes", 0.0)) for row in rows),
        "weight_h2d_bytes_sum": sum(float(row.get("weight_h2d_bytes", 0.0)) for row in rows),
        "weight_h2d_time_sum": sum(float(row.get("weight_h2d_time", 0.0)) for row in rows),
        "blocking_promotion_count_sum": sum(int(row.get("blocking_promotion_count", 0)) for row in rows),
        "observed_overlap_modes": overlap_modes,
        "observed_miss_policies": miss_policies,
        "moe_kernel_calls_sum": sum(int(row.get("moe_kernel_calls", 0)) for row in rows),
        "moe_kernel_tokens_sum": sum(int(row.get("moe_kernel_tokens", 0)) for row in rows),
    }


def fetch_available_lora_adapters(server_url: str, timeout_s: float = 5.0) -> dict:
    endpoint = server_url.rstrip("/") + "/v1/lora/adapters"
    try:
        response = requests.get(endpoint, timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            return {"ok": False, "endpoint": endpoint, "error": "unexpected adapter-list payload"}
        adapter_names = []
        for adapter in body.get("adapters", []):
            if isinstance(adapter, dict):
                adapter_names.append(str(adapter.get("id") or adapter.get("adapter_id") or adapter.get("path") or ""))
        return {
            "ok": True,
            "endpoint": endpoint,
            "count": int(body.get("count", len(adapter_names))),
            "adapter_names": [name for name in adapter_names if name],
            "raw": body,
        }
    except Exception as exc:
        return {"ok": False, "endpoint": endpoint, "error": str(exc)}


def fetch_metrics_snapshot(
    server_url: str,
    metric_names: Sequence[str] = DEFAULT_METRIC_NAMES,
    timeout_s: float = 5.0,
) -> Optional[dict]:
    endpoint = server_url.rstrip("/") + "/metrics"
    try:
        response = requests.get(endpoint, timeout=timeout_s)
        response.raise_for_status()
    except Exception:
        return None

    metrics = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_METRIC_PATTERN.match(line.strip())
        if match is None:
            continue
        name = match.group("name")
        if name not in metric_names:
            continue
        metrics[name] = float(match.group("value"))
    return metrics


def diff_metric_snapshots(start: Optional[Mapping[str, float]], end: Optional[Mapping[str, float]]) -> Optional[dict]:
    if start is None or end is None:
        return None
    keys = sorted(set(start) | set(end))
    return {key: float(end.get(key, 0.0)) - float(start.get(key, 0.0)) for key in keys}


def summarize_per_window_counters(per_window_counter_rows: Sequence[Mapping[str, object]]) -> dict:
    total_hits = sum(int(row.get("colora_hit_tokens", 0)) for row in per_window_counter_rows)
    total_misses = sum(int(row.get("colora_miss_tokens", 0)) for row in per_window_counter_rows)
    total_tokens = total_hits + total_misses
    return {
        "colora_line_count": sum(int(row.get("colora_line_count", 0)) for row in per_window_counter_rows),
        "colora_hit_tokens": total_hits,
        "colora_miss_tokens": total_misses,
        "observed_hit_rate": (float(total_hits) / float(total_tokens)) if total_tokens else 0.0,
        "cache_hit_rate_max": max((float(row.get("cache_hit_rate_max", 0.0)) for row in per_window_counter_rows), default=0.0),
        "promotion_queue_depth_max": max(
            (int(row.get("promotion_queue_depth_max", 0)) for row in per_window_counter_rows),
            default=0,
        ),
        "cpu_queue_depth_max": max((int(row.get("cpu_queue_depth_max", 0)) for row in per_window_counter_rows), default=0),
        "promotion_drop_total_end": max(
            (int(row.get("promotion_drop_total_end", 0)) for row in per_window_counter_rows),
            default=0,
        ),
        "promotion_drop_queue_high_watermark_end": max(
            (int(row.get("promotion_drop_queue_high_watermark_end", 0)) for row in per_window_counter_rows),
            default=0,
        ),
        "promotion_drop_cooldown_end": max(
            (int(row.get("promotion_drop_cooldown_end", 0)) for row in per_window_counter_rows),
            default=0,
        ),
        "fallback_degrade_count_sum": sum(
            int(row.get("fallback_degrade_count_sum", 0)) for row in per_window_counter_rows
        ),
        "cpu_compute_time_sum": sum(float(row.get("cpu_compute_time_sum", 0.0)) for row in per_window_counter_rows),
        "gpu_compute_time_sum": sum(float(row.get("gpu_compute_time_sum", 0.0)) for row in per_window_counter_rows),
        "cpu_queue_wait_time_sum": sum(
            float(row.get("cpu_queue_wait_time_sum", 0.0)) for row in per_window_counter_rows
        ),
        "d2h_bytes_sum": sum(float(row.get("d2h_bytes_sum", 0.0)) for row in per_window_counter_rows),
        "h2d_bytes_sum": sum(float(row.get("h2d_bytes_sum", 0.0)) for row in per_window_counter_rows),
        "weight_h2d_bytes_sum": sum(float(row.get("weight_h2d_bytes_sum", 0.0)) for row in per_window_counter_rows),
        "weight_h2d_time_sum": sum(float(row.get("weight_h2d_time_sum", 0.0)) for row in per_window_counter_rows),
        "blocking_promotion_count_sum": sum(
            int(row.get("blocking_promotion_count_sum", 0)) for row in per_window_counter_rows
        ),
        "observed_overlap_modes": sorted(
            {mode for row in per_window_counter_rows for mode in row.get("observed_overlap_modes", [])}
        ),
        "observed_miss_policies": sorted(
            {policy for row in per_window_counter_rows for policy in row.get("observed_miss_policies", [])}
        ),
        "moe_kernel_calls_sum": sum(int(row.get("moe_kernel_calls_sum", 0)) for row in per_window_counter_rows),
        "moe_kernel_tokens_sum": sum(int(row.get("moe_kernel_tokens_sum", 0)) for row in per_window_counter_rows),
    }


def send_live_requests(
    schedule_rows: Sequence[Mapping[str, object]],
    server_url: str,
    model_name: str,
    request_timeout_s: float,
    server_log_path: Optional[Path],
    server_log_settle_s: float,
) -> Tuple[List[dict], List[dict], List[dict]]:
    session = requests.Session()
    latency_rows: List[dict] = []
    request_result_rows: List[dict] = []
    per_window_counter_rows: List[dict] = []

    log_offset = 0
    if server_log_path is not None and server_log_path.exists():
        log_offset = server_log_path.stat().st_size

    for schedule_row in schedule_rows:
        submit_order = int(schedule_row["submit_order"])
        request_row = {
            "req_idx": int(schedule_row["req_idx"]),
            "messages": schedule_row["messages"],
            "prompt_len_tokens": int(schedule_row["prompt_len_tokens"]),
            "target_decode_len": int(schedule_row["max_tokens_live"]),
            "prompt_text": "",
        }
        payload = build_request_body(
            request_row=request_row,
            model_name=model_name,
            adapter_id=str(schedule_row["live_adapter_name"]),
            ignore_eos=False,
            respect_request_decode_len=True,
        )

        start_wall = now_iso()
        start_perf = time.perf_counter()
        status = "ok"
        status_code: Optional[int] = None
        error_text: Optional[str] = None
        response_id = None
        finish_reason = None
        completion_chars = 0

        try:
            response = session.post(
                server_url.rstrip("/") + "/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=request_timeout_s,
            )
            status_code = response.status_code
            if response.ok:
                body = response.json()
                response_id = body.get("id")
                choices = body.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                    message = choices[0].get("message") or {}
                    completion_chars = len(message.get("content") or "")
            else:
                status = "http_error"
                error_text = response.text
        except Exception as exc:
            status = "exception"
            error_text = str(exc)

        end_perf = time.perf_counter()
        end_wall = now_iso()
        latency_ms = (end_perf - start_perf) * 1000.0

        window_log_text = ""
        window_end_offset = log_offset
        parsed_colora_rows: List[dict] = []
        if server_log_path is not None:
            window_end_offset, window_log_text = read_log_window(server_log_path, log_offset, server_log_settle_s)
            parsed_colora_rows = parse_colora_log_lines(window_log_text)

        counter_summary = summarize_colora_rows(parsed_colora_rows)
        counter_row = {
            "submit_order": submit_order,
            "req_idx": int(schedule_row["req_idx"]),
            "adapter_id": str(schedule_row["adapter_id"]),
            "live_adapter_name": str(schedule_row["live_adapter_name"]),
            "server_log_start_offset": log_offset,
            "server_log_end_offset": window_end_offset,
        }
        counter_row.update(counter_summary)
        per_window_counter_rows.append(counter_row)
        log_offset = window_end_offset

        latency_rows.append(
            {
                "req_idx": int(schedule_row["req_idx"]),
                "adapter_id": str(schedule_row["adapter_id"]),
                "submit_order": submit_order,
                "start_time": start_wall,
                "end_time": end_wall,
                "latency_ms": latency_ms,
                "status": status if status_code is None else ("ok" if status_code == 200 and status == "ok" else status),
                "http_status": "" if status_code is None else int(status_code),
                "response_id": "" if response_id is None else str(response_id),
                "completion_chars": completion_chars,
                "error": "" if error_text is None else error_text,
            }
        )
        request_result_rows.append(
            {
                "submit_order": submit_order,
                "req_idx": int(schedule_row["req_idx"]),
                "adapter_id": str(schedule_row["adapter_id"]),
                "live_adapter_name": str(schedule_row["live_adapter_name"]),
                "status_code": status_code,
                "status": status if status_code is None else ("ok" if status_code == 200 and status == "ok" else status),
                "latency_ms": latency_ms,
                "response_id": response_id,
                "finish_reason": finish_reason,
                "completion_chars": completion_chars,
                "error": error_text,
                "colora_line_count": counter_summary["colora_line_count"],
                "colora_hit_tokens": counter_summary["colora_hit_tokens"],
                "colora_miss_tokens": counter_summary["colora_miss_tokens"],
                "observed_hit_rate": counter_summary["observed_hit_rate"],
            }
        )
        log_step(
            f"completed live request {submit_order + 1}/{len(schedule_rows)} "
            f"(req_idx={schedule_row['req_idx']}, adapter={schedule_row['adapter_id']}, status={latency_rows[-1]['status']})"
        )

    return request_result_rows, latency_rows, per_window_counter_rows


def summarize_reuse_agreement(
    schedule_rows: Sequence[Mapping[str, object]],
    latency_rows: Sequence[Mapping[str, object]],
    per_window_counter_rows: Sequence[Mapping[str, object]],
) -> dict:
    latency_by_submit = {int(row["submit_order"]): row for row in latency_rows}
    counters_by_submit = {int(row["submit_order"]): row for row in per_window_counter_rows}
    groups: Dict[str, List[dict]] = defaultdict(list)
    for schedule_row in schedule_rows:
        submit_order = int(schedule_row["submit_order"])
        groups[str(schedule_row["adapter_id"])].append(
            {
                "submit_order": submit_order,
                "req_idx": int(schedule_row["req_idx"]),
                "latency_ms": float(latency_by_submit.get(submit_order, {}).get("latency_ms", 0.0)),
                "observed_hit_rate": float(counters_by_submit.get(submit_order, {}).get("observed_hit_rate", 0.0)),
                "colora_hit_tokens": int(counters_by_submit.get(submit_order, {}).get("colora_hit_tokens", 0)),
                "colora_miss_tokens": int(counters_by_submit.get(submit_order, {}).get("colora_miss_tokens", 0)),
            }
        )

    comparisons = []
    improved_count = 0
    comparable_count = 0
    for adapter_id, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        comparable_count += 1
        baseline = rows[0]
        followups = rows[1:]
        best_followup = max(
            followups,
            key=lambda row: (float(row["observed_hit_rate"]), int(row["colora_hit_tokens"]), -float(row["latency_ms"])),
        )
        improved = bool(
            float(best_followup["observed_hit_rate"]) > float(baseline["observed_hit_rate"])
            or int(best_followup["colora_hit_tokens"]) > int(baseline["colora_hit_tokens"])
        )
        improved_count += int(improved)
        comparisons.append(
            {
                "adapter_id": adapter_id,
                "baseline_submit_order": int(baseline["submit_order"]),
                "baseline_req_idx": int(baseline["req_idx"]),
                "baseline_hit_rate": float(baseline["observed_hit_rate"]),
                "baseline_hit_tokens": int(baseline["colora_hit_tokens"]),
                "followup_submit_order": int(best_followup["submit_order"]),
                "followup_req_idx": int(best_followup["req_idx"]),
                "followup_hit_rate": float(best_followup["observed_hit_rate"]),
                "followup_hit_tokens": int(best_followup["colora_hit_tokens"]),
                "improved_hit_signal": improved,
            }
        )

    if comparable_count == 0:
        verdict = "inconclusive"
        rationale = "no adapter was replayed more than once in the selected live window"
    elif improved_count > 0:
        verdict = "agree"
        rationale = f"{improved_count}/{comparable_count} repeated adapters showed a stronger hit signal on a later reuse"
    else:
        verdict = "disagree"
        rationale = "repeated adapters were observed, but later reuses did not show a stronger COLoRA hit signal"

    return {
        "verdict": verdict,
        "rationale": rationale,
        "comparisons": comparisons,
        "comparable_adapter_count": comparable_count,
        "improved_adapter_count": improved_count,
    }


def build_summary_markdown(
    schedule_rows: Sequence[Mapping[str, object]],
    latency_rows: Sequence[Mapping[str, object]],
    per_window_counter_rows: Sequence[Mapping[str, object]],
    counter_aggregate: Mapping[str, object],
    reuse_summary: Mapping[str, object],
    server_url: str,
    adapter_trace_path: Path,
    available_adapters: Mapping[str, object],
    metrics_start: Optional[Mapping[str, float]],
    metrics_end: Optional[Mapping[str, float]],
    metrics_delta: Optional[Mapping[str, float]],
    launched_server: bool,
) -> str:
    successful_latencies = [float(row["latency_ms"]) for row in latency_rows if str(row["status"]) == "ok"]
    latency_summary = numeric_summary(successful_latencies)
    request_orders = ", ".join(str(int(row["req_idx"])) for row in schedule_rows)
    adapter_schedule = ", ".join(str(row["adapter_id"]) for row in schedule_rows)
    live_adapter_schedule = ", ".join(str(row["live_adapter_name"]) for row in schedule_rows)
    aggregate_hits = int(counter_aggregate.get("colora_hit_tokens", 0))
    aggregate_misses = int(counter_aggregate.get("colora_miss_tokens", 0))
    aggregate_hit_rate = float(counter_aggregate.get("observed_hit_rate", 0.0))

    lines = [
        "# Small Live Validation",
        "",
        "## Setup",
        f"- Server: `{server_url}` ({'launched by this script' if launched_server else 'reused existing server'})",
        f"- Request subset: `{len(schedule_rows)}` sequential requests derived from `{schedule_rows[0]['requests_path']}` with aligned adapter replay from `{adapter_trace_path}`",
        f"- Selected request order (`req_idx`): `{request_orders}`",
        f"- Adapter replay schedule: `{adapter_schedule}`",
        f"- Live adapter names sent to the server: `{live_adapter_schedule}`",
        f"- Live decode cap: `{schedule_rows[0]['max_tokens_live']}` tokens per request",
        "",
        "## Observations",
        f"- Successful requests: `{sum(1 for row in latency_rows if str(row['status']) == 'ok')}/{len(latency_rows)}`",
        (
            f"- Latency summary (successful only): mean `{latency_summary['mean']:.2f}` ms, "
            f"p50 `{latency_summary['p50']:.2f}` ms, p95 `{latency_summary['p95']:.2f}` ms, "
            f"max `{latency_summary['max']:.2f}` ms"
        ),
        (
            f"- COLoRA counters captured from server logs: hit tokens `{aggregate_hits}`, "
            f"miss tokens `{aggregate_misses}`, observed hit rate `{aggregate_hit_rate:.4f}`, "
            f"log windows with COLoRA lines `{sum(1 for row in per_window_counter_rows if int(row['colora_line_count']) > 0)}`"
        ),
    ]

    if available_adapters.get("ok"):
        lines.append(f"- LoRA adapters reported by `/v1/lora/adapters`: `{available_adapters.get('count', 0)}`")
    if metrics_delta:
        request_delta = metrics_delta.get("lightllm_request_count_total")
        success_delta = metrics_delta.get("lightllm_request_success_total")
        failure_delta = metrics_delta.get("lightllm_request_failure_total")
        if request_delta is not None:
            lines.append(
                f"- HTTP metric deltas: request_count `{request_delta:.0f}`, success `{(success_delta or 0.0):.0f}`, failure `{(failure_delta or 0.0):.0f}`"
            )

    lines.extend(
        [
            "",
            "## Qualitative Agreement",
            "- Offline replay predicts a cold-first, warmer-on-reuse mechanism once the same adapter path reappears.",
            f"- Live verdict: `{reuse_summary['verdict']}`. {reuse_summary['rationale']}.",
        ]
    )
    for comparison in reuse_summary.get("comparisons", []):
        lines.append(
            "- "
            f"{comparison['adapter_id']}: first hit_rate `{comparison['baseline_hit_rate']:.4f}` "
            f"(hit_tokens `{comparison['baseline_hit_tokens']}`) at submit `{comparison['baseline_submit_order']}`, "
            f"best later hit_rate `{comparison['followup_hit_rate']:.4f}` "
            f"(hit_tokens `{comparison['followup_hit_tokens']}`) at submit `{comparison['followup_submit_order']}`"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "- This is a narrow sanity check only, not the primary evidence of the paper.",
            "- The run is sequential and tiny by design, so it does not measure throughput or concurrency behavior.",
            "- The replay uses dummy LoRAs and a live decode cap to keep the validation robust and short.",
            "- Prompt lengths still vary across requests, so raw latency should be interpreted together with the captured COLoRA counters rather than by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    case_config = config.get("case_study", {})
    router_config = case_config.get("router_trace", {})
    paths_config = config.get("paths", {})

    output_dir = stage_output_dir(DEFAULT_STAGE, config, args.run_id, args.output_dir)
    requests_path = Path(args.requests_path or (stage_output_dir("prompt_corpus", config, args.run_id) / "fixed_requests.jsonl"))
    dummy_lora_dirs = [str(path) for path in paths_config.get("dummy_lora_dirs", [])]
    if not dummy_lora_dirs:
        raise ValueError("paths.dummy_lora_dirs is empty; live validation needs the loaded dummy LoRAs")

    if args.adapter_trace_path:
        adapter_trace_path = Path(args.adapter_trace_path)
        adapter_cardinality = int(args.adapter_cardinality or 0)
    else:
        adapter_trace_path, adapter_cardinality = select_default_adapter_trace_path(
            config=config,
            run_id=args.run_id,
            mapping_mode=args.mapping_mode,
            adapter_cardinality=args.adapter_cardinality,
            live_adapter_capacity=len(dummy_lora_dirs),
        )

    request_rows = load_fixed_requests(requests_path)
    adapter_rows = load_adapter_trace_rows(adapter_trace_path, limit=len(request_rows))
    aligned_count = min(len(request_rows), len(adapter_rows))
    if aligned_count < args.request_count:
        raise ValueError(
            f"only {aligned_count} aligned request/adapter rows are available, cannot replay {args.request_count} requests"
        )

    if args.selection_strategy == "explicit_indices":
        selected_indices = parse_request_indices(args.request_indices)
        if not selected_indices:
            raise ValueError("--request_indices must be set when --selection_strategy=explicit_indices")
        selection_metadata = {"explicit_indices": selected_indices}
    elif args.selection_strategy == "first_n":
        selected_indices = list(range(args.request_count))
        selection_metadata = {"start_index": 0, "token_cost": None}
    else:
        auto_window = select_auto_reuse_window(
            request_rows=request_rows,
            adapter_rows=adapter_rows,
            request_count=args.request_count,
            search_limit=args.search_limit,
            max_decode_tokens=args.max_decode_tokens,
        )
        start_index = int(auto_window["start_index"])
        selected_indices = list(range(start_index, start_index + args.request_count))
        selection_metadata = dict(auto_window)

    alias_map = build_live_adapter_alias_map(dummy_lora_dirs)
    schedule_rows = build_schedule_rows(
        request_rows=request_rows,
        adapter_rows=adapter_rows,
        selected_indices=selected_indices,
        alias_map=alias_map,
        requests_path=requests_path,
        adapter_trace_path=adapter_trace_path,
        adapter_cardinality=adapter_cardinality,
        selection_strategy=args.selection_strategy,
        selection_metadata=selection_metadata,
        max_decode_tokens=args.max_decode_tokens,
    )

    live_requests_path = output_dir / "live_requests.jsonl"
    live_latency_path = output_dir / "live_latency.csv"
    live_counters_path = output_dir / "live_counters.json"
    live_summary_path = output_dir / "live_summary.md"
    server_log_path = Path(args.server_log_path) if args.server_log_path else output_dir / "live_server.log"
    ensure_parent_dir(live_summary_path)
    write_jsonl(live_requests_path, schedule_rows)

    server_url = args.server_url or str(router_config.get("host", "http://127.0.0.1:8040"))
    port = int(router_config.get("port", 8040))
    health_path = args.health_path or str(router_config.get("health_path", "/healthz"))
    model_name = args.model_name or str(case_config.get("model_name", "Qwen3-VL-30B-A3B-Instruct"))
    request_timeout_s = float(args.request_timeout_s or router_config.get("request_timeout_s", 180))
    startup_timeout_s = float(args.startup_timeout_s or router_config.get("startup_timeout_s", 900))
    tp = int(router_config.get("tp", 2))
    model_dir = str(paths_config["model_dir"])

    launch_command: Optional[List[str]] = None
    server_process: Optional[subprocess.Popen] = None
    server_log_handle = None
    server_output_thread: Optional[threading.Thread] = None
    launched_server = False

    metrics_start = None
    metrics_end = None
    metrics_delta = None

    try:
        log_step(f"selected {len(schedule_rows)} requests from {requests_path}")
        log_step("live request order: " + ", ".join(f"{row['req_idx']}:{row['adapter_id']}" for row in schedule_rows))

        if not args.reuse_server:
            launcher_path = Path(paths_config.get("router_server_launcher", "test/lora/start_server.sh"))
            if not launcher_path.is_absolute():
                launcher_path = Path(__file__).resolve().parents[2] / launcher_path
            launch_command = build_live_server_command(
                launcher_path=launcher_path,
                model_dir=model_dir,
                port=port,
                tp=tp,
                dummy_lora_dirs=dummy_lora_dirs,
            )
            launched_server = True
            log_step(f"launching live server with command: {format_command(launch_command)}")
            server_log_handle = server_log_path.open("w", encoding="utf-8", buffering=1)
            server_process = subprocess.Popen(
                launch_command,
                stdout=subprocess.PIPE if args.tee_server_output else server_log_handle,
                stderr=subprocess.STDOUT,
                text=True if args.tee_server_output else False,
                bufsize=1 if args.tee_server_output else -1,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            if args.tee_server_output:
                server_output_thread = stream_subprocess_output(
                    server_process,
                    log_handle=server_log_handle,
                    mirror_to_stdout=True,
                )
        else:
            log_step(f"reusing existing live server at {server_url}")

        wait_for_server_ready(
            server_url=server_url,
            health_path=health_path,
            timeout_s=startup_timeout_s,
            server_process=server_process,
        )
        log_step(f"server is healthy at {server_url.rstrip('/') + health_path}")

        available_adapters = fetch_available_lora_adapters(server_url)
        if available_adapters.get("ok"):
            available_adapter_names = set(str(name) for name in available_adapters.get("adapter_names", []))
            scheduled_live_names = {str(row["live_adapter_name"]) for row in schedule_rows}
            missing_live_names = sorted(scheduled_live_names - available_adapter_names)
            available_adapters["missing_schedule_adapters"] = missing_live_names
            if missing_live_names:
                raise RuntimeError(
                    "live server did not report the adapters required by the replay schedule: "
                    + ", ".join(missing_live_names)
                )
        metrics_start = fetch_metrics_snapshot(server_url)

        request_result_rows, latency_rows, per_window_counter_rows = send_live_requests(
            schedule_rows=schedule_rows,
            server_url=server_url,
            model_name=model_name,
            request_timeout_s=request_timeout_s,
            server_log_path=server_log_path if server_log_path.exists() or launched_server else None,
            server_log_settle_s=max(float(args.server_log_settle_s), 0.0),
        )
        metrics_end = fetch_metrics_snapshot(server_url)
        metrics_delta = diff_metric_snapshots(metrics_start, metrics_end)

        write_csv(
            live_latency_path,
            [
                "req_idx",
                "adapter_id",
                "submit_order",
                "start_time",
                "end_time",
                "latency_ms",
                "status",
                "http_status",
                "response_id",
                "completion_chars",
                "error",
            ],
            latency_rows,
        )

        counter_aggregate = summarize_per_window_counters(per_window_counter_rows)
        counter_aggregate["windows_with_colora_logs"] = sum(1 for row in per_window_counter_rows if int(row["colora_line_count"]) > 0)
        counter_aggregate["request_count"] = len(per_window_counter_rows)

        live_counters_payload = {
            "source": {
                "server_log_path": str(server_log_path),
                "counter_origin": "[COLoRA] debug log lines captured by per-request server-log byte windows",
                "metrics_path": server_url.rstrip("/") + "/metrics",
                "adapter_list_path": server_url.rstrip("/") + "/v1/lora/adapters",
            },
            "aggregate": counter_aggregate,
            "per_window": per_window_counter_rows,
            "available_adapters": available_adapters,
            "metrics_start": metrics_start,
            "metrics_end": metrics_end,
            "metrics_delta": metrics_delta,
            "request_results": request_result_rows,
            "launch_command": launch_command,
        }
        write_json(live_counters_path, live_counters_payload)

        reuse_summary = summarize_reuse_agreement(
            schedule_rows=schedule_rows,
            latency_rows=latency_rows,
            per_window_counter_rows=per_window_counter_rows,
        )
        live_summary = build_summary_markdown(
            schedule_rows=schedule_rows,
            latency_rows=latency_rows,
            per_window_counter_rows=per_window_counter_rows,
            counter_aggregate=counter_aggregate,
            reuse_summary=reuse_summary,
            server_url=server_url,
            adapter_trace_path=adapter_trace_path,
            available_adapters=available_adapters,
            metrics_start=metrics_start,
            metrics_end=metrics_end,
            metrics_delta=metrics_delta,
            launched_server=launched_server,
        )
        live_summary_path.write_text(live_summary, encoding="utf-8")
        log_step(f"wrote live-validation artifacts under {output_dir}")
    except Exception:
        print_log_tail(server_log_path, args.server_log_tail_lines)
        raise
    finally:
        if server_log_handle is not None:
            server_log_handle.flush()
        if server_process is not None:
            terminate_process(server_process)
        if server_output_thread is not None:
            server_output_thread.join(timeout=2.0)
        if server_log_handle is not None:
            server_log_handle.close()


if __name__ == "__main__":
    main()
