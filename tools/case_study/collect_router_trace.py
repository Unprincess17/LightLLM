#!/usr/bin/env python3
"""Collect and canonicalize a real router trace from the local server."""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import requests

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import iter_jsonl, load_global_config, numeric_summary, stage_output_dir, write_csv, write_json, write_jsonl
from replay_fixed_requests import load_fixed_requests, send_fixed_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a canonical router trace")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override router-trace stage directory")
    parser.add_argument("--requests_path", type=str, default=None, help="Path to fixed_requests.jsonl")
    parser.add_argument("--server_url", type=str, default=None, help="Base server URL")
    parser.add_argument("--health_path", type=str, default=None, help="Health endpoint path")
    parser.add_argument("--model_name", type=str, default=None, help="Model name sent to the API")
    parser.add_argument("--request_timeout_s", type=float, default=None, help="Per-request timeout")
    parser.add_argument("--startup_timeout_s", type=float, default=None, help="Server startup timeout")
    parser.add_argument("--router_trace_phases", type=str, default=None, help="prefill,decode or one subset")
    parser.add_argument("--limit", type=int, default=None, help="Optional request cap")
    parser.add_argument("--reuse_server", action="store_true", help="Use an already-running server")
    parser.add_argument(
        "--with_lora",
        action="store_true",
        help="Launch the traced server with LoRA adapters enabled. Default is no-LoRA for pure B5 router tracing.",
    )
    parser.add_argument("--raw_router_trace_path", type=str, default=None, help="Raw hook output path")
    parser.add_argument("--request_log_path", type=str, default=None, help="Optional replay request log path")
    parser.add_argument("--server_log_path", type=str, default=None, help="Optional server stdout/stderr log path")
    parser.add_argument(
        "--tee_server_output",
        action="store_true",
        help="Mirror the launched server stdout/stderr to this terminal while still writing server_log_path",
    )
    parser.add_argument(
        "--server_log_tail_lines",
        type=int,
        default=60,
        help="How many server log lines to print when startup or replay debugging is needed",
    )
    return parser.parse_args()


def build_request_id_mapping(replay_rows: List[dict]) -> Dict[str, int]:
    request_id_mapping: Dict[str, int] = {}
    for row in replay_rows:
        response_id = row.get("response_id")
        if response_id is None:
            continue
        request_id_mapping[str(response_id)] = int(row["req_idx"])
    return request_id_mapping


def canonicalize_router_trace(
    raw_trace_path: Path,
    model_name: str,
    trace_run_id: str,
    request_id_mapping: Optional[Dict[str, int]] = None,
) -> List[dict]:
    per_request_event_idx: Dict[int, int] = defaultdict(int)
    canonical_rows: List[dict] = []
    seen_event_keys = set()
    for raw_record in iter_jsonl(raw_trace_path):
        if raw_record.get("event") not in (None, "router_trace"):
            continue
        raw_request_id = raw_record.get("server_request_id", raw_record.get("req_idx"))
        mapped_req_idx = None if request_id_mapping is None else request_id_mapping.get(str(raw_request_id))
        req_idx = int(mapped_req_idx) if mapped_req_idx is not None else int(raw_record["req_idx"])
        topk_experts = [int(expert_id) for expert_id in raw_record.get("topk_experts", [])]
        topk_scores = raw_record.get("topk_scores", raw_record.get("topk_weights", []))
        phase = str(raw_record.get("phase", "unknown"))
        layer_id = int(raw_record.get("layer_id", 0))
        token_pos = int(raw_record.get("token_pos", 0))
        normalized_topk_scores = [float(score) for score in topk_scores]
        event_key = (
            req_idx,
            phase,
            layer_id,
            token_pos,
            tuple(topk_experts),
            tuple(normalized_topk_scores),
        )
        if event_key in seen_event_keys:
            continue
        seen_event_keys.add(event_key)

        event_idx = per_request_event_idx[req_idx]
        per_request_event_idx[req_idx] += 1
        canonical_rows.append(
            {
                "req_idx": req_idx,
                "server_request_id": str(raw_request_id),
                "event_idx": event_idx,
                "arrival_idx": len(canonical_rows),
                "raw_arrival_idx": int(raw_record.get("arrival_idx", len(canonical_rows))),
                "layer_id": layer_id,
                "token_pos": token_pos,
                "phase": phase,
                "topk_experts": topk_experts,
                "topk_scores": normalized_topk_scores,
                "num_selected_experts": len(topk_experts),
                "model_name": model_name,
                "trace_run_id": trace_run_id,
            }
        )
    return canonical_rows


def summarize_router_trace(canonical_rows: List[dict]) -> dict:
    expert_counter: Counter = Counter()
    phase_counter: Counter = Counter()
    layer_counter: Counter = Counter()
    request_unique_experts: Dict[int, set] = defaultdict(set)
    request_event_count: Counter = Counter()
    request_phase_count: Dict[int, Counter] = defaultdict(Counter)

    for row in canonical_rows:
        phase_counter[row["phase"]] += 1
        layer_counter[row["layer_id"]] += 1
        req_idx = int(row["req_idx"])
        request_event_count[req_idx] += 1
        request_phase_count[req_idx][row["phase"]] += 1
        for expert_id in row["topk_experts"]:
            expert_counter[int(expert_id)] += 1
            request_unique_experts[req_idx].add(int(expert_id))

    expert_rows = []
    total_expert_hits = sum(expert_counter.values())
    for rank, (expert_id, count) in enumerate(expert_counter.most_common(), start=1):
        expert_rows.append(
            {
                "rank": rank,
                "expert_id": expert_id,
                "count": int(count),
                "share": float(count) / total_expert_hits if total_expert_hits else 0.0,
            }
        )

    request_rows = []
    unique_expert_counts = []
    for req_idx in sorted(request_event_count):
        unique_experts = len(request_unique_experts[req_idx])
        unique_expert_counts.append(unique_experts)
        request_rows.append(
            {
                "req_idx": req_idx,
                "event_count": int(request_event_count[req_idx]),
                "unique_experts": unique_experts,
                "prefill_events": int(request_phase_count[req_idx].get("prefill", 0)),
                "decode_events": int(request_phase_count[req_idx].get("decode", 0)),
            }
        )

    return {
        "summary": {
            "row_count": len(canonical_rows),
            "request_count": len(request_event_count),
            "phase_counts": dict(phase_counter),
            "layer_count": len(layer_counter),
            "unique_experts_per_request": numeric_summary(unique_expert_counts),
        },
        "expert_rows": expert_rows,
        "request_rows": request_rows,
    }


def build_server_command(
    launcher_path: Path,
    model_dir: str,
    port: int,
    router_trace_phases: str,
    raw_router_trace_path: Path,
    tp: int,
    with_lora: bool,
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
        "--router_trace",
        "--router_trace_path",
        str(raw_router_trace_path),
        "--router_trace_phases",
        router_trace_phases,
    ]
    if not with_lora:
        command.append("--no_lora")
    return command


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def log_step(message: str) -> None:
    print(f"[collect_router_trace] {message}", flush=True)


def format_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def read_log_tail(path: Path, max_lines: int) -> List[str]:
    if max_lines <= 0 or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=max_lines)]


def print_log_tail(path: Path, max_lines: int) -> None:
    tail_lines = read_log_tail(path, max_lines)
    if not tail_lines:
        return
    log_step(f"last {len(tail_lines)} lines from {path}:")
    for line in tail_lines:
        print(line, flush=True)


def stream_subprocess_output(process: subprocess.Popen, log_handle: TextIO, mirror_to_stdout: bool) -> threading.Thread:
    if process.stdout is None:
        raise ValueError("process.stdout must be captured before streaming")

    def _pump() -> None:
        try:
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                if mirror_to_stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        finally:
            process.stdout.close()

    thread = threading.Thread(target=_pump, name="collect-router-trace-server-log", daemon=True)
    thread.start()
    return thread


def wait_for_server_ready(
    server_url: str,
    health_path: str,
    timeout_s: float,
    server_process: Optional[subprocess.Popen] = None,
) -> None:
    deadline = time.time() + timeout_s
    health_url = server_url.rstrip("/") + health_path
    next_status_time = time.time()
    last_status = "startup checks have not completed yet"

    while time.time() < deadline:
        if server_process is not None:
            exit_code = server_process.poll()
            if exit_code is not None:
                raise RuntimeError(f"server exited before becoming healthy with exit code {exit_code}")

        try:
            response = requests.get(health_url, timeout=5)
            last_status = f"HTTP {response.status_code}"
            if response.ok:
                return
        except Exception as exc:
            last_status = str(exc)

        now = time.time()
        if now >= next_status_time:
            remaining_s = max(0.0, deadline - now)
            log_step(f"waiting for server health at {health_url} ({remaining_s:.0f}s remaining; last check: {last_status})")
            next_status_time = now + 10.0
        time.sleep(2.0)

    raise TimeoutError(f"server did not become healthy within {timeout_s}s: {health_url}")


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    case_config = config.get("case_study", {})
    router_config = case_config.get("router_trace", {})
    paths_config = config.get("paths", {})

    trace_run_id = args.run_id or str(case_config.get("default_run_id", "router_lora_case_v1"))
    output_dir = stage_output_dir("router_trace", config, args.run_id, args.output_dir)
    requests_path = Path(args.requests_path or (stage_output_dir("prompt_corpus", config, args.run_id) / "fixed_requests.jsonl"))
    request_rows = load_fixed_requests(requests_path, limit=args.limit)

    server_url = args.server_url or str(router_config.get("host", "http://127.0.0.1:8040"))
    port = int(router_config.get("port", 8040))
    health_path = args.health_path or str(router_config.get("health_path", "/healthz"))
    model_name = args.model_name or str(case_config.get("model_name", "Qwen3-VL-30B-A3B-Instruct"))
    model_dir = str(paths_config["model_dir"])
    router_trace_phases = args.router_trace_phases or str(router_config.get("router_trace_phases", "prefill,decode"))
    request_timeout_s = float(args.request_timeout_s or router_config.get("request_timeout_s", 180))
    startup_timeout_s = float(args.startup_timeout_s or router_config.get("startup_timeout_s", 900))
    tp = int(router_config.get("tp", 2))

    raw_router_trace_path = Path(args.raw_router_trace_path) if args.raw_router_trace_path else output_dir / "router_trace_raw.jsonl"
    canonical_router_trace_path = output_dir / "router_trace.jsonl"
    router_summary_path = output_dir / "router_summary.json"
    request_log_path = Path(args.request_log_path) if args.request_log_path else output_dir / "router_request_log.jsonl"
    server_log_path = Path(args.server_log_path) if args.server_log_path else output_dir / "router_server.log"
    collect_summary_path = output_dir / "collect_router_trace_summary.json"
    expert_popularity_path = output_dir / "expert_popularity.csv"
    per_request_stats_path = output_dir / "per_request_expert_stats.csv"

    raw_router_trace_path.parent.mkdir(parents=True, exist_ok=True)
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_router_trace_path.exists():
        raw_router_trace_path.unlink()

    server_process: Optional[subprocess.Popen] = None
    server_log_handle = None
    server_output_thread: Optional[threading.Thread] = None
    launch_command = None
    start_time = time.time()

    try:
        log_step(f"loaded {len(request_rows)} requests from {requests_path}")
        log_step(f"router trace will be written to {raw_router_trace_path}")
        if not args.reuse_server:
            launcher_path = Path(paths_config.get("router_server_launcher", "test/lora/start_server.sh"))
            if not launcher_path.is_absolute():
                launcher_path = Path(__file__).resolve().parents[2] / launcher_path
            launch_command = build_server_command(
                launcher_path=launcher_path,
                model_dir=model_dir,
                port=port,
                router_trace_phases=router_trace_phases,
                raw_router_trace_path=raw_router_trace_path,
                tp=tp,
                with_lora=args.with_lora,
            )
            log_step(f"launching traced server with command: {format_command(launch_command)}")
            log_step(f"server stdout/stderr log: {server_log_path}")
            if args.tee_server_output:
                log_step("teeing launched server output to the terminal")
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
            log_step(f"reusing existing server at {server_url}")
            log_step(f"expecting routed trace output at {raw_router_trace_path}")

        wait_for_server_ready(
            server_url=server_url,
            health_path=health_path,
            timeout_s=startup_timeout_s,
            server_process=server_process,
        )
        log_step(f"server is healthy at {server_url.rstrip('/') + health_path}")
        log_step(f"replaying {len(request_rows)} requests against {server_url}")
        replay_rows = send_fixed_requests(
            request_rows=request_rows,
            server_url=server_url,
            model_name=model_name,
            request_timeout_s=request_timeout_s,
            adapter_ids=[None] * len(request_rows),
            ignore_eos=False,
            respect_request_decode_len=True,
        )
        write_jsonl(request_log_path, replay_rows)
        request_id_mapping = build_request_id_mapping(replay_rows)
        success_count = sum(1 for row in replay_rows if row["success"])
        failure_count = len(replay_rows) - success_count
        log_step(f"completed request replay: {success_count}/{len(replay_rows)} requests succeeded")
        if failure_count:
            first_failure = next(row for row in replay_rows if not row["success"])
            log_step(
                "first failed request: "
                f"req_idx={first_failure['req_idx']} "
                f"status_code={first_failure['status_code']} "
                f"error={first_failure['error']}"
            )
            print_log_tail(server_log_path, args.server_log_tail_lines)

        time.sleep(1.0)
        canonical_rows = canonicalize_router_trace(
            raw_router_trace_path,
            model_name=model_name,
            trace_run_id=trace_run_id,
            request_id_mapping=request_id_mapping,
        )
        write_jsonl(canonical_router_trace_path, canonical_rows)
        log_step(f"loaded {len(canonical_rows)} canonical router events from {raw_router_trace_path}")
        if not canonical_rows:
            log_step("router trace is empty after replay")
            print_log_tail(server_log_path, args.server_log_tail_lines)

        router_outputs = summarize_router_trace(canonical_rows)
        write_json(router_summary_path, router_outputs["summary"])
        write_csv(expert_popularity_path, ["rank", "expert_id", "count", "share"], router_outputs["expert_rows"])
        write_csv(
            per_request_stats_path,
            ["req_idx", "event_count", "unique_experts", "prefill_events", "decode_events"],
            router_outputs["request_rows"],
        )

        write_json(
            collect_summary_path,
            {
                "trace_run_id": trace_run_id,
                "requests_path": str(requests_path),
                "raw_router_trace_path": str(raw_router_trace_path),
                "router_trace_path": str(canonical_router_trace_path),
                "router_summary_path": str(router_summary_path),
                "request_log_path": str(request_log_path),
                "server_log_path": str(server_log_path),
                "request_count": len(replay_rows),
                "router_event_count": len(canonical_rows),
                "launch_command": launch_command,
                "elapsed_s": time.time() - start_time,
            },
        )
    except Exception:
        print_log_tail(server_log_path, args.server_log_tail_lines)
        raise
    finally:
        if server_process is not None:
            terminate_process(server_process)
        if server_output_thread is not None:
            server_output_thread.join(timeout=5)
        if server_log_handle is not None:
            server_log_handle.close()

    print(f"wrote canonical router trace to {canonical_router_trace_path}")


if __name__ == "__main__":
    main()
