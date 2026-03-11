#!/usr/bin/env python3
"""Replay a fixed request list against the local chat-completions endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional

import requests

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import iter_jsonl, load_global_config, numeric_summary, stage_output_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay fixed requests to a running server")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override router-trace stage directory")
    parser.add_argument("--requests_path", type=str, default=None, help="Path to fixed_requests.jsonl")
    parser.add_argument("--adapter_trace_path", type=str, default=None, help="Optional adapter assignment trace")
    parser.add_argument("--server_url", type=str, default=None, help="Base server URL, e.g. http://127.0.0.1:8040")
    parser.add_argument("--health_path", type=str, default=None, help="Server health path")
    parser.add_argument("--model_name", type=str, default=None, help="Model name passed to the API")
    parser.add_argument("--request_timeout_s", type=float, default=None, help="Per-request timeout")
    parser.add_argument("--startup_timeout_s", type=float, default=None, help="Health wait timeout")
    parser.add_argument("--limit", type=int, default=None, help="Optional request cap")
    parser.add_argument("--ignore_eos", action="store_true", help="Force ignore_eos=true")
    parser.add_argument("--respect_request_decode_len", action="store_true", help="Use target_decode_len from the corpus")
    parser.add_argument("--request_log_path", type=str, default=None, help="Optional per-request JSONL output")
    parser.add_argument("--summary_path", type=str, default=None, help="Optional replay summary JSON output")
    return parser.parse_args()


def load_fixed_requests(path: Path, limit: Optional[int] = None) -> List[dict]:
    requests_rows = list(iter_jsonl(path))
    requests_rows.sort(key=lambda record: int(record["req_idx"]))
    if limit is not None:
        return requests_rows[:limit]
    return requests_rows


def load_adapter_assignments(path: Path, limit: Optional[int] = None) -> List[Optional[str]]:
    rows = []
    for record in iter_jsonl(path):
        arrival_idx = int(record.get("arrival_idx", len(rows)))
        req_idx = int(record.get("req_idx", len(rows)))
        adapter_id = record.get("adapter_id")
        rows.append((arrival_idx, req_idx, None if adapter_id is None else str(adapter_id)))
    rows.sort(key=lambda item: (item[0], item[1]))
    adapter_ids = [adapter_id for _arrival_idx, _req_idx, adapter_id in rows]
    if limit is not None:
        return adapter_ids[:limit]
    return adapter_ids


def wait_for_server(server_url: str, health_path: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = server_url.rstrip("/") + health_path
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s: {url}")


def build_request_body(request_row: dict, model_name: str, adapter_id: Optional[str], ignore_eos: bool, respect_request_decode_len: bool) -> dict:
    messages = request_row.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = [{"role": "user", "content": request_row["prompt_text"]}]

    max_tokens = int(request_row["target_decode_len"]) if respect_request_decode_len else max(int(request_row["target_decode_len"]), 1)
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": bool(ignore_eos),
        "stream": False,
    }
    if adapter_id:
        payload["adapters"] = [adapter_id]
    return payload


def send_fixed_requests(
    request_rows: Iterable[dict],
    server_url: str,
    model_name: str,
    request_timeout_s: float,
    adapter_ids: Optional[List[Optional[str]]] = None,
    ignore_eos: bool = False,
    respect_request_decode_len: bool = True,
) -> List[dict]:
    session = requests.Session()
    output_rows: List[dict] = []

    request_rows_list = list(request_rows)
    adapter_ids = adapter_ids or [None] * len(request_rows_list)
    if len(adapter_ids) != len(request_rows_list):
        raise ValueError("adapter_ids length must match request count")

    for arrival_idx, (request_row, adapter_id) in enumerate(zip(request_rows_list, adapter_ids)):
        payload = build_request_body(
            request_row=request_row,
            model_name=model_name,
            adapter_id=adapter_id,
            ignore_eos=ignore_eos,
            respect_request_decode_len=respect_request_decode_len,
        )
        start_time = time.perf_counter()
        status_code = None
        error_text = None
        finish_reason = None
        response_id = None
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
                    content = message.get("content") or ""
                    completion_chars = len(content)
            else:
                error_text = response.text
        except Exception as exc:
            error_text = str(exc)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        output_rows.append(
            {
                "arrival_idx": arrival_idx,
                "req_idx": int(request_row["req_idx"]),
                "adapter_id": adapter_id,
                "status_code": status_code,
                "success": bool(status_code == 200 and error_text is None),
                "latency_ms": latency_ms,
                "prompt_len_tokens": int(request_row["prompt_len_tokens"]),
                "target_decode_len": int(request_row["target_decode_len"]),
                "response_id": response_id,
                "finish_reason": finish_reason,
                "completion_chars": completion_chars,
                "error": error_text,
            }
        )

        if (arrival_idx + 1) % 25 == 0:
            print(f"completed {arrival_idx + 1} requests")

    return output_rows


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    case_config = config.get("case_study", {})
    router_config = case_config.get("router_trace", {})
    paths_config = config.get("paths", {})

    output_dir = stage_output_dir("router_trace", config, args.run_id, args.output_dir)
    requests_path = Path(args.requests_path or (stage_output_dir("prompt_corpus", config, args.run_id) / "fixed_requests.jsonl"))
    request_rows = load_fixed_requests(requests_path, limit=args.limit)
    adapter_ids = None
    if args.adapter_trace_path:
        adapter_ids = load_adapter_assignments(Path(args.adapter_trace_path), limit=len(request_rows))

    server_url = args.server_url or str(router_config.get("host", "http://127.0.0.1:8040"))
    health_path = args.health_path or str(router_config.get("health_path", "/healthz"))
    startup_timeout_s = float(args.startup_timeout_s or router_config.get("startup_timeout_s", 900))
    request_timeout_s = float(args.request_timeout_s or router_config.get("request_timeout_s", 180))
    model_name = args.model_name or str(case_config.get("model_name", "Qwen3-VL-30B-A3B-Instruct"))

    wait_for_server(server_url, health_path, startup_timeout_s)
    replay_rows = send_fixed_requests(
        request_rows=request_rows,
        server_url=server_url,
        model_name=model_name,
        request_timeout_s=request_timeout_s,
        adapter_ids=adapter_ids,
        ignore_eos=args.ignore_eos,
        respect_request_decode_len=args.respect_request_decode_len or True,
    )

    request_log_path = Path(args.request_log_path) if args.request_log_path else output_dir / "router_request_log.jsonl"
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "router_request_summary.json"

    latencies = [row["latency_ms"] for row in replay_rows]
    success_count = sum(1 for row in replay_rows if row["success"])
    write_jsonl(request_log_path, replay_rows)
    write_json(
        summary_path,
        {
            "requests_path": str(requests_path),
            "server_url": server_url,
            "model_name": model_name,
            "request_count": len(replay_rows),
            "success_count": success_count,
            "failure_count": len(replay_rows) - success_count,
            "latency_ms": numeric_summary(latencies),
            "adapter_trace_path": args.adapter_trace_path,
        },
    )

    print(f"wrote replay log to {request_log_path}")


if __name__ == "__main__":
    main()
