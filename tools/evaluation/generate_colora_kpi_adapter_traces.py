#!/usr/bin/env python3
"""Generate adapter-assignment traces for COLoRA live E2E KPI runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.case_study.common import write_jsonl


def build_trace_rows(
    *,
    requests: int,
    adapter_count: int,
    adapter_start: int = 0,
    pattern: str = "round_robin",
    adapter_prefix: str = "lora_dummy_",
    stride: int = 7,
) -> list[dict[str, object]]:
    if requests < 0:
        raise ValueError("requests must be non-negative")
    if adapter_count <= 0:
        raise ValueError("adapter_count must be positive")
    if pattern not in {"round_robin", "stride"}:
        raise ValueError(f"unsupported pattern: {pattern}")

    rows: list[dict[str, object]] = []
    for req_idx in range(requests):
        if pattern == "stride":
            adapter_idx = adapter_start + ((req_idx * stride) % adapter_count)
        else:
            adapter_idx = adapter_start + (req_idx % adapter_count)
        rows.append(
            {
                "arrival_idx": req_idx,
                "req_idx": req_idx,
                "adapter_id": f"{adapter_prefix}{adapter_idx}",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSONL trace path to write")
    parser.add_argument("--requests", type=int, required=True, help="Number of adapter assignments")
    parser.add_argument("--adapter-count", type=int, default=10, help="Number of adapters to cycle through")
    parser.add_argument("--adapter-start", type=int, default=0, help="First adapter index in the cycled range")
    parser.add_argument("--adapter-prefix", default="lora_dummy_", help="Adapter id prefix")
    parser.add_argument("--pattern", choices=["round_robin", "stride"], default="stride")
    parser.add_argument("--stride", type=int, default=7, help="Stride for the stride pattern")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_trace_rows(
        requests=args.requests,
        adapter_count=args.adapter_count,
        adapter_start=args.adapter_start,
        pattern=args.pattern,
        adapter_prefix=args.adapter_prefix,
        stride=args.stride,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
