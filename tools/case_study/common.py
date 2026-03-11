#!/usr/bin/env python3
"""Shared helpers for the MoE x LoRA case-study pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOBAL_CONFIG = REPO_ROOT / "configs/global.yaml"
DEFAULT_SEEDS_CONFIG = REPO_ROOT / "configs/seeds.yaml"


def load_yaml_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in YAML file: {path}")
    return payload


def load_global_config(path: Optional[str] = None) -> dict:
    config_path = Path(path) if path else DEFAULT_GLOBAL_CONFIG
    return load_yaml_file(config_path)


def load_seed_config(path: Optional[str] = None) -> dict:
    config_path = Path(path) if path else DEFAULT_SEEDS_CONFIG
    return load_yaml_file(config_path)


def artifact_root(config: Mapping[str, Any]) -> Path:
    root_dir = str(config.get("artifacts", {}).get("root_dir", "artifacts"))
    return REPO_ROOT / root_dir


def resolve_run_id(config: Mapping[str, Any], override: Optional[str] = None) -> str:
    if override:
        return override
    return str(config.get("case_study", {}).get("default_run_id", "router_lora_case_v1"))


def stage_output_dir(
    stage: str,
    config: Mapping[str, Any],
    run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    if output_dir:
        return ensure_dir(Path(output_dir))
    resolved_run_id = resolve_run_id(config, run_id)
    return ensure_dir(artifact_root(config) / "case_study" / resolved_run_id / stage)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True))
            handle.write("\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            cursor = 0
            while cursor < len(payload):
                while cursor < len(payload) and payload[cursor].isspace():
                    cursor += 1
                if cursor >= len(payload):
                    break
                try:
                    record, cursor = decoder.raw_decode(payload, cursor)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}:{exc.colno} invalid JSON object: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_num} is not a JSON object")
                yield record


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def stable_hash_int(token: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def parse_cardinalities(raw_value: Any) -> List[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, int):
        return [int(raw_value)]
    if isinstance(raw_value, (list, tuple)):
        return [int(value) for value in raw_value]
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    return [int(token) for token in tokens]


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if quantile <= 0.0:
        return ordered[0]
    if quantile >= 1.0:
        return ordered[-1]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }

    total = sum(float(value) for value in values)
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": total / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def bucket_histogram(values: Sequence[int], bucket_size: int) -> List[dict]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    counter = Counter()
    for value in values:
        bucket_start = int(value // bucket_size) * bucket_size
        counter[bucket_start] += 1
    rows = []
    for bucket_start in sorted(counter):
        rows.append(
            {
                "bucket_start": bucket_start,
                "bucket_end": bucket_start + bucket_size - 1,
                "count": counter[bucket_start],
            }
        )
    return rows


def top_counter_rows(counter: Counter, key_name: str, total: Optional[int] = None, limit: int = 50) -> List[dict]:
    total_count = total if total is not None else sum(counter.values())
    rows = []
    for rank, (key, count) in enumerate(counter.most_common(limit), start=1):
        share = float(count) / total_count if total_count else 0.0
        rows.append(
            {
                "rank": rank,
                key_name: key,
                "count": int(count),
                "share": share,
            }
        )
    return rows
