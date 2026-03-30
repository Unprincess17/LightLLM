#!/usr/bin/env python3
"""Shared helpers for the COLoRA ablation toolchain."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


THIS_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = THIS_DIR.parent / "case_study"

import sys

if str(CASE_STUDY_DIR) not in sys.path:
    sys.path.append(str(CASE_STUDY_DIR))

from analyze_locality import iter_jsonl_bytes, resolve_total_events  # type: ignore
from common import artifact_root, ensure_dir, ensure_parent_dir, load_global_config, percentile, resolve_run_id  # type: ignore


DEFAULT_EVENT_WINDOW = 10_000
DEFAULT_REPEAT_WINDOWS = 1
DEFAULT_TIMELINE_BUDGET = 2048


@dataclass(frozen=True)
class AblationOutputPaths:
    root_dir: Path
    metrics_dir: Path
    figures_dir: Path
    timelines_dir: Path
    manifests_dir: Path


@dataclass(frozen=True)
class RepeatWindow:
    repeat_id: str
    request_start: int
    request_end: int
    request_count: int


@dataclass(frozen=True)
class MaterializedWindow:
    repeat_id: str
    joined_indep_path: Path
    joined_corr_path: Path
    total_events: int
    request_start: int
    request_end: int
    request_count: int


def resolve_ablation_output_paths(
    config: Mapping[str, object],
    run_id: str,
    suite_id: str,
    output_root: Optional[str] = None,
) -> AblationOutputPaths:
    if output_root:
        root_dir = ensure_dir(Path(output_root) / run_id / suite_id)
    else:
        root_dir = ensure_dir(artifact_root(config) / "ablation" / run_id / suite_id)
    return AblationOutputPaths(
        root_dir=root_dir,
        metrics_dir=ensure_dir(root_dir / "metrics"),
        figures_dir=ensure_dir(root_dir / "figures"),
        timelines_dir=ensure_dir(root_dir / "timelines"),
        manifests_dir=ensure_dir(root_dir / "manifests"),
    )


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def quantile_summary(values: Sequence[float]) -> dict:
    ordered = [float(value) for value in values]
    if not ordered:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p999": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "p50": float(percentile(ordered, 0.50)),
        "p90": float(percentile(ordered, 0.90)),
        "p95": float(percentile(ordered, 0.95)),
        "p99": float(percentile(ordered, 0.99)),
        "p999": float(percentile(ordered, 0.999)),
        "max": float(max(ordered)),
    }


def detect_request_count(joined_path: Path) -> int:
    current_req_idx: Optional[int] = None
    request_count = 0
    for row in iter_jsonl_bytes(joined_path):
        req_idx = int(row["req_idx"])
        if current_req_idx is None:
            current_req_idx = req_idx
            request_count = 1
            continue
        if req_idx == current_req_idx:
            continue
        if req_idx != current_req_idx + 1:
            raise ValueError(
                "joined trace req_idx order must be contiguous for ablation windows: "
                f"previous={current_req_idx}, current={req_idx}"
            )
        current_req_idx = req_idx
        request_count += 1
    return request_count


def build_repeat_windows(request_count: int, repeat_windows: int) -> List[RepeatWindow]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    requested = max(int(repeat_windows), 1)
    actual = min(requested, request_count)
    base = request_count // actual
    remainder = request_count % actual
    windows: List[RepeatWindow] = []
    cursor = 0
    for index in range(actual):
        width = base + (1 if index < remainder else 0)
        repeat_id = f"r{index:02d}"
        windows.append(
            RepeatWindow(
                repeat_id=repeat_id,
                request_start=cursor,
                request_end=cursor + width,
                request_count=width,
            )
        )
        cursor += width
    return windows


def materialize_repeat_window(
    joined_indep_path: Path,
    joined_corr_path: Path,
    window: RepeatWindow,
    temp_dir: Path,
) -> MaterializedWindow:
    temp_dir.mkdir(parents=True, exist_ok=True)
    indep_output = temp_dir / f"{window.repeat_id}_joined_trace_indep.jsonl"
    corr_output = temp_dir / f"{window.repeat_id}_joined_trace_corr.jsonl"

    selected_rows = 0
    first_arrival_idx: Optional[int] = None
    first_start_ts: Optional[int] = None

    with indep_output.open("w", encoding="utf-8") as indep_handle, corr_output.open("w", encoding="utf-8") as corr_handle:
        for indep_row, corr_row in zip_longest(iter_jsonl_bytes(joined_indep_path), iter_jsonl_bytes(joined_corr_path)):
            if indep_row is None or corr_row is None:
                raise ValueError("joined indep/corr traces have different row counts")
            req_idx = int(indep_row["req_idx"])
            if req_idx < window.request_start or req_idx >= window.request_end:
                continue
            if first_arrival_idx is None:
                first_arrival_idx = int(indep_row.get("arrival_idx", req_idx))
            if first_start_ts is None and indep_row.get("start_ts") is not None:
                first_start_ts = int(indep_row["start_ts"])

            indep_payload = dict(indep_row)
            corr_payload = dict(corr_row)
            reindexed_req_idx = req_idx - window.request_start
            indep_payload["req_idx"] = reindexed_req_idx
            corr_payload["req_idx"] = reindexed_req_idx
            if "arrival_idx" in indep_payload:
                indep_payload["arrival_idx"] = int(indep_payload.get("arrival_idx", req_idx)) - int(first_arrival_idx)
            if "arrival_idx" in corr_payload:
                corr_payload["arrival_idx"] = int(corr_payload.get("arrival_idx", req_idx)) - int(first_arrival_idx)
            if first_start_ts is not None and indep_payload.get("start_ts") is not None:
                indep_payload["start_ts"] = int(indep_payload["start_ts"]) - int(first_start_ts)
            if first_start_ts is not None and corr_payload.get("start_ts") is not None:
                corr_payload["start_ts"] = int(corr_payload["start_ts"]) - int(first_start_ts)
            indep_handle.write(json.dumps(indep_payload, ensure_ascii=True))
            indep_handle.write("\n")
            corr_handle.write(json.dumps(corr_payload, ensure_ascii=True))
            corr_handle.write("\n")
            selected_rows += 1

    if selected_rows <= 0:
        raise ValueError(
            f"repeat window {window.repeat_id} selected no rows from {joined_indep_path}"
        )
    return MaterializedWindow(
        repeat_id=window.repeat_id,
        joined_indep_path=indep_output,
        joined_corr_path=corr_output,
        total_events=selected_rows,
        request_start=window.request_start,
        request_end=window.request_end,
        request_count=window.request_count,
    )


def resolve_case_study_inputs(
    config: Mapping[str, object],
    run_id: str,
    joined_indep_path: Optional[str],
    joined_corr_path: Optional[str],
    qc_report_path: Optional[str],
    calibration_path: Optional[str],
) -> dict:
    run_root = artifact_root(config) / "case_study" / run_id
    resolved = {
        "joined_indep_path": Path(joined_indep_path) if joined_indep_path else run_root / "joined_trace" / "joined_trace_indep.jsonl",
        "joined_corr_path": Path(joined_corr_path) if joined_corr_path else run_root / "joined_trace" / "joined_trace_corr.jsonl",
        "qc_report_path": Path(qc_report_path) if qc_report_path else run_root / "joined_trace" / "join_qc_report.json",
        "calibration_path": Path(calibration_path) if calibration_path else run_root / "calibration" / "system_baseline_calibration.json",
    }
    return resolved


def resolve_total_events_from_qc(qc_report_path: Path) -> int:
    return int(resolve_total_events(load_json(qc_report_path)))


def create_temp_dir(prefix: str = "ablation_window_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def ensure_inputs_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing required inputs:\n{formatted}")


def resolve_config_and_run_id(config_path: Optional[str], run_id: Optional[str]) -> tuple[dict, str]:
    config = load_global_config(config_path)
    return config, resolve_run_id(config, run_id)


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
