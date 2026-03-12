#!/usr/bin/env python3
"""Join router events with mapped adapter identities and emit QC artifacts."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from common import iter_jsonl, load_global_config, load_seed_config, stage_output_dir, write_csv, write_json


ADAPTER_ID_PATTERN = re.compile(r"^lora_(\d+)$")
PROGRESS_EVERY_ROWS = 250000


@dataclass(frozen=True)
class SelectedAssignment:
    arrival_idx: int
    req_idx: int
    adapter_id: str
    mapping_mode: str
    cardinality: Optional[int]
    start_ts: Optional[int]
    duration_ms: Optional[int]


@dataclass
class AssignmentSelectionStats:
    source_path: str
    total_rows_seen: int
    selected_request_count: int
    first_selected_arrival_idx: Optional[int]
    last_selected_arrival_idx: Optional[int]
    selected_arrival_gap_count: int
    selected_duplicate_arrival_count: int
    selected_arrival_order_violations: int
    mapping_mode_mismatch_count: int
    missing_adapter_id_count: int
    invalid_adapter_id_count: int
    request_count_match: bool
    dropped_request_count: int


@dataclass
class ModeState:
    assignment_by_req_idx: Dict[int, SelectedAssignment]
    selection_stats: AssignmentSelectionStats
    output_path: Path
    writer: object
    joined_row_count: int = 0
    ordering_violations: int = 0
    last_output_key: Optional[Tuple[int, int, int, int]] = None
    request_without_adapter_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join router trace events with mapped adapter identities")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/global.yaml")
    parser.add_argument("--seeds", type=str, default=None, help="Path to configs/seeds.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="Case-study run id")
    parser.add_argument("--output_dir", type=str, default=None, help="Override joined-trace stage directory")
    parser.add_argument("--router_trace_path", type=str, default=None, help="Canonical router_trace.jsonl path")
    parser.add_argument("--router_summary_path", type=str, default=None, help="Router summary JSON path")
    parser.add_argument("--adapter_indep_path", type=str, default=None, help="Mapped independent adapter trace JSONL path")
    parser.add_argument("--adapter_corr_path", type=str, default=None, help="Mapped correlated adapter trace JSONL path")
    parser.add_argument("--joined_indep_path", type=str, default=None, help="Override output joined_trace_indep.jsonl")
    parser.add_argument("--joined_corr_path", type=str, default=None, help="Override output joined_trace_corr.jsonl")
    parser.add_argument("--qc_report_path", type=str, default=None, help="Override output join_qc_report.json")
    parser.add_argument(
        "--projection_summary_path",
        type=str,
        default=None,
        help="Override output join_projection_summary.csv",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def resolve_router_request_count(router_summary: Mapping[str, object]) -> int:
    if "request_count" in router_summary:
        return int(router_summary["request_count"])
    nested_summary = router_summary.get("summary")
    if isinstance(nested_summary, Mapping) and "request_count" in nested_summary:
        return int(nested_summary["request_count"])
    raise ValueError("router summary is missing request_count")


def parse_adapter_slot(adapter_id: Optional[str]) -> Optional[int]:
    if adapter_id is None:
        return None
    match = ADAPTER_ID_PATTERN.fullmatch(str(adapter_id))
    if match is None:
        return None
    return int(match.group(1))


def adapter_selection_sort_key(record: Mapping[str, object], line_idx: int) -> Tuple[int, int]:
    arrival_idx = int(record.get("arrival_idx", line_idx))
    req_idx = int(record.get("req_idx", line_idx))
    return (arrival_idx, req_idx)


def select_first_request_assignments(
    path: Path,
    request_count: int,
    expected_mode: str,
) -> Tuple[Dict[int, SelectedAssignment], AssignmentSelectionStats]:
    if request_count < 0:
        raise ValueError("request_count must be non-negative")

    if request_count == 0:
        stats = AssignmentSelectionStats(
            source_path=str(path),
            total_rows_seen=0,
            selected_request_count=0,
            first_selected_arrival_idx=None,
            last_selected_arrival_idx=None,
            selected_arrival_gap_count=0,
            selected_duplicate_arrival_count=0,
            selected_arrival_order_violations=0,
            mapping_mode_mismatch_count=0,
            missing_adapter_id_count=0,
            invalid_adapter_id_count=0,
            request_count_match=True,
            dropped_request_count=0,
        )
        return {}, stats

    heap: List[Tuple[int, int, int, dict]] = []
    total_rows_seen = 0

    for line_idx, record in enumerate(iter_jsonl(path)):
        total_rows_seen += 1
        sort_key = adapter_selection_sort_key(record, line_idx)
        heap_item = (-sort_key[0], -sort_key[1], -line_idx, dict(record))
        if len(heap) < request_count:
            heapq.heappush(heap, heap_item)
            continue
        largest_key = (-heap[0][0], -heap[0][1], -heap[0][2])
        current_key = (sort_key[0], sort_key[1], line_idx)
        if current_key < largest_key:
            heapq.heapreplace(heap, heap_item)

    selected_records_with_keys = [((-item[0], -item[1], -item[2]), item[3]) for item in heap]
    selected_records_with_keys.sort(key=lambda item: item[0])

    assignment_by_req_idx: Dict[int, SelectedAssignment] = {}
    selected_arrival_gap_count = 0
    selected_duplicate_arrival_count = 0
    selected_arrival_order_violations = 0
    mapping_mode_mismatch_count = 0
    missing_adapter_id_count = 0
    invalid_adapter_id_count = 0
    last_arrival_idx: Optional[int] = None

    for req_idx, (_key, record) in enumerate(selected_records_with_keys):
        arrival_idx = int(record.get("arrival_idx", req_idx))
        adapter_id_value = record.get("adapter_id")
        adapter_id = None if adapter_id_value is None else str(adapter_id_value)
        mapping_mode = str(record.get("mapping_mode", ""))
        cardinality_value = record.get("cardinality")
        cardinality = None if cardinality_value is None else int(cardinality_value)
        if mapping_mode != expected_mode:
            mapping_mode_mismatch_count += 1
        if adapter_id is None:
            missing_adapter_id_count += 1
        else:
            adapter_slot = parse_adapter_slot(adapter_id)
            if adapter_slot is None or (cardinality is not None and adapter_slot >= cardinality):
                invalid_adapter_id_count += 1

        if last_arrival_idx is not None:
            if arrival_idx < last_arrival_idx:
                selected_arrival_order_violations += 1
            elif arrival_idx == last_arrival_idx:
                selected_duplicate_arrival_count += 1
            elif arrival_idx > last_arrival_idx + 1:
                selected_arrival_gap_count += arrival_idx - last_arrival_idx - 1
        last_arrival_idx = arrival_idx

        assignment_by_req_idx[req_idx] = SelectedAssignment(
            arrival_idx=arrival_idx,
            req_idx=req_idx,
            adapter_id="" if adapter_id is None else adapter_id,
            mapping_mode=expected_mode,
            cardinality=cardinality,
            start_ts=int(record["start_ts"]) if record.get("start_ts") is not None else None,
            duration_ms=int(record["duration_ms"]) if record.get("duration_ms") is not None else None,
        )

    selected_request_count = len(assignment_by_req_idx)
    first_selected_arrival_idx = None
    last_selected_arrival_idx = None
    if selected_request_count:
        first_selected_arrival_idx = assignment_by_req_idx[0].arrival_idx
        last_selected_arrival_idx = assignment_by_req_idx[selected_request_count - 1].arrival_idx

    stats = AssignmentSelectionStats(
        source_path=str(path),
        total_rows_seen=total_rows_seen,
        selected_request_count=selected_request_count,
        first_selected_arrival_idx=first_selected_arrival_idx,
        last_selected_arrival_idx=last_selected_arrival_idx,
        selected_arrival_gap_count=selected_arrival_gap_count,
        selected_duplicate_arrival_count=selected_duplicate_arrival_count,
        selected_arrival_order_violations=selected_arrival_order_violations,
        mapping_mode_mismatch_count=mapping_mode_mismatch_count,
        missing_adapter_id_count=missing_adapter_id_count,
        invalid_adapter_id_count=invalid_adapter_id_count,
        request_count_match=selected_request_count == request_count,
        dropped_request_count=max(request_count - selected_request_count, 0),
    )
    return assignment_by_req_idx, stats


def normalize_position_field(record: Mapping[str, object]) -> Tuple[str, int]:
    if record.get("token_pos") is not None:
        return "token_pos", int(record["token_pos"])
    if record.get("chunk_idx") is not None:
        return "chunk_idx", int(record["chunk_idx"])
    raise ValueError("router record is missing token_pos/chunk_idx")


def extract_expert_projection(
    record: Mapping[str, object],
) -> Tuple[str, List[Tuple[int, int]], Optional[int], int]:
    if record.get("topk_experts") is not None:
        raw_topk = record["topk_experts"]
        if isinstance(raw_topk, (list, tuple)):
            topk_experts = [int(expert_id) for expert_id in raw_topk]
        else:
            topk_experts = [int(raw_topk)]
        expected_count = len(topk_experts)
        return "array", list(enumerate(topk_experts)), int(record.get("num_selected_experts", expected_count)), expected_count

    if record.get("expert_id") is not None:
        expert_id = int(record["expert_id"])
        return "exploded", [(0, expert_id)], int(record.get("num_selected_experts", 1)), 1

    raise ValueError("router record must contain topk_experts or expert_id")


def emit_joined_row(
    writer: object,
    assignment: SelectedAssignment,
    router_record: Mapping[str, object],
    position_field: str,
    position_value: int,
    event_idx: int,
    expert_id: int,
) -> int:
    joined_record = {
        "arrival_idx": assignment.arrival_idx,
        "req_idx": int(router_record["req_idx"]),
        "adapter_id": assignment.adapter_id,
        "mapping_mode": assignment.mapping_mode,
        "event_idx": event_idx,
        "layer_id": int(router_record["layer_id"]),
        position_field: position_value,
        "phase": str(router_record["phase"]),
        "expert_id": expert_id,
    }
    if assignment.start_ts is not None:
        joined_record["start_ts"] = assignment.start_ts
    if assignment.duration_ms is not None:
        joined_record["duration_ms"] = assignment.duration_ms
    if router_record.get("model_name") is not None:
        joined_record["model_name"] = str(router_record["model_name"])
    if router_record.get("trace_run_id") is not None:
        joined_record["trace_run_id"] = str(router_record["trace_run_id"])

    writer.write(json.dumps(joined_record, ensure_ascii=True))
    writer.write("\n")
    return 1


def request_bucket_path(temp_dir: Path, req_idx: int) -> Path:
    return temp_dir / f"req_{req_idx:06d}.jsonl"


def main() -> None:
    args = parse_args()
    config = load_global_config(args.config)
    seeds = load_seed_config(args.seeds)
    output_dir = stage_output_dir("joined_trace", config, args.run_id, args.output_dir)
    run_id = str(config.get("case_study", {}).get("default_run_id", "router_lora_case_v1")) if args.run_id is None else args.run_id

    router_dir = stage_output_dir("router_trace", config, run_id)
    adapter_dir = stage_output_dir("adapter_trace", config, run_id)

    router_trace_path = Path(args.router_trace_path) if args.router_trace_path else router_dir / "router_trace.jsonl"
    router_summary_path = Path(args.router_summary_path) if args.router_summary_path else router_dir / "router_summary.json"
    adapter_indep_path = Path(args.adapter_indep_path) if args.adapter_indep_path else adapter_dir / "adapter_trace_mapped_indep.jsonl"
    adapter_corr_path = Path(args.adapter_corr_path) if args.adapter_corr_path else adapter_dir / "adapter_trace_mapped_corr.jsonl"

    joined_indep_path = Path(args.joined_indep_path) if args.joined_indep_path else output_dir / "joined_trace_indep.jsonl"
    joined_corr_path = Path(args.joined_corr_path) if args.joined_corr_path else output_dir / "joined_trace_corr.jsonl"
    qc_report_path = Path(args.qc_report_path) if args.qc_report_path else output_dir / "join_qc_report.json"
    projection_summary_path = (
        Path(args.projection_summary_path) if args.projection_summary_path else output_dir / "join_projection_summary.csv"
    )

    router_summary = load_json(router_summary_path)
    router_request_count_expected = resolve_router_request_count(router_summary)

    indep_assignments, indep_stats = select_first_request_assignments(
        path=adapter_indep_path,
        request_count=router_request_count_expected,
        expected_mode="indep",
    )
    corr_assignments, corr_stats = select_first_request_assignments(
        path=adapter_corr_path,
        request_count=router_request_count_expected,
        expected_mode="corr",
    )

    mode_states: Dict[str, ModeState] = {}
    joined_indep_path.parent.mkdir(parents=True, exist_ok=True)
    joined_corr_path.parent.mkdir(parents=True, exist_ok=True)
    indep_handle = joined_indep_path.open("w", encoding="utf-8")
    corr_handle = joined_corr_path.open("w", encoding="utf-8")
    temp_dir_context = tempfile.TemporaryDirectory(prefix="join_router_buckets_", dir=str(output_dir))
    temp_dir = Path(temp_dir_context.name)
    bucket_handles: Dict[int, object] = {}

    try:
        mode_states["indep"] = ModeState(
            assignment_by_req_idx=indep_assignments,
            selection_stats=indep_stats,
            output_path=joined_indep_path,
            writer=indep_handle,
        )
        mode_states["corr"] = ModeState(
            assignment_by_req_idx=corr_assignments,
            selection_stats=corr_stats,
            output_path=joined_corr_path,
            writer=corr_handle,
        )

        router_row_count = 0
        router_req_idx_set = set()
        router_req_idx_regressions = 0
        router_req_idx_missing_gaps = 0
        router_event_idx_regressions = 0
        router_missing_assignment_row_count = 0
        router_projection_expected_rows = 0
        router_projection_input_form = Counter()
        router_position_field_counts = Counter()
        router_num_selected_mismatch_count = 0
        router_model_name_values = Counter()
        router_trace_run_id_values = Counter()
        last_router_req_idx: Optional[int] = None
        last_event_idx_by_req: Dict[int, int] = {}

        for router_record in iter_jsonl(router_trace_path):
            router_row_count += 1
            req_idx = int(router_record["req_idx"])
            router_req_idx_set.add(req_idx)
            bucket_handle = bucket_handles.get(req_idx)
            if bucket_handle is None:
                bucket_handle = request_bucket_path(temp_dir, req_idx).open("w", encoding="utf-8")
                bucket_handles[req_idx] = bucket_handle
            bucket_handle.write(json.dumps(dict(router_record), ensure_ascii=True))
            bucket_handle.write("\n")

            if last_router_req_idx is not None:
                if req_idx < last_router_req_idx:
                    router_req_idx_regressions += 1
                elif req_idx > last_router_req_idx + 1:
                    router_req_idx_missing_gaps += req_idx - last_router_req_idx - 1
            last_router_req_idx = req_idx

            event_idx = int(router_record.get("event_idx", last_event_idx_by_req.get(req_idx, -1) + 1))
            previous_event_idx = last_event_idx_by_req.get(req_idx)
            if previous_event_idx is not None and event_idx <= previous_event_idx:
                router_event_idx_regressions += 1
            last_event_idx_by_req[req_idx] = event_idx

            position_field, position_value = normalize_position_field(router_record)
            router_position_field_counts[position_field] += 1
            if router_record.get("model_name") is not None:
                router_model_name_values[str(router_record["model_name"])] += 1
            if router_record.get("trace_run_id") is not None:
                router_trace_run_id_values[str(router_record["trace_run_id"])] += 1

            projection_form, ranked_experts, declared_selected_count, actual_selected_count = extract_expert_projection(router_record)
            router_projection_input_form[projection_form] += 1
            router_projection_expected_rows += actual_selected_count
            if declared_selected_count is not None and declared_selected_count != actual_selected_count:
                router_num_selected_mismatch_count += 1

            if router_row_count % PROGRESS_EVERY_ROWS == 0:
                print(f"bucketed {router_row_count} router rows")

        for handle in bucket_handles.values():
            handle.close()
        bucket_handles.clear()

        emitted_router_rows = 0
        for req_idx in sorted(router_req_idx_set):
            bucket_path = request_bucket_path(temp_dir, req_idx)
            missing_assignment_this_router_row = False
            for router_record in iter_jsonl(bucket_path):
                emitted_router_rows += 1
                event_idx = int(router_record.get("event_idx", 0))
                position_field, position_value = normalize_position_field(router_record)
                _projection_form, ranked_experts, _declared_selected_count, _actual_selected_count = extract_expert_projection(
                    router_record
                )

                for _mode_name, state in mode_states.items():
                    assignment = state.assignment_by_req_idx.get(req_idx)
                    if assignment is None or not assignment.adapter_id:
                        state.request_without_adapter_count += 1
                        missing_assignment_this_router_row = True
                        continue

                    for expert_rank, expert_id in ranked_experts:
                        output_key = (assignment.arrival_idx, req_idx, event_idx, expert_rank)
                        if state.last_output_key is not None and output_key < state.last_output_key:
                            state.ordering_violations += 1
                        state.last_output_key = output_key
                        state.joined_row_count += emit_joined_row(
                            writer=state.writer,
                            assignment=assignment,
                            router_record=router_record,
                            position_field=position_field,
                            position_value=position_value,
                            event_idx=event_idx,
                            expert_id=expert_id,
                        )

                if missing_assignment_this_router_row:
                    router_missing_assignment_row_count += 1
                    missing_assignment_this_router_row = False

                if emitted_router_rows % PROGRESS_EVERY_ROWS == 0:
                    print(f"emitted {emitted_router_rows} router rows into joined traces")
    finally:
        for handle in bucket_handles.values():
            handle.close()
        indep_handle.close()
        corr_handle.close()
        temp_dir_context.cleanup()

    router_request_count_observed = len(router_req_idx_set)
    router_req_idx_contiguous = router_req_idx_set == set(range(router_request_count_observed))
    projection_expansion_factor = (
        float(router_projection_expected_rows) / float(router_row_count) if router_row_count else 0.0
    )

    mode_summary_rows = []
    per_mode_report = {}
    total_dropped_or_misaligned = 0
    overall_request_count_match = True
    overall_joined_row_count_match = True
    overall_invalid_adapter_ids = 0
    overall_ordering_ok = True

    for mode_name, state in mode_states.items():
        selection = state.selection_stats
        request_count_match = selection.request_count_match and router_request_count_observed == selection.selected_request_count
        joined_row_count_match = state.joined_row_count == router_projection_expected_rows
        ordering_ok = state.ordering_violations == 0
        invalid_adapter_ids = selection.invalid_adapter_id_count + selection.missing_adapter_id_count
        dropped_or_misaligned = (
            selection.dropped_request_count
            + selection.selected_arrival_gap_count
            + selection.selected_duplicate_arrival_count
            + selection.selected_arrival_order_violations
            + selection.mapping_mode_mismatch_count
            + selection.missing_adapter_id_count
            + selection.invalid_adapter_id_count
            + state.request_without_adapter_count
        )

        total_dropped_or_misaligned += dropped_or_misaligned
        overall_request_count_match = overall_request_count_match and request_count_match
        overall_joined_row_count_match = overall_joined_row_count_match and joined_row_count_match
        overall_invalid_adapter_ids += invalid_adapter_ids
        overall_ordering_ok = overall_ordering_ok and ordering_ok

        report_row = {
            "source_path": selection.source_path,
            "output_path": str(state.output_path),
            "selected_request_count": selection.selected_request_count,
            "source_total_rows": selection.total_rows_seen,
            "first_selected_arrival_idx": selection.first_selected_arrival_idx,
            "last_selected_arrival_idx": selection.last_selected_arrival_idx,
            "selected_arrival_gap_count": selection.selected_arrival_gap_count,
            "selected_duplicate_arrival_count": selection.selected_duplicate_arrival_count,
            "selected_arrival_order_violations": selection.selected_arrival_order_violations,
            "mapping_mode_mismatch_count": selection.mapping_mode_mismatch_count,
            "missing_adapter_id_count": selection.missing_adapter_id_count,
            "invalid_adapter_id_count": selection.invalid_adapter_id_count,
            "request_without_adapter_count": state.request_without_adapter_count,
            "dropped_request_count": selection.dropped_request_count,
            "joined_row_count": state.joined_row_count,
            "joined_row_count_matches_projection": joined_row_count_match,
            "ordering_violations": state.ordering_violations,
            "request_count_match": request_count_match,
            "dropped_or_misaligned_row_count": dropped_or_misaligned,
        }
        per_mode_report[mode_name] = report_row
        mode_summary_rows.append(
            {
                "mapping_mode": mode_name,
                "request_count": selection.selected_request_count,
                "router_request_count": router_request_count_observed,
                "router_row_count": router_row_count,
                "projected_expected_rows": router_projection_expected_rows,
                "joined_row_count": state.joined_row_count,
                "projection_expansion_factor": projection_expansion_factor,
                "dropped_or_misaligned_row_count": dropped_or_misaligned,
                "invalid_adapter_id_count": invalid_adapter_ids,
                "ordering_violations": state.ordering_violations,
                "request_count_match": request_count_match,
                "joined_row_count_matches_projection": joined_row_count_match,
                "output_path": str(state.output_path),
            }
        )

    projection_summary_fieldnames = [
        "mapping_mode",
        "request_count",
        "router_request_count",
        "router_row_count",
        "projected_expected_rows",
        "joined_row_count",
        "projection_expansion_factor",
        "dropped_or_misaligned_row_count",
        "invalid_adapter_id_count",
        "ordering_violations",
        "request_count_match",
        "joined_row_count_matches_projection",
        "output_path",
    ]
    write_csv(projection_summary_path, projection_summary_fieldnames, mode_summary_rows)

    qc_report = {
        "inputs": {
            "router_trace_path": str(router_trace_path),
            "router_summary_path": str(router_summary_path),
            "adapter_indep_path": str(adapter_indep_path),
            "adapter_corr_path": str(adapter_corr_path),
        },
        "outputs": {
            "joined_trace_indep_path": str(joined_indep_path),
            "joined_trace_corr_path": str(joined_corr_path),
            "join_qc_report_path": str(qc_report_path),
            "join_projection_summary_path": str(projection_summary_path),
        },
        "determinism": {
            "global_seed": seeds.get("global_seed"),
            "adapter_mapping_seed": seeds.get("adapter_mapping_seed"),
            "join_seed": None,
            "selection_rule": "Select the first router request_count adapter rows by sorted (arrival_idx, req_idx_or_line_idx), mirroring replay_fixed_requests.load_adapter_assignments truncation semantics.",
            "projection_rule": "For each router request, explode topk_experts in listed order. If the router trace is already exploded and only exposes expert_id, emit one joined row per router row.",
        },
        "router": {
            "request_count_expected_from_summary": router_request_count_expected,
            "request_count_observed_from_trace": router_request_count_observed,
            "request_count_match": router_request_count_expected == router_request_count_observed,
            "row_count": router_row_count,
            "req_idx_contiguous": router_req_idx_contiguous,
            "req_idx_regression_count": router_req_idx_regressions,
            "req_idx_missing_gap_count": router_req_idx_missing_gaps,
            "event_idx_regression_count": router_event_idx_regressions,
            "position_field_counts": dict(router_position_field_counts),
            "projection_input_form_counts": dict(router_projection_input_form),
            "projection_expected_rows_per_mode": router_projection_expected_rows,
            "projection_expansion_factor": projection_expansion_factor,
            "num_selected_experts_mismatch_count": router_num_selected_mismatch_count,
            "missing_assignment_router_row_count": router_missing_assignment_row_count,
            "input_req_idx_regression_count": router_req_idx_regressions,
            "input_req_idx_missing_gap_count": router_req_idx_missing_gaps,
            "model_name_values": dict(router_model_name_values),
            "trace_run_id_values": dict(router_trace_run_id_values),
        },
        "modes": per_mode_report,
        "checks": {
            "request_count_match": overall_request_count_match and (router_request_count_expected == router_request_count_observed),
            "dropped_or_misaligned_row_count": total_dropped_or_misaligned,
            "projection_expansion_factor": projection_expansion_factor,
            "ordering_sanity": overall_ordering_ok
            and router_event_idx_regressions == 0,
            "per_mode_row_counts": {mode_name: state.joined_row_count for mode_name, state in mode_states.items()},
            "joined_row_count_matches_projection": overall_joined_row_count_match,
            "invalid_adapter_id_count": overall_invalid_adapter_ids,
            "indep_and_corr_generated": joined_indep_path.exists() and joined_corr_path.exists(),
            "acceptance_passed": (
                overall_request_count_match
                and router_request_count_expected == router_request_count_observed
                and total_dropped_or_misaligned == 0
                and overall_joined_row_count_match
                and overall_invalid_adapter_ids == 0
                and overall_ordering_ok
                and router_event_idx_regressions == 0
                and joined_indep_path.exists()
                and joined_corr_path.exists()
            ),
        },
    }
    write_json(qc_report_path, qc_report)

    print(f"wrote joined traces to {joined_indep_path} and {joined_corr_path}")
    print(f"wrote QC report to {qc_report_path}")


if __name__ == "__main__":
    main()
