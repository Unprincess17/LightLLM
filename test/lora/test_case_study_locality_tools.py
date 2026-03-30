import importlib.util
import json
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


locality_mod = _load_module("tools/case_study/analyze_locality.py", "case_study_locality")


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    return rows


def test_run_locality_analysis_emits_expected_json_and_csv_outputs(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    joined_corr_path = tmp_path / "joined_trace_corr.jsonl"
    qc_report_path = tmp_path / "join_qc_report.json"
    output_dir = tmp_path / "locality"
    work_dir = tmp_path / "work"

    indep_rows = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 2,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_1",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_1",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 2,
        },
    ]
    corr_rows = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 2,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 2,
        },
    ]

    _write_jsonl(joined_indep_path, indep_rows)
    _write_jsonl(joined_corr_path, corr_rows)
    _write_json(qc_report_path, {"checks": {"per_mode_row_counts": {"indep": 4, "corr": 4}}})

    result = locality_mod.run_locality_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=output_dir,
        work_dir=work_dir,
        progress_every=0,
    )

    assert result["request_count"] == 2
    expert_only = _read_json(output_dir / "expert_only_locality.json")
    joint_indep = _read_json(output_dir / "joint_indep_locality.json")
    joint_corr = _read_json(output_dir / "joint_corr_locality.json")

    assert expert_only["total_events"] == 4
    assert expert_only["total_distinct_objects"] == 2
    assert expert_only["reuse_distance"]["cold_count"] == 2
    assert expert_only["reuse_distance"]["finite_reuse_count"] == 2
    assert expert_only["reuse_distance"]["finite_mean"] == 1.0
    assert expert_only["per_request_unique_object_count"]["mean"] == 2.0
    assert expert_only["per_request_unique_object_count"]["median"] == 2.0

    assert joint_indep["total_distinct_objects"] == 4
    assert joint_indep["reuse_distance"]["cold_count"] == 4
    assert joint_indep["reuse_distance"]["finite_reuse_count"] == 0
    assert joint_indep["topk_coverage"]["1"] == 0.25

    assert joint_corr["total_distinct_objects"] == 2
    assert joint_corr["reuse_distance"]["cold_count"] == 2
    assert joint_corr["reuse_distance"]["finite_reuse_count"] == 2

    popularity_rows = _read_csv_rows(output_dir / "popularity_rank.csv")
    topk_rows = _read_csv_rows(output_dir / "topk_coverage.csv")
    reuse_rows = _read_csv_rows(output_dir / "reuse_distance_cdf.csv")

    assert len(popularity_rows) == 8
    assert any(row["condition"] == "expert_only" and row["rank"] == "1" and row["fraction"] == "0.5" for row in popularity_rows)
    assert any(row["condition"] == "joint_indep" and row["k"] == "4" and row["coverage"] == "1.0" for row in topk_rows)
    assert any(
        row["condition"] == "expert_only" and row["reuse_distance"] == "-1" and row["cdf"] == "0.5"
        for row in reuse_rows
    )
    assert any(row["condition"] == "joint_indep" and row["reuse_distance"] == "-1" and row["cdf"] == "1.0" for row in reuse_rows)


def test_run_locality_analysis_can_incrementally_refresh_joint_corr_only(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    joined_corr_path = tmp_path / "joined_trace_corr.jsonl"
    qc_report_path = tmp_path / "join_qc_report.json"
    output_dir = tmp_path / "locality"
    work_dir = tmp_path / "work"

    indep_rows = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_1",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
    ]
    corr_rows_initial = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
    ]
    corr_rows_updated = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_1",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
    ]

    _write_jsonl(joined_indep_path, indep_rows)
    _write_jsonl(joined_corr_path, corr_rows_initial)
    _write_json(qc_report_path, {"checks": {"per_mode_row_counts": {"indep": 2, "corr": 2}}})

    locality_mod.run_locality_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=output_dir,
        work_dir=work_dir,
        progress_every=0,
    )

    initial_joint_corr = _read_json(output_dir / "joint_corr_locality.json")
    initial_joint_indep = _read_json(output_dir / "joint_indep_locality.json")

    _write_jsonl(joined_corr_path, corr_rows_updated)
    locality_mod.run_locality_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=output_dir,
        work_dir=work_dir,
        selected_conditions=["joint_corr"],
        progress_every=0,
    )

    refreshed_joint_corr = _read_json(output_dir / "joint_corr_locality.json")
    preserved_joint_indep = _read_json(output_dir / "joint_indep_locality.json")
    topk_rows = _read_csv_rows(output_dir / "topk_coverage.csv")

    assert initial_joint_corr["total_distinct_objects"] == 1
    assert refreshed_joint_corr["total_distinct_objects"] == 2
    assert preserved_joint_indep == initial_joint_indep
    assert {row["condition"] for row in topk_rows} == {"expert_only", "joint_indep", "joint_corr"}


def test_build_access_streams_raises_when_joined_traces_are_not_aligned(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    joined_corr_path = tmp_path / "joined_trace_corr.jsonl"
    work_dir = tmp_path / "work"

    _write_jsonl(
        joined_indep_path,
        [
            {
                "arrival_idx": 0,
                "req_idx": 0,
                "adapter_id": "lora_0",
                "mapping_mode": "indep",
                "event_idx": 0,
                "layer_id": 0,
                "token_pos": 0,
                "phase": "decode",
                "expert_id": 1,
            }
        ],
    )
    _write_jsonl(
        joined_corr_path,
        [
            {
                "arrival_idx": 0,
                "req_idx": 0,
                "adapter_id": "lora_0",
                "mapping_mode": "corr",
                "event_idx": 0,
                "layer_id": 0,
                "token_pos": 0,
                "phase": "decode",
                "expert_id": 2,
            }
        ],
    )

    with pytest.raises(ValueError, match="not event-aligned"):
        locality_mod.build_access_streams(
            joined_indep_path=joined_indep_path,
            joined_corr_path=joined_corr_path,
            total_events=1,
            work_dir=work_dir,
            progress_every=0,
        )
