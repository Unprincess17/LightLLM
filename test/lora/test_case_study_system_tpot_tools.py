import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_CALIBRATION_V2_PATH = _ROOT / "artifacts/case_study/router_lora_case_v1/calibration/system_baseline_calib_v2.json"


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


system_tpot_mod = _load_module("tools/case_study/analyze_system_tpot.py", "case_study_system_tpot")


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_write_condition_merged_csv_preserves_unselected_conditions(tmp_path: Path):
    output_path = tmp_path / "tpot_quantiles.csv"
    fieldnames = system_tpot_mod.TPOT_QUANTILE_FIELDS

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "condition": "expert_only",
                    "condition_label": "B0",
                    "cache_budget": 1,
                    "miss_handling_mode": "load_then_run",
                    "overlap_policy": "calibrated",
                    "mean": 1.0,
                    "p50": 1.0,
                    "p90": 1.0,
                    "p95": 1.0,
                    "p99": 1.0,
                    "max": 1.0,
                    "tail_threshold_ms": 1.0,
                    "token_count": 1,
                },
                {
                    "condition": "joint_indep",
                    "condition_label": "B1",
                    "cache_budget": 1,
                    "miss_handling_mode": "load_then_run",
                    "overlap_policy": "calibrated",
                    "mean": 2.0,
                    "p50": 2.0,
                    "p90": 2.0,
                    "p95": 2.0,
                    "p99": 2.0,
                    "max": 2.0,
                    "tail_threshold_ms": 2.0,
                    "token_count": 1,
                },
                {
                    "condition": "joint_corr",
                    "condition_label": "B2",
                    "cache_budget": 1,
                    "miss_handling_mode": "load_then_run",
                    "overlap_policy": "calibrated",
                    "mean": 3.0,
                    "p50": 3.0,
                    "p90": 3.0,
                    "p95": 3.0,
                    "p99": 3.0,
                    "max": 3.0,
                    "tail_threshold_ms": 3.0,
                    "token_count": 1,
                },
            ]
        )

    system_tpot_mod.write_condition_merged_csv(
        output_path=output_path,
        fieldnames=fieldnames,
        selected_conditions=["joint_corr"],
        preserve_existing=True,
        rows=[
            {
                "condition": "joint_corr",
                "condition_label": "B2",
                "cache_budget": 1,
                "miss_handling_mode": "load_then_run",
                "overlap_policy": "disabled",
                "mean": 9.0,
                "p50": 9.0,
                "p90": 9.0,
                "p95": 9.0,
                "p99": 9.0,
                "max": 9.0,
                "tail_threshold_ms": 9.0,
                "token_count": 1,
            }
        ],
    )

    rows = _read_csv_rows(output_path)
    by_condition = {row["condition"]: row for row in rows}

    assert by_condition["expert_only"]["p99"] == "1.0"
    assert by_condition["joint_indep"]["p99"] == "2.0"
    assert by_condition["joint_corr"]["p99"] == "9.0"


def test_system_tpot_modes_cover_paper_comparison_set():
    """E3: verify the system-TPOT tooling exposes all modes the paper compares."""
    expected_modes = {
        "load_then_run",
        "execution_first",
        "no_cpu_path",
        "no_deferred_sync",
    }
    actual_modes = set(system_tpot_mod.MISS_HANDLING_MODE_ORDER)
    assert actual_modes == expected_modes, f"Mode mismatch: extra={actual_modes - expected_modes}, missing={expected_modes - actual_modes}"
    assert system_tpot_mod.MISS_HANDLING_MODE_ORDER[0] == "load_then_run"
    assert system_tpot_mod.MISS_HANDLING_MODE_ORDER[1] == "execution_first"


def test_validate_execution_first_calibration_rejects_missing_cold_path_profile():
    with pytest.raises(ValueError, match="execution-first replay requires a populated execution-first calibration manifest"):
        system_tpot_mod.validate_execution_first_calibration(
            calibration={"overlap_windows_ms": {"1": {"early": {"p50_ms": 0.1, "mean_ms": 0.1, "p90_ms": 0.1}, "late": {"p50_ms": 0.1, "mean_ms": 0.1, "p90_ms": 0.1}}}},
            load_profile="stressed",
        )


@pytest.mark.parametrize("load_profile", ["idle", "stressed"])
def test_validate_execution_first_calibration_artifact_has_required_cold_path_curves(load_profile: str):
    calibration = json.loads(_CALIBRATION_V2_PATH.read_text(encoding="utf-8"))
    system_tpot_mod.validate_execution_first_calibration(
        calibration=calibration,
        load_profile=load_profile,
        tool_name="pytest",
    )


def test_build_replay_policy_state_uses_decode_iteration_distance_for_execution_first(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    joined_corr_path = tmp_path / "joined_trace_corr.jsonl"

    base_rows = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
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
            "event_idx": 0,
            "layer_id": 1,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 2,
        },
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "event_idx": 0,
            "layer_id": 1,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 2,
        },
    ]
    indep_rows = [{**row, "mapping_mode": "indep"} for row in base_rows]
    corr_rows = [{**row, "mapping_mode": "corr"} for row in base_rows]
    _write_jsonl(joined_indep_path, indep_rows)
    _write_jsonl(joined_corr_path, corr_rows)

    stream = system_tpot_mod.build_system_streams(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        total_events=len(base_rows),
        progress_every=0,
    )
    policy_state = system_tpot_mod.build_replay_policy_state(
        stream=stream,
        access_buffer=stream.condition_buffers["expert_only"],
        enable_temporal_prefetch=False,
    )

    assert stream.decode_step_ids.tolist() == [0, 0, 1, 1]
    assert policy_state.next_decode_reuse_distance.tolist() == [1, 1, -1, -1]
