import importlib.util
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cache_mod = _load_module("tools/case_study/analyze_cache_replay.py", "case_study_cache_replay")


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


def test_run_cache_replay_analysis_emits_expected_metrics_and_csvs(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    joined_corr_path = tmp_path / "joined_trace_corr.jsonl"
    qc_report_path = tmp_path / "join_qc_report.json"
    output_dir = tmp_path / "cache"

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
            "event_idx": 1,
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
            "event_idx": 1,
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
            "event_idx": 1,
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
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 2,
        },
    ]

    _write_jsonl(joined_indep_path, indep_rows)
    _write_jsonl(joined_corr_path, corr_rows)
    _write_json(qc_report_path, {"checks": {"per_mode_row_counts": {"indep": 4, "corr": 4}}})

    result = cache_mod.run_cache_replay_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=output_dir,
        policy="lru",
        cache_budgets=[0, 1, 2],
        progress_every=0,
    )

    assert result["cache_policy"] == "lru"
    assert result["cache_budget_grid"] == [0, 1, 2]
    assert result["comparison_contract"]["identical_event_ordering_across_conditions"] is True
    assert result["request_count"] == 2

    metrics_json = _read_json(output_dir / "cache_metrics.json")
    curve_rows = _read_csv_rows(output_dir / "cache_curve.csv")
    per_request_rows = _read_csv_rows(output_dir / "per_request_miss_count.csv")
    eviction_rows = _read_csv_rows(output_dir / "eviction_stats.csv")

    expert_only_budget_2 = metrics_json["conditions"]["expert_only"]["budgets"]["2"]
    assert expert_only_budget_2["hits"] == 2
    assert expert_only_budget_2["misses"] == 2
    assert expert_only_budget_2["cold_misses"] == 2
    assert expert_only_budget_2["capacity_misses"] == 0
    assert expert_only_budget_2["total_evictions"] == 0

    joint_indep_budget_2 = metrics_json["conditions"]["joint_indep"]["budgets"]["2"]
    assert joint_indep_budget_2["hits"] == 0
    assert joint_indep_budget_2["misses"] == 4
    assert joint_indep_budget_2["cold_misses"] == 4
    assert joint_indep_budget_2["capacity_misses"] == 0
    assert joint_indep_budget_2["total_evictions"] == 2
    assert joint_indep_budget_2["unique_evicted_objects"] == 2
    assert joint_indep_budget_2["mean_residency_if_available"] == 2.0

    joint_corr_budget_2 = metrics_json["conditions"]["joint_corr"]["budgets"]["2"]
    assert joint_corr_budget_2["hits"] == 2
    assert joint_corr_budget_2["misses"] == 2

    assert any(
        row["condition"] == "expert_only"
        and row["cache_budget"] == "2"
        and row["hits"] == "2"
        and row["misses"] == "2"
        and row["hit_rate"] == "0.5"
        for row in curve_rows
    )
    assert any(
        row["condition"] == "joint_indep"
        and row["cache_budget"] == "2"
        and row["total_evictions"] == "2"
        and row["unique_evicted_objects"] == "2"
        and row["mean_residency_if_available"] == "2.0"
        for row in eviction_rows
    )
    assert any(
        row["condition"] == "expert_only"
        and row["cache_budget"] == "2"
        and row["req_idx"] == "1"
        and row["miss_count"] == "0"
        and row["hit_count"] == "2"
        and row["total_events"] == "2"
        for row in per_request_rows
    )
    assert any(
        row["condition"] == "joint_indep"
        and row["cache_budget"] == "2"
        and row["req_idx"] == "1"
        and row["miss_count"] == "2"
        and row["hit_count"] == "0"
        for row in per_request_rows
    )


def test_resolve_cache_budget_grid_rejects_unsorted_duplicates():
    with_ascending = cache_mod.resolve_cache_budget_grid(
        {"case_study": {"replay_cache": {"budget_grid": [0, 8, 16]}}},
        None,
    )
    assert with_ascending == [0, 8, 16]

    try:
        cache_mod.resolve_cache_budget_grid(
            {"case_study": {"replay_cache": {"budget_grid": [0, 16, 16]}}},
            None,
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate cache budgets to raise")
