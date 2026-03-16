import argparse
import csv
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


cache_mod = _load_module("tools/case_study/analyze_cache_replay.py", "case_study_cache_replay_for_latency")
latency_mod = _load_module("tools/case_study/analyze_latency_proxy.py", "case_study_latency_proxy")


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fit_base_compute_model_recovers_phase_coefficients():
    result = latency_mod.fit_base_compute_model(
        request_ids=[0, 1, 2, 3],
        per_request_prefill_events=latency_mod.np.asarray([0, 1, 2, 1], dtype=latency_mod.np.int64),
        per_request_decode_events=latency_mod.np.asarray([1, 1, 1, 3], dtype=latency_mod.np.int64),
        request_latencies_ms={
            0: 13.0,
            1: 15.0,
            2: 17.0,
            3: 21.0,
        },
        base_request_ms_override=None,
        prefill_event_cost_ms_override=None,
        decode_event_cost_ms_override=None,
    )

    assert abs(result["base_request_ms"] - 10.0) < 1e-6
    assert abs(result["prefill_event_cost_ms"] - 2.0) < 1e-6
    assert abs(result["decode_event_cost_ms"] - 3.0) < 1e-6


def test_run_latency_proxy_analysis_emits_quantiles_tail_breakdown_and_model_params(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    joined_corr_path = tmp_path / "joined_trace_corr.jsonl"
    qc_report_path = tmp_path / "join_qc_report.json"
    cache_output_dir = tmp_path / "cache"
    latency_output_dir = tmp_path / "latency"
    router_request_log_path = tmp_path / "router_request_log.jsonl"

    indep_rows = [
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "prefill",
            "expert_id": 1,
        },
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 1,
            "phase": "prefill",
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
            "phase": "prefill",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_1",
            "mapping_mode": "indep",
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 2,
        },
        {
            "arrival_idx": 2,
            "req_idx": 2,
            "adapter_id": "lora_2",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 2,
            "req_idx": 2,
            "adapter_id": "lora_2",
            "mapping_mode": "indep",
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 1,
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
            "phase": "prefill",
            "expert_id": 1,
        },
        {
            "arrival_idx": 0,
            "req_idx": 0,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 1,
            "phase": "prefill",
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
            "phase": "prefill",
            "expert_id": 1,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_0",
            "mapping_mode": "corr",
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 2,
        },
        {
            "arrival_idx": 2,
            "req_idx": 2,
            "adapter_id": "lora_2",
            "mapping_mode": "corr",
            "event_idx": 0,
            "layer_id": 0,
            "token_pos": 0,
            "phase": "decode",
            "expert_id": 1,
        },
        {
            "arrival_idx": 2,
            "req_idx": 2,
            "adapter_id": "lora_2",
            "mapping_mode": "corr",
            "event_idx": 1,
            "layer_id": 0,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 2,
        },
    ]
    request_log_rows = [
        {"req_idx": 0, "success": True, "latency_ms": 14.0},
        {"req_idx": 1, "success": True, "latency_ms": 15.0},
        {"req_idx": 2, "success": True, "latency_ms": 16.0},
    ]

    _write_jsonl(joined_indep_path, indep_rows)
    _write_jsonl(joined_corr_path, corr_rows)
    _write_jsonl(router_request_log_path, request_log_rows)
    _write_json(qc_report_path, {"checks": {"per_mode_row_counts": {"indep": 6, "corr": 6}}})

    cache_mod.run_cache_replay_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        output_dir=cache_output_dir,
        policy="lru",
        cache_budgets=[0, 2],
        progress_every=0,
    )

    args = argparse.Namespace(
        base_request_ms=10.0,
        prefill_event_cost_ms=2.0,
        decode_event_cost_ms=3.0,
        hit_cost_ms=0.0,
        miss_penalty_ms=None,
        miss_penalty_prefill_ms=5.0,
        miss_penalty_decode_ms=7.0,
        prefill_miss_ratio=0.25,
        tail_quantile=0.95,
    )

    result = latency_mod.run_latency_proxy_analysis(
        joined_indep_path=joined_indep_path,
        joined_corr_path=joined_corr_path,
        qc_report_path=qc_report_path,
        cache_curve_path=cache_output_dir / "cache_curve.csv",
        output_dir=latency_output_dir,
        router_request_log_path=router_request_log_path,
        per_request_miss_path=cache_output_dir / "per_request_miss_count.csv",
        progress_every=0,
        tail_quantile=0.95,
        args=args,
        config={},
    )

    assert result["phase_aware"] is True
    assert result["condition_labels"] == {"expert_only": "B0", "joint_indep": "B1", "joint_corr": "B2"}

    latency_json = _read_json(latency_output_dir / "latency_proxy.json")
    quantile_rows = _read_csv_rows(latency_output_dir / "latency_quantiles.csv")
    tail_rows = _read_csv_rows(latency_output_dir / "tail_request_breakdown.csv")
    phase_rows = _read_csv_rows(latency_output_dir / "prefill_decode_breakdown.csv")

    params = latency_json["parameters"]
    assert abs(params["base_request_ms"] - 10.0) < 1e-6
    assert abs(params["prefill_event_cost_ms"] - 2.0) < 1e-6
    assert abs(params["decode_event_cost_ms"] - 3.0) < 1e-6
    assert params["miss_penalty_prefill_ms"] == 5.0
    assert params["miss_penalty_decode_ms"] == 7.0

    budget2_rows = {(row["condition"], row["cache_budget"]): row for row in quantile_rows}
    assert ("B0", "2") in budget2_rows
    assert ("B1", "2") in budget2_rows
    assert ("B2", "2") in budget2_rows

    b0_budget2 = budget2_rows[("B0", "2")]
    assert abs(float(b0_budget2["mean"]) - (24.0 + 15.0 + 16.0) / 3.0) < 1e-9
    assert abs(float(b0_budget2["p50"]) - 16.0) < 1e-9
    assert abs(float(b0_budget2["max"]) - 24.0) < 1e-9

    b1_budget2 = budget2_rows[("B1", "2")]
    assert abs(float(b1_budget2["mean"]) - 27.0) < 1e-9
    assert abs(float(b1_budget2["p50"]) - 27.0) < 1e-9
    assert abs(float(b1_budget2["max"]) - 30.0) < 1e-9

    assert any(
        row["condition"] == "B0"
        and row["cache_budget"] == "2"
        and row["req_idx"] == "0"
        and row["miss_count"] == "2"
        and row["cold_miss_count"] == "2"
        and row["prefill_miss_count"] == "2"
        and row["decode_miss_count"] == "0"
        for row in tail_rows
    )
    assert any(
        row["condition"] == "B1"
        and row["cache_budget"] == "2"
        and row["req_idx"] == "2"
        and row["miss_count"] == "2"
        and row["prefill_miss_count"] == "0"
        and row["decode_miss_count"] == "2"
        for row in tail_rows
    )
    assert any(
        row["condition"] == "B1"
        and row["cache_budget"] == "2"
        and row["metric_name"] == "mean_prefill_miss_count"
        and abs(float(row["metric_value"]) - 1.0) < 1e-9
        for row in phase_rows
    )
    assert any(
        row["condition"] == "B1"
        and row["cache_budget"] == "2"
        and row["metric_name"] == "tail_p95_decode_miss_count_mean"
        and abs(float(row["metric_value"]) - 2.0) < 1e-9
        for row in phase_rows
    )
