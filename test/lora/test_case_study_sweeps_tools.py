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


sweeps_mod = _load_module("tools/case_study/run_synthetic_control_sweeps.py", "case_study_synthetic_sweeps")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_synthetic_control_sweeps_is_reproducible_and_emits_required_artifacts(tmp_path: Path):
    joined_indep_path = tmp_path / "joined_trace_indep.jsonl"
    router_request_log_path = tmp_path / "router_request_log.jsonl"
    output_dir_a = tmp_path / "out_a"
    output_dir_b = tmp_path / "out_b"

    joined_rows = [
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
            "phase": "decode",
            "expert_id": 2,
        },
        {
            "arrival_idx": 1,
            "req_idx": 1,
            "adapter_id": "lora_0",
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
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 1,
            "layer_id": 1,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 3,
        },
        {
            "arrival_idx": 2,
            "req_idx": 2,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 0,
            "layer_id": 1,
            "token_pos": 0,
            "phase": "prefill",
            "expert_id": 2,
        },
        {
            "arrival_idx": 2,
            "req_idx": 2,
            "adapter_id": "lora_0",
            "mapping_mode": "indep",
            "event_idx": 1,
            "layer_id": 1,
            "token_pos": 1,
            "phase": "decode",
            "expert_id": 3,
        },
    ]
    request_log_rows = [
        {"req_idx": 0, "success": True, "latency_ms": 13.0},
        {"req_idx": 1, "success": True, "latency_ms": 14.0},
        {"req_idx": 2, "success": True, "latency_ms": 15.0},
    ]

    _write_jsonl(joined_indep_path, joined_rows)
    _write_jsonl(router_request_log_path, request_log_rows)

    common_kwargs = dict(
        joined_indep_path=joined_indep_path,
        router_request_log_path=router_request_log_path,
        template_request_count=3,
        synthetic_request_count=6,
        num_classes=2,
        num_loras=[2, 4],
        skew_levels=[0.0, 1.0],
        burstiness_levels=[1.0],
        corr_levels=[0.0, 1.0],
        cache_budgets=[1, 2],
        tail_quantile=0.95,
        config={},
        seeds={"global_seed": 7, "synthetic_trace_seed": 17, "replay_seed": 23},
        source_run_id="unit_test_run",
    )

    manifest_a = sweeps_mod.run_synthetic_control_sweeps(output_dir=output_dir_a, **common_kwargs)
    manifest_b = sweeps_mod.run_synthetic_control_sweeps(output_dir=output_dir_b, **common_kwargs)

    results_a = (output_dir_a / "sweep_results.csv").read_text(encoding="utf-8")
    results_b = (output_dir_b / "sweep_results.csv").read_text(encoding="utf-8")
    assert results_a == results_b

    manifest_payload = json.loads((output_dir_a / "sweep_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["parameter_grid"]["num_loras"] == [2, 4]
    assert len(manifest_payload["reproducibility"]["run_points"]) == 12
    assert manifest_payload["strongest_p99_drivers"]

    result_rows = _read_csv_rows(output_dir_a / "sweep_results.csv")
    assert result_rows
    assert sorted(result_rows[0].keys()) == sorted(
        ["run_id", "condition", "num_loras", "skew", "burstiness", "corr_strength", "cache_budget", "miss_rate", "p95", "p99"]
    )
    assert {row["condition"] for row in result_rows} == {"expert_only", "joint_indep", "joint_corr"}

    num_loras_rows = _read_csv_rows(output_dir_a / "num_loras_vs_p99.csv")
    grid_rows = _read_csv_rows(output_dir_a / "skew_burst_corr_grid.csv")
    assert num_loras_rows
    assert grid_rows
    assert manifest_a["outputs"]["sweep_results_csv"].endswith("sweep_results.csv")
    assert manifest_b["outputs"]["skew_burst_corr_grid_csv"].endswith("skew_burst_corr_grid.csv")
