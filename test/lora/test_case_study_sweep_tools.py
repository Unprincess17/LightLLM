import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sweep_mod = _load_module("tools/case_study/run_synthetic_control_sweeps.py", "case_study_sweep_tools")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_adapter_assignment_pair_is_reproducible_and_zero_corr_matches_indep():
    request_classes = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int64)
    knobs = sweep_mod.SweepKnobs(num_loras=8, skew=1.2, burstiness=0.6, corr_strength=0.0)

    indep_a, corr_a, summary_a = sweep_mod.generate_adapter_assignment_pair(
        request_classes=request_classes,
        knobs=knobs,
        num_classes=3,
        seed=17,
    )
    indep_b, corr_b, summary_b = sweep_mod.generate_adapter_assignment_pair(
        request_classes=request_classes,
        knobs=knobs,
        num_classes=3,
        seed=17,
    )

    assert indep_a.tolist() == indep_b.tolist()
    assert corr_a.tolist() == corr_b.tolist()
    assert indep_a.tolist() == corr_a.tolist()
    assert summary_a == summary_b


def test_run_synthetic_control_sweeps_emits_required_artifacts_and_is_reproducible(tmp_path: Path):
    router_trace_path = tmp_path / "router_trace.jsonl"
    request_log_path = tmp_path / "router_request_log.jsonl"
    output_dir_a = tmp_path / "sweeps_a"
    output_dir_b = tmp_path / "sweeps_b"

    router_rows = [
        {
            "event": "router_trace",
            "arrival_idx": 0,
            "req_idx": 0,
            "phase": "prefill",
            "layer_id": 0,
            "token_pos": 0,
            "event_idx": 0,
            "topk_experts": [1, 2],
        },
        {
            "event": "router_trace",
            "arrival_idx": 1,
            "req_idx": 0,
            "phase": "decode",
            "layer_id": 0,
            "token_pos": 1,
            "event_idx": 1,
            "topk_experts": [1],
        },
        {
            "event": "router_trace",
            "arrival_idx": 2,
            "req_idx": 1,
            "phase": "prefill",
            "layer_id": 0,
            "token_pos": 0,
            "event_idx": 0,
            "topk_experts": [1],
        },
        {
            "event": "router_trace",
            "arrival_idx": 3,
            "req_idx": 1,
            "phase": "decode",
            "layer_id": 0,
            "token_pos": 1,
            "event_idx": 1,
            "topk_experts": [2, 3],
        },
        {
            "event": "router_trace",
            "arrival_idx": 4,
            "req_idx": 2,
            "phase": "prefill",
            "layer_id": 0,
            "token_pos": 0,
            "event_idx": 0,
            "topk_experts": [2],
        },
        {
            "event": "router_trace",
            "arrival_idx": 5,
            "req_idx": 2,
            "phase": "decode",
            "layer_id": 0,
            "token_pos": 1,
            "event_idx": 1,
            "topk_experts": [2],
        },
        {
            "event": "router_trace",
            "arrival_idx": 6,
            "req_idx": 3,
            "phase": "prefill",
            "layer_id": 0,
            "token_pos": 0,
            "event_idx": 0,
            "topk_experts": [1, 3],
        },
        {
            "event": "router_trace",
            "arrival_idx": 7,
            "req_idx": 3,
            "phase": "decode",
            "layer_id": 0,
            "token_pos": 1,
            "event_idx": 1,
            "topk_experts": [3],
        },
    ]
    request_log_rows = [
        {"req_idx": 0, "success": True, "latency_ms": 17.0},
        {"req_idx": 1, "success": True, "latency_ms": 18.0},
        {"req_idx": 2, "success": True, "latency_ms": 15.0},
        {"req_idx": 3, "success": True, "latency_ms": 17.0},
    ]

    _write_jsonl(router_trace_path, router_rows)
    _write_jsonl(request_log_path, request_log_rows)

    baseline_knobs = sweep_mod.SweepKnobs(num_loras=4, skew=1.2, burstiness=0.5, corr_strength=0.5)
    latency_args = argparse.Namespace(
        base_request_ms=None,
        prefill_event_cost_ms=None,
        decode_event_cost_ms=None,
        hit_cost_ms=0.0,
        miss_penalty_ms=None,
        miss_penalty_prefill_ms=5.0,
        miss_penalty_decode_ms=7.0,
        prefill_miss_ratio=0.25,
        tail_quantile=0.95,
    )

    manifest_a = sweep_mod.run_synthetic_control_sweeps(
        output_dir=output_dir_a,
        router_trace_path=router_trace_path,
        router_request_log_path=request_log_path,
        base_run_id="synthetic_test",
        request_limit=4,
        num_classes=3,
        cache_budgets=[0, 2],
        num_loras_levels=[2, 4],
        skew_levels=[0.8, 1.2],
        burstiness_levels=[0.0, 0.8],
        corr_strength_levels=[0.0, 1.0],
        baseline_knobs=baseline_knobs,
        sweep_seed=17,
        seeds={"global_seed": 7, "synthetic_trace_seed": 17},
        latency_args=latency_args,
        config={},
    )
    manifest_b = sweep_mod.run_synthetic_control_sweeps(
        output_dir=output_dir_b,
        router_trace_path=router_trace_path,
        router_request_log_path=request_log_path,
        base_run_id="synthetic_test",
        request_limit=4,
        num_classes=3,
        cache_budgets=[0, 2],
        num_loras_levels=[2, 4],
        skew_levels=[0.8, 1.2],
        burstiness_levels=[0.0, 0.8],
        corr_strength_levels=[0.0, 1.0],
        baseline_knobs=baseline_knobs,
        sweep_seed=17,
        seeds={"global_seed": 7, "synthetic_trace_seed": 17},
        latency_args=latency_args,
        config={},
    )

    for rel_path in (
        "sweep_manifest.json",
        "sweep_results.csv",
        "num_loras_vs_p99.csv",
        "skew_burst_corr_grid.csv",
    ):
        assert (output_dir_a / rel_path).exists()

    sweep_rows = _read_csv_rows(output_dir_a / "sweep_results.csv")
    assert list(sweep_rows[0].keys()) == sweep_mod.SWEEP_RESULTS_FIELDS
    assert all(row["miss_rate"] != "" for row in sweep_rows)
    assert all(row["p95"] != "" for row in sweep_rows)
    assert all(row["p99"] != "" for row in sweep_rows)

    num_loras_rows = _read_csv_rows(output_dir_a / "num_loras_vs_p99.csv")
    assert list(num_loras_rows[0].keys()) == sweep_mod.NUM_LORAS_P99_FIELDS

    grid_rows = _read_csv_rows(output_dir_a / "skew_burst_corr_grid.csv")
    assert list(grid_rows[0].keys()) == sweep_mod.GRID_FIELDS

    assert manifest_a["request_sampling"]["selected_request_ids"] == [0, 1, 2, 3]
    assert manifest_a["robustness_summary"]["joint_p99_knob_sensitivity"]
    assert all("seed" in point for point in manifest_a["points"])

    assert manifest_a["points"] == manifest_b["points"]
    assert (output_dir_a / "sweep_results.csv").read_text(encoding="utf-8") == (
        output_dir_b / "sweep_results.csv"
    ).read_text(encoding="utf-8")
