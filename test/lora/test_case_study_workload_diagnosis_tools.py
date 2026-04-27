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


diag_mod = _load_module("tools/case_study/workload_diagnosis.py", "case_study_workload_diagnosis")


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_stratified_sample_preserves_group_mix(tmp_path: Path):
    rows = []
    for index in range(12):
        key = "mix:A+B" if index < 9 else "mix:C+D"
        rows.append({"row_idx": index, "composite_key": key})

    sampled = diag_mod.stratified_sample_by_key(rows, key_field="composite_key", sample_size=6, seed=11)
    counts = {}
    for row in sampled:
        counts[row["composite_key"]] = counts.get(row["composite_key"], 0) + 1

    assert len(sampled) == 6
    assert counts["mix:A+B"] == 4
    assert counts["mix:C+D"] == 2


def test_generate_pressure_trace_reaches_target_size():
    seed_rows = [
        {"arrival_idx": 0, "adapter_id": "lora_0"},
        {"arrival_idx": 1, "adapter_id": "lora_1"},
        {"arrival_idx": 2, "adapter_id": "lora_0"},
        {"arrival_idx": 3, "adapter_id": "lora_2"},
    ]
    pressure = diag_mod.generate_pressure_trace(
        seed_rows=seed_rows,
        target_count=10,
        burst_length=3,
        seed=7,
    )

    assert len(pressure) == 10
    assert [row["arrival_idx"] for row in pressure] == list(range(10))
    assert all("pressure_source_idx" in row for row in pressure)


def test_build_summary_rows_reads_locality_and_cache_outputs(tmp_path: Path):
    workload_dir = tmp_path / "wk1"
    locality_dir = workload_dir / "replay" / "locality"
    cache_dir = workload_dir / "replay" / "cache"
    locality_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    _write_json(
        locality_dir / "joint_corr_locality.json",
        {
            "condition": "joint_corr",
            "reuse_distance": {"finite_p95": 55.0},
            "topk_coverage": {"1000": 0.31},
        },
    )
    _write_json(
        locality_dir / "expert_only_locality.json",
        {
            "condition": "expert_only",
            "reuse_distance": {"finite_p95": 21.0},
            "topk_coverage": {"1000": 0.88},
        },
    )
    _write_csv(
        cache_dir / "cache_curve.csv",
        fieldnames=["condition", "cache_budget", "miss_rate"],
        rows=[
            {"condition": "expert_only", "cache_budget": 2048, "miss_rate": 0.002},
            {"condition": "joint_corr", "cache_budget": 2048, "miss_rate": 0.015},
        ],
    )

    rows = diag_mod.build_summary_rows(
        workloads=[
            diag_mod.WorkloadSummaryInput(
                workload="poisson_current",
                run_root=workload_dir,
            )
        ],
        summary_budget=2048,
        topk_key="1000",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["workload"] == "poisson_current"
    assert abs(float(row["joint_corr_miss_rate"]) - 0.015) < 1e-9
    assert abs(float(row["expert_only_miss_rate"]) - 0.002) < 1e-9
    assert abs(float(row["joint_corr_reuse_p95"]) - 55.0) < 1e-9
    assert abs(float(row["joint_corr_hotset_coverage"]) - 0.31) < 1e-9
