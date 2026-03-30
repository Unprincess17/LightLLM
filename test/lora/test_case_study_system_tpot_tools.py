import csv
import importlib.util
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


system_tpot_mod = _load_module("tools/case_study/analyze_system_tpot.py", "case_study_system_tpot")


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
