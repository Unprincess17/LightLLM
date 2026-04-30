"""Tests for the P2 cold miss recovery microbenchmark pipeline.

Validates:
  - MicrobenchConfig YAML parsing
  - CSV column contract for recovery and payload artifacts
  - Component timing schema (all required fields present in dispatcher stats)
  - Plotting script CSV reader contract
"""

import csv
import importlib.util
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

MICROBENCH_CONFIG_PATH = REPO_ROOT / "configs/evaluation/colora_kpi/microbench_config.yaml"
MICROBENCH_RUNNER_PATH = REPO_ROOT / "tools/evaluation/colora_microbench_cold_recovery.py"
PLOT_SCRIPT_PATH = REPO_ROOT / "tools/evaluation/plot_cold_recovery_microbench.py"
DISPATCHER_PATH = REPO_ROOT / "lightllm/models/qwen3_vl_moe/lora_dispatch.py"


# ---------------------------------------------------------------------------
# MicrobenchConfig parsing
# ---------------------------------------------------------------------------

def _load_runner_module():
    import sys
    spec = importlib.util.spec_from_file_location("colora_microbench_cold_recovery", MICROBENCH_RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colora_microbench_cold_recovery"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner_module():
    return _load_runner_module()


def test_microbench_config_yaml_exists():
    assert MICROBENCH_CONFIG_PATH.exists(), f"microbench_config.yaml not found at {MICROBENCH_CONFIG_PATH}"


def test_microbench_config_parses(runner_module):
    config = runner_module.MicrobenchConfig.from_yaml(MICROBENCH_CONFIG_PATH)
    assert config.hidden_size > 0
    assert config.dtype in ("fp16", "bf16")
    assert len(config.ranks) >= 2, "should have at least 2 rank values"
    assert len(config.batch_sizes) >= 2, "should have at least 2 batch size values"
    assert config.repeats >= 1
    assert config.output_dir


def test_microbench_config_defaults(runner_module):
    config = runner_module.MicrobenchConfig()
    assert config.ranks == [8, 16, 32, 64]
    assert config.batch_sizes == [1, 2, 4, 8]
    assert config.repeats == 3
    assert config.dtype == "fp16"


# ---------------------------------------------------------------------------
# CSV column contracts
# ---------------------------------------------------------------------------

RECOVERY_FIELDS = [
    "policy", "lora_rank", "batch_size", "repeat_idx",
    "service_time_us", "component_sum_us", "residual_us",
    # ALL_COMPONENTS in alphabetical order (matches runner's sorted(set(...)))
    "TD2H_activation_us", "TH2D_residual_us", "TH2D_weights_us",
    "Tadmit_us", "Tcpu_us", "Tgpu_us", "Tmerge_us", "Tpack_us",
    "exposed_time_us",
]

PAYLOAD_FIELDS = [
    "policy", "lora_rank", "batch_size",
    "promotion_weight_bytes", "execution_activation_bytes",
    "execution_residual_bytes", "payload_ratio",
]

def test_recovery_csv_fieldnames_match(runner_module):
    """Verify the RECOVERY_FIELDS constant in the runner matches the contract."""
    assert runner_module.RECOVERY_FIELDS == RECOVERY_FIELDS


def test_payload_csv_fieldnames_match(runner_module):
    """Verify the PAYLOAD_FIELDS constant in the runner matches the contract."""
    assert runner_module.PAYLOAD_FIELDS == PAYLOAD_FIELDS


def test_csv_writer_produces_correct_columns(runner_module):
    """Verify CSV writers produce the correct column headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recovery_path = Path(tmpdir) / "microbench_cold_recovery.csv"
        payload_path = Path(tmpdir) / "microbench_payload_size.csv"

        sample_recovery = [{
            "policy": "cpu_first", "lora_rank": 16, "batch_size": 4,
            "repeat_idx": 1, "service_time_us": 100.0, "component_sum_us": 100.0,
            "residual_us": 0.0, "TD2H_activation_us": 20.0, "TH2D_residual_us": 20.0,
            "TH2D_weights_us": 0.0, "Tadmit_us": 0.0, "Tcpu_us": 50.0,
            "Tgpu_us": 0.0, "Tmerge_us": 5.0, "Tpack_us": 5.0,
            "exposed_time_us": 0.0,
        }]
        sample_payload = [{
            "policy": "cpu_first", "lora_rank": 16, "batch_size": 4,
            "promotion_weight_bytes": 262144.0, "execution_activation_bytes": 32768.0,
            "execution_residual_bytes": 32768.0, "payload_ratio": 4.0,
        }]

        runner_module._write_recovery_csv(recovery_path, sample_recovery)
        runner_module._write_payload_csv(payload_path, sample_payload)

        with recovery_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == RECOVERY_FIELDS
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["policy"] == "cpu_first"

        with payload_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == PAYLOAD_FIELDS
            rows = list(reader)
            assert len(rows) == 1


# ---------------------------------------------------------------------------
# Dispatcher component timing fields
# ---------------------------------------------------------------------------

DISPATCHER_P2_FIELDS = [
    "pack_time",
    "d2h_activation_time",
    "h2d_residual_time",
    "merge_time",
    "admit_time",
]


def test_dispatcher_has_p2_timing_fields():
    """Verify the dispatcher stats dict includes P2 component timing fields."""
    dispatcher_src = DISPATCHER_PATH.read_text(encoding="utf-8")
    for field in DISPATCHER_P2_FIELDS:
        assert f'"{field}"' in dispatcher_src, f"dispatcher missing P2 field: {field}"


def test_dispatcher_state_has_p2_fields():
    """Verify _MoEHybridPhaseState dataclass includes P2 timing fields."""
    dispatcher_src = DISPATCHER_PATH.read_text(encoding="utf-8")
    for field in DISPATCHER_P2_FIELDS:
        assert f"{field}: float" in dispatcher_src, f"_MoEHybridPhaseState missing: {field}: float"


# ---------------------------------------------------------------------------
# Plotting contract
# ---------------------------------------------------------------------------

def test_plot_script_exists():
    assert PLOT_SCRIPT_PATH.exists(), f"plot script not found at {PLOT_SCRIPT_PATH}"


def test_aggregated_csv_field_contract(runner_module):
    """Verify _aggregate_repeats matches AGG_FIELDS (median + MAD per numeric column)."""
    sample_rows = [
        {
            "policy": "cpu_first", "lora_rank": 16, "batch_size": 4,
            "repeat_idx": 1, "service_time_us": 100.0, "component_sum_us": 100.0,
            "residual_us": 0.0, "Tpack_us": 5.0, "TD2H_activation_us": 20.0,
            "Tcpu_us": 50.0, "TH2D_residual_us": 20.0, "Tmerge_us": 5.0,
            "Tadmit_us": 0.0, "TH2D_weights_us": 0.0, "Tgpu_us": 0.0,
            "exposed_time_us": 0.0,
        },
        {
            "policy": "cpu_first", "lora_rank": 16, "batch_size": 4,
            "repeat_idx": 2, "service_time_us": 110.0, "component_sum_us": 110.0,
            "residual_us": 0.0, "Tpack_us": 6.0, "TD2H_activation_us": 22.0,
            "Tcpu_us": 52.0, "TH2D_residual_us": 22.0, "Tmerge_us": 6.0,
            "Tadmit_us": 0.0, "TH2D_weights_us": 0.0, "Tgpu_us": 0.0,
            "exposed_time_us": 0.0,
        },
    ]
    agg = runner_module._aggregate_repeats(sample_rows)
    assert len(agg) == 1
    row = agg[0]
    assert row["policy"] == "cpu_first"
    assert row["lora_rank"] == 16
    assert row["batch_size"] == 4
    assert row["n_repeats"] == 2
    assert "service_time_us_median" in row
    assert "service_time_us_mad" in row
    for field in runner_module.AGG_FIELDS:
        assert field in row, f"aggregated row missing field: {field}"


# ---------------------------------------------------------------------------
# Component additivity check
# ---------------------------------------------------------------------------

def test_component_additivity_for_execution_first():
    """Verify execution-first component sum equals service time for synthetic data."""
    mod = _load_runner_module()
    stats = {
        "cpu_compute_time": 0.000100,  # 100 us
        "gpu_compute_time": 0.0,
        "weight_h2d_time": 0.0,
        "pack_time": 0.000005,
        "d2h_activation_time": 0.000020,
        "h2d_residual_time": 0.000020,
        "merge_time": 0.000005,
        "admit_time": 0.000010,
    }
    rows = mod._extract_component_timings([stats], "cpu_first")
    assert len(rows) == 1
    row = rows[0]
    # service_time should equal sum of components
    expected_sum = row["Tpack_us"] + row["TD2H_activation_us"] + row["Tcpu_us"] + row["TH2D_residual_us"] + row["Tmerge_us"] + row["Tadmit_us"]
    assert abs(row["service_time_us"] - expected_sum) < 0.01, \
        f"service_time={row['service_time_us']} != component_sum={expected_sum}"
    assert row["residual_us"] < 0.01


def test_component_additivity_for_promotion_first():
    """Verify promotion-first component sum equals service time for synthetic data."""
    mod = _load_runner_module()
    stats = {
        "cpu_compute_time": 0.0,
        "gpu_compute_time": 0.000200,
        "weight_h2d_time": 0.000300,
        "pack_time": 0.0,
        "d2h_activation_time": 0.0,
        "h2d_residual_time": 0.0,
        "merge_time": 0.000010,
        "admit_time": 0.000050,
    }
    rows = mod._extract_component_timings([stats], "load_then_run")
    assert len(rows) == 1
    row = rows[0]
    expected_sum = row["Tadmit_us"] + row["TH2D_weights_us"] + row["Tgpu_us"] + row["Tmerge_us"]
    assert abs(row["service_time_us"] - expected_sum) < 0.01, \
        f"service_time={row['service_time_us']} != component_sum={expected_sum}"
    assert row["residual_us"] < 0.01
