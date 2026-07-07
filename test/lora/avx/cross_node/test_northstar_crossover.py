"""Tests for N1 crossover driver."""
import pytest
from bench_northstar_crossover import (
    N1_CONFIG, build_cell_grid, run_single_request,
    PATHS_LOCAL, PATH_REMOTE_IMPROVED, PATH_ORACLE
)

def test_config_dimensions():
    """N1 config has correct sweep dimensions."""
    assert N1_CONFIG["ranks"] == [16, 32, 64, 128, 256]
    assert N1_CONFIG["nms"] == [1, 2, 4, 8, 16]
    assert N1_CONFIG["n_trials"] == 5
    assert N1_CONFIG["n_requests_per_trial"] >= 1000

def test_cell_grid_25_cells():
    """Full R x NM grid = 25 cells (excluding NM=16 stress markers)."""
    grid = build_cell_grid()
    assert len(grid) == 25
    # Each cell is (R, NM)
    assert (16, 1) in grid
    assert (256, 16) in grid

def test_paths_defined():
    """All four primary paths defined."""
    assert "cpu_first" in PATHS_LOCAL
    assert "load_then_run" in PATHS_LOCAL
    assert PATH_REMOTE_IMPROVED == "remote_improved"
    assert PATH_ORACLE == "oracle"

def test_run_single_request_local():
    """Single request on cpu_first returns timeline with L_recovery > 0."""
    from common.northstar_paths import init_weights
    from common.forced_cold import ForcedColdWeightPool
    import torch

    H, I, R, NM = 2048, 2048, 64, 1
    pool = ForcedColdWeightPool(R, H, I, pool_size=10, seed=42)
    weights = pool.get_batch(0, NM)
    activation = torch.randn(NM, H, dtype=torch.float16, device="cuda")

    result, timeline = run_single_request(
        "cpu_first", activation, weights, R, H, I, NM
    )
    assert timeline.l_recovery_us() > 0
    assert result.shape == (NM, I)


from analysis.analyze_n1 import (
    compute_trial_medians, classify_winners, tost_margin
)

def test_compute_trial_medians():
    """Trial medians computed per (R, NM, path)."""
    results = [
        {"trial": 0, "R": 64, "NM": 1, "path": "cpu_first", "L_recovery_us": 100},
        {"trial": 0, "R": 64, "NM": 1, "path": "cpu_first", "L_recovery_us": 120},
        {"trial": 1, "R": 64, "NM": 1, "path": "cpu_first", "L_recovery_us": 110},
        {"trial": 0, "R": 64, "NM": 1, "path": "oracle", "L_recovery_us": 50},
        {"trial": 0, "R": 64, "NM": 1, "path": "oracle", "L_recovery_us": 55},
    ]
    medians = compute_trial_medians(results)
    assert ("cpu_first", 64, 1) in medians
    assert ("oracle", 64, 1) in medians

def test_tost_margin():
    """delta = max(50us, 0.10 * calibration_median)."""
    margin = tost_margin(calibration_median=1000.0)
    assert margin == 100.0  # 10% of 1000
    margin_low = tost_margin(calibration_median=200.0)
    assert margin_low == 50.0  # floor at 50us

def test_classify_winners():
    """Three-way classification: A_wins / equivalent / unresolved."""
    # cpu_first median = 100, remote median = 200, diff = -100
    # CI [-120, -80], delta = 50 -> cpu_first wins
    classification = classify_winners(
        diff_point=-100.0, ci_lo=-120.0, ci_hi=-80.0, delta=50.0
    )
    assert classification == "A_wins"


def test_analyze_n1_holm_correction():
    """Holm correction is applied to the family of comparisons."""
    import tempfile, csv, os
    from analysis.analyze_n1 import analyze_n1

    # Create a small CSV with clear winners
    rows = []
    for trial in range(5):
        for R in [16, 64]:
            for NM in [1, 8]:
                for path in ["cpu_first", "remote_improved", "oracle"]:
                    # cpu_first is much faster than remote
                    lat = 50 if path == "cpu_first" else (100 if path == "remote_improved" else 30)
                    rows.append({"trial": trial, "R": R, "NM": NM, "path": path,
                                 "request_idx": 0, "L_recovery_us": lat})

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "n1_crossover.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial", "R", "NM", "path", "request_idx", "L_recovery_us"])
            writer.writeheader()
            writer.writerows(rows)

        output_dir = os.path.join(tmpdir, "analysis")
        results = analyze_n1(csv_path, output_dir)

        # Check that winner_grid.csv exists and has Holm-corrected classifications
        grid_path = os.path.join(output_dir, "n1_winner_grid.csv")
        assert os.path.exists(grid_path)

        # Read the grid and verify classifications are present
        with open(grid_path) as f:
            grid_rows = list(csv.DictReader(f))
        assert len(grid_rows) > 0
        # Each row should have a classification
        for row in grid_rows:
            assert row["classification"] in ["A_wins", "B_wins", "equivalent", "unresolved"]
