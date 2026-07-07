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
