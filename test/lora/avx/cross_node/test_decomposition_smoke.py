# test_decomposition_smoke.py
"""Smoke test: B0 (local, no network) runs end-to-end and accounting closes."""
import pytest
import torch
from bench_decomposition import DecompositionConfig, run_cell

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@CUDA
def test_b0_local_runs_and_accounts():
    config = DecompositionConfig(cell="B0", nm=4, rank=64, n_trials=1, n_iters=10)
    result = run_cell(config)
    assert len(result["latencies_us"]) == 10
    assert all(lat > 0 for lat in result["latencies_us"])
    # B0 is local, so cross_domain_residual should be 0 (no network)
    if "accounting" in result:
        assert result["accounting"]["cross_domain_residual_us"] == pytest.approx(0, abs=100)


@CUDA
def test_b0_correct_output_shape():
    """B0 compute must produce [1, I] output for each miss."""
    config = DecompositionConfig(cell="B0", nm=1, rank=64, n_trials=1, n_iters=1)
    result = run_cell(config)
    assert "outputs" in result
    assert result["outputs"][0].shape == (1, 2048)
