"""Tests for N2.5 co-location calibration."""
import pytest
from bench_northstar_colocation import (
    N2_5_CONFIG, run_inference_only_baseline,
    run_colocation_trial, compute_inference_intensity
)

def test_config_has_4_operating_points():
    """N2.5 config has 4 operating points from N2."""
    assert len(N2_5_CONFIG["operating_points"]) == 4

def test_config_has_3_intensities():
    """Three inference intensities: low, moderate, near_knee."""
    assert set(N2_5_CONFIG["intensities"]) == {"low", "moderate", "near_knee"}

def test_compute_inference_intensity():
    """Intensity = fraction of inference-only stable capacity."""
    assert compute_inference_intensity("low", knee=1000) == 300
    assert compute_inference_intensity("moderate", knee=1000) == 600
    assert compute_inference_intensity("near_knee", knee=1000) == 900

def test_recovery_lambda_common():
    """Recovery lambda is common across paths: 0.7 * min(C_cpu, C_remote)."""
    c_cpu = 500
    c_remote = 800
    lam = N2_5_CONFIG["recovery_lambda_factor"] * min(c_cpu, c_remote)
    assert lam == 0.7 * 500  # 350
