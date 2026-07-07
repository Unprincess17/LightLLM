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


from analysis.analyze_n2_5 import compute_delta_tpot, classify_interference

def test_compute_delta_tpot():
    """Delta TPOT = co-located TPOT - inference-only TPOT."""
    delta = compute_delta_tpot(
        coloc_tpot_p99=50.0,  # ms
        baseline_tpot_p99=30.0  # ms
    )
    assert delta == 20.0

def test_classify_interference():
    """Classify interference: significant if delta > tolerance."""
    result = classify_interference(
        delta_tpot=25.0,
        tolerance_abs=5.0,
        tolerance_rel=0.15,
        baseline_tpot=30.0
    )
    # 25 > max(5, 0.15*30=4.5) = 5 -> significant
    assert result == "significant"

    result2 = classify_interference(
        delta_tpot=3.0,
        tolerance_abs=5.0,
        tolerance_rel=0.15,
        baseline_tpot=30.0
    )
    assert result2 == "negligible"
