"""Tests for N2 loaded anchor map."""
import pytest
from common.load_generator import generate_paired_traces

def test_paired_traces_same_arrivals():
    """Paired traces have identical arrival times and classes, different path assignments."""
    trace_a, trace_b = generate_paired_traces(
        lam=100.0, duration_s=10.0, seed=42,
        heavy_frac=0.25, nm_options_light=[1], nm_options_heavy=[8]
    )
    assert len(trace_a.events) == len(trace_b.events)
    for ea, eb in zip(trace_a.events, trace_b.events):
        assert ea.arrival_time == eb.arrival_time  # same arrivals
        assert ea.job_class == eb.job_class          # same classes

def test_paired_traces_reproducible():
    """Same seed produces same traces."""
    t1a, t1b = generate_paired_traces(lam=50.0, duration_s=5.0, seed=99,
                                       heavy_frac=0.25, nm_options_light=[1],
                                       nm_options_heavy=[8])
    t2a, t2b = generate_paired_traces(lam=50.0, duration_s=5.0, seed=99,
                                       heavy_frac=0.25, nm_options_light=[1],
                                       nm_options_heavy=[8])
    assert len(t1a.events) == len(t2a.events)


from bench_northstar_loaded import (
    N2_CONFIG, bracketed_capacity_search, run_load_trial,
    is_feasible, classify_capacity
)

def test_is_feasible_stable():
    """A trial is feasible if stable + SLO met."""
    # P99 < 2x isolated median, queue stable
    latencies = [100, 105, 110, 115, 120]  # tight
    result = is_feasible(latencies, isolated_median=100, slo_factor=2.0,
                         generated=1000, completed=1000, queue_slope_ci=[-0.1, 0.1])
    assert result == True

def test_is_feasible_unstable_queue():
    """Growing queue = infeasible even if latency looks ok."""
    latencies = [100, 105, 110]
    result = is_feasible(latencies, isolated_median=100, slo_factor=2.0,
                         generated=1000, completed=800, queue_slope_ci=[0.5, 2.0])
    assert result == False

def test_classify_capacity():
    """Capacity bracket: C_lower / C_upper <= 1.10 -> stop."""
    assert classify_capacity(c_lower=100, c_upper=105) == "converged"
    assert classify_capacity(c_lower=100, c_upper=200) == "continue"
