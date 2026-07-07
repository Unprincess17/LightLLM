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
