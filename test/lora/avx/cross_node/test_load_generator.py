"""Tests for common.load_generator — true open-loop Poisson generator."""
import threading
import time
import pytest
from common.load_generator import (
    Trace, generate_poisson_trace, OpenLoopRunner, NoopSink, AdmissionCounters
)


def test_trace_generation():
    trace = generate_poisson_trace(lam=100.0, duration_s=1.0, seed=42,
                                   classes=["light", "heavy"], heavy_frac=0.25)
    assert len(trace) > 50 and len(trace) < 200
    assert all(ev.arrival_time >= 0 for ev in trace)
    heavy_count = sum(1 for ev in trace if ev.job_class == "heavy")
    assert 0.15 < heavy_count / len(trace) < 0.35


def test_trace_reproducible():
    t1 = generate_poisson_trace(lam=50.0, duration_s=0.5, seed=7)
    t2 = generate_poisson_trace(lam=50.0, duration_s=0.5, seed=7)
    assert [ev.arrival_time for ev in t1] == [ev.arrival_time for ev in t2]


def test_admission_counters():
    c = AdmissionCounters()
    c.generated += 1
    c.c0_inserted += 1
    c.c2_admitted += 1
    c.completed += 1
    d = c.to_dict()
    assert d["generated"] == 1
    assert d["c2_admitted"] == 1


def test_open_loop_runner_does_not_block_on_slow_sink():
    """Generator must generate at the scheduled rate even if sink is slow."""
    trace = generate_poisson_trace(lam=1000.0, duration_s=0.1, seed=1)
    sink = NoopSink(process_time_s=0.01)  # slow: 10ms per request
    runner = OpenLoopRunner(trace, sink, ingress_capacity=len(trace) + 100,
                            drain_timeout_s=0.0)
    runner.run()
    c = runner.counters
    assert c.generated == len(trace)
    assert c.c0_inserted == len(trace)
    assert c.completed < c.c0_inserted  # some unfinished


def test_noop_sink_validates_generator():
    """No-op sink (no work) should complete everything near-instantly."""
    trace = generate_poisson_trace(lam=100.0, duration_s=0.2, seed=2)
    sink = NoopSink(process_time_s=0.0)
    runner = OpenLoopRunner(trace, sink, ingress_capacity=len(trace) + 100)
    runner.run()
    assert runner.counters.completed == len(trace)
    assert runner.counters.unfinished == 0
