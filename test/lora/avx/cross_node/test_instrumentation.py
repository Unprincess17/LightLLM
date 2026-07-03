import pytest
import torch
from common.instrumentation import RequestTimeline, RunMetadata, account_request, should_flag_instrumentation_gap


def test_timeline_set_and_get():
    tl = RequestTimeline(req_id=1)
    tl.set("t0", 1.0)
    tl.set("t18", 1.05)
    assert tl.get("t0") == 1.0
    assert tl.e2e_us() == pytest.approx(50000.0)  # 0.05s * 1e6 = 50000us


def test_timeline_null_persistent_tcp():
    tl = RequestTimeline(req_id=1, cell="B2")
    tl.set("t0", 0.0)
    tl.set("t5", 0.01)
    tl.set("t6", 0.011)
    tl.set("t17", 0.04)
    tl.set("t18", 0.041)
    acc = account_request(tl)
    assert acc["client_request_span_us"] == pytest.approx(10000.0)   # 0.01s
    assert acc["server_span_us"] == pytest.approx(29000.0)           # 0.029s
    assert acc["cross_domain_residual_us"] == pytest.approx(2000.0)  # 41000 - 10000 - 29000


def test_instrumentation_gap_flag():
    tl = RequestTimeline(req_id=1, cell="B4")
    tl.set("t0", 0.0)
    tl.set("t5", 0.01)   # client span = 10000us
    tl.set("t6", 0.011)
    tl.set("t17", 0.04)  # server span = 29000us
    tl.set("t18", 0.041)
    # No sub-intervals instrumented -> client_local_gap = full client span, server_local_gap = full server span
    acc = account_request(tl)
    assert acc["instrumentation_gap_us"] == pytest.approx(39000.0)
    assert acc["instrumentation_gap_fraction"] > 0.5


def test_metadata_round_trip():
    md = RunMetadata(
        schema_version=1,
        git_commit="abc1234",
        config_hash="cell=B3,N=8",
        cell_id="B3",
        policy_id="FIFO",
        trial_id=0,
    )
    d = md.to_dict()
    assert d["schema_version"] == 1
    assert d["cell_id"] == "B3"


def test_should_flag_gap_both_conditions():
    """Flag only when BOTH fraction > 5% AND absolute > 50us."""
    # Large gap, large fraction -> flag
    big = {"instrumentation_gap_us": 39000.0, "instrumentation_gap_fraction": 0.95}
    assert should_flag_instrumentation_gap(big)
    # Small absolute, large fraction -> no flag (abs < 50us)
    small_abs = {"instrumentation_gap_us": 30.0, "instrumentation_gap_fraction": 0.95}
    assert not should_flag_instrumentation_gap(small_abs)
    # Large absolute, small fraction -> no flag (frac < 5%)
    small_frac = {"instrumentation_gap_us": 39000.0, "instrumentation_gap_fraction": 0.01}
    assert not should_flag_instrumentation_gap(small_frac)
