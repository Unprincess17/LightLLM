import pytest
import torch
from common.instrumentation import RequestTimeline, RunMetadata, account_request


def test_timeline_set_and_get():
    tl = RequestTimeline(req_id=1)
    tl.set("t0", 100.0)
    tl.set("t18", 150.0)
    assert tl.get("t0") == 100.0
    assert tl.e2e_us() == pytest.approx(50000.0)


def test_timeline_null_persistent_tcp():
    tl = RequestTimeline(req_id=1, cell="B2")
    tl.set("t0", 0.0)
    tl.set("t5", 10.0)
    tl.set("t6", 11.0)
    tl.set("t17", 40.0)
    tl.set("t18", 41.0)
    acc = account_request(tl)
    assert acc["client_request_span_us"] == pytest.approx(10000.0)
    assert acc["server_span_us"] == pytest.approx(29000.0)
    assert acc["cross_domain_residual_us"] == pytest.approx(2000.0)  # 41000 - 10000 - 29000


def test_instrumentation_gap_flag():
    tl = RequestTimeline(req_id=1, cell="B4")
    tl.set("t0", 0.0)
    tl.set("t5", 10.0)   # client span = 10000us
    tl.set("t6", 11.0)
    tl.set("t17", 40.0)  # server span = 29000us
    tl.set("t18", 41.0)
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
