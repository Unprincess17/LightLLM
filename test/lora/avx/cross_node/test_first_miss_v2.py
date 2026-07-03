# test_first_miss_v2.py
"""Test that the server returns dual-domain timing in the response."""
import pytest
from concurrent_server import build_timing_response


def test_timing_response_has_cpu_and_gpu_fields():
    """Response must include per-segment CPU and GPU timings."""
    resp = build_timing_response(
        nm=8,
        segments=[
            {"name": "alloc", "cpu_us": 5.0, "gpu_us": 3.0},
            {"name": "dtype", "cpu_us": 2.0, "gpu_us": 1.0},
            {"name": "mm1", "cpu_us": 10.0, "gpu_us": 8.0},
            {"name": "mm2", "cpu_us": 12.0, "gpu_us": 9.0},
        ],
        variant="baseline",
    )
    assert resp["nm"] == 8
    assert resp["variant"] == "baseline"
    assert len(resp["segments"]) == 4
    assert resp["segments"][0]["name"] == "alloc"
    assert resp["segments"][0]["cpu_us"] == 5.0
    assert resp["segments"][0]["gpu_us"] == 3.0
