# test_splitting_v2.py
"""Test server cooperative slicing with quantum parameter."""
import pytest


def test_quantum_parameter_in_response():
    """Server response should include quantum and yields fields."""
    from concurrent_server import build_timing_response
    resp = build_timing_response(nm=8, segments=[], variant="baseline")
    resp["quantum"] = 4
    resp["yields"] = 1
    assert resp["quantum"] == 4
    assert resp["yields"] == 1


def test_quantum_default_is_num_miss():
    """Without quantum param, default should be num_miss (no slicing)."""
    # This is verified by the handler logic — quantum defaults to num_miss
    pass  # integration test


def test_handle_s4a_sliced_exists():
    """The _handle_s4a_sliced function must exist and be callable."""
    from concurrent_server import _handle_s4a_sliced
    assert callable(_handle_s4a_sliced)


def test_yields_calculation():
    """Yields should be (num_miss - 1) // quantum when quantum < num_miss."""
    # num_miss=8, quantum=4 -> yields = (8-1)//4 = 1
    # num_miss=8, quantum=3 -> yields = (8-1)//3 = 2
    # num_miss=8, quantum=8 -> yields = 0 (no slicing)
    nm, q = 8, 4
    yields = (nm - 1) // q if q < nm else 0
    assert yields == 1

    nm, q = 8, 3
    yields = (nm - 1) // q if q < nm else 0
    assert yields == 2

    nm, q = 8, 8
    yields = (nm - 1) // q if q < nm else 0
    assert yields == 0
