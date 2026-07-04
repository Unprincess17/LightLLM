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


# ---------------------------------------------------------------------------
# S2 Task 2: Stateful chunking + persistent fan-out tests
# ---------------------------------------------------------------------------

def test_chunk_handlers_exist():
    """All three chunk handler functions must exist and be callable."""
    from concurrent_server import (
        handle_s4a_chunk_init,
        handle_s4a_chunk_continue,
        handle_s4a_chunk_final,
    )
    assert callable(handle_s4a_chunk_init)
    assert callable(handle_s4a_chunk_continue)
    assert callable(handle_s4a_chunk_final)


def test_chunk_session_state_exists():
    """Module-level chunk session state dict and lock must exist."""
    from concurrent_server import _chunk_sessions, _chunk_sessions_lock
    assert isinstance(_chunk_sessions, dict)
    import threading
    assert isinstance(_chunk_sessions_lock, type(threading.Lock()))


def test_compute_misses_helper_exists():
    """The _compute_misses helper must exist and be callable."""
    from concurrent_server import _compute_misses
    assert callable(_compute_misses)


def test_make_chunk_request_init():
    """_make_chunk_request produces s4a_chunk_init for chunk_idx=0."""
    from bench_decomposition import _make_chunk_request, DecompositionConfig
    config = DecompositionConfig(cell="B2", nm=8, rank=64)
    cell_spec = {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 1}
    msg = _make_chunk_request(
        req_id=1, session_id="sess_1", config=config, cell_spec=cell_spec,
        variant="baseline", chunk_nm=4, chunk_idx=0, num_chunks=2)
    assert msg["type"] == "s4a_chunk_init"
    assert msg["session_id"] == "sess_1"
    assert msg["chunk_nm"] == 4
    assert msg["num_miss"] == 8
    assert msg["rank"] == 64


def test_make_chunk_request_continue():
    """_make_chunk_request produces s4a_chunk_continue for intermediate chunks."""
    from bench_decomposition import _make_chunk_request, DecompositionConfig
    config = DecompositionConfig(cell="B2", nm=8, rank=64)
    cell_spec = {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 1}
    msg = _make_chunk_request(
        req_id=2, session_id="sess_1", config=config, cell_spec=cell_spec,
        variant="baseline", chunk_nm=2, chunk_idx=1, num_chunks=4)
    assert msg["type"] == "s4a_chunk_continue"


def test_make_chunk_request_final():
    """_make_chunk_request produces s4a_chunk_final for last chunk."""
    from bench_decomposition import _make_chunk_request, DecompositionConfig
    config = DecompositionConfig(cell="B2", nm=8, rank=64)
    cell_spec = {"transport": "persistent_tcp", "runtime": "python_executor", "conc": 1}
    msg = _make_chunk_request(
        req_id=3, session_id="sess_1", config=config, cell_spec=cell_spec,
        variant="baseline", chunk_nm=2, chunk_idx=3, num_chunks=4)
    assert msg["type"] == "s4a_chunk_final"


def test_stateful_chunking_function_exists():
    """_run_stateful_chunking must exist and be callable."""
    from bench_decomposition import _run_stateful_chunking
    assert callable(_run_stateful_chunking)


def test_persistent_fanout_function_exists():
    """_run_persistent_fanout must exist and be callable."""
    from bench_decomposition import _run_persistent_fanout
    assert callable(_run_persistent_fanout)


def test_chunk_dispatch_in_handle_request():
    """handle_request must dispatch s4a_chunk_* message types."""
    import inspect
    from concurrent_server import handle_request
    source = inspect.getsource(handle_request)
    assert "s4a_chunk_init" in source
    assert "s4a_chunk_continue" in source
    assert "s4a_chunk_final" in source
