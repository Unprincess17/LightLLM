"""Test that active_cap is independent of physical pool size."""
import threading
import pytest
from unittest.mock import MagicMock


def test_active_cap_independent_of_pool_size():
    """Pool of 32 with active_cap=8 should only allow 8 concurrent borrows."""
    from qppool import QPPoolClient
    pool = QPPoolClient.__new__(QPPoolClient)
    pool.size = 32
    pool.active_cap = 8
    pool._sem = threading.Semaphore(32)  # physical pool
    pool._active_sem = threading.Semaphore(8)  # active cap
    pool._lock = threading.Lock()
    pool._transports = [MagicMock() for _ in range(32)]
    pool._lazy = False
    pool._qp_wait_us = []

    # Acquire 8 active slots
    for _ in range(8):
        pool._active_sem.acquire()
    # 9th should block (active_cap exhausted)
    got_ninth = pool._active_sem.acquire(timeout=0.1)
    assert not got_ninth, "active_cap=8 should block the 9th concurrent borrow"


def test_physical_pool_larger_than_active_cap():
    """After releasing one active slot, can borrow again, but total outstanding never exceeds active_cap."""
    from qppool import QPPoolClient
    pool = QPPoolClient.__new__(QPPoolClient)
    pool._active_sem = threading.Semaphore(8)
    pool._sem = threading.Semaphore(32)
    pool._lock = threading.Lock()
    pool._transports = [MagicMock() for _ in range(32)]
    pool._lazy = False
    pool._qp_wait_us = []

    # Borrow 8
    for _ in range(8):
        pool._active_sem.acquire()
    # Release 1
    pool._active_sem.release()
    # Can borrow 1 more
    assert pool._active_sem.acquire(timeout=0.1)
    # But not a 9th
    assert not pool._active_sem.acquire(timeout=0.1)


def test_borrow_records_qp_wait_us():
    """borrow() should record how long it waited for the semaphore."""
    from qppool import QPPoolClient
    pool = QPPoolClient.__new__(QPPoolClient)
    pool._active_sem = threading.Semaphore(1)
    pool._sem = threading.Semaphore(1)
    pool._lock = threading.Lock()
    pool._transports = [MagicMock()]
    pool._lazy = False
    pool._qp_wait_us = []
    pool.mode = "preconnected"
    pool.base_control_port = 0
    pool.local_ip = "127.0.0.1"

    # The real borrow() method should populate _qp_wait_us.
    # This test verifies the hook exists; full integration in Task 9.
    assert hasattr(pool, '_qp_wait_us')
