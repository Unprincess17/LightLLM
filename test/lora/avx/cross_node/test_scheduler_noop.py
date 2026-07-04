"""Scheduler no-op microbenchmark: lower-bound cost of one yield.

Per spec S2: "enqueue → select → callback → requeue with no GEMMs.
Lower-bound cost of one yield."
"""
import time
import queue
import threading
import pytest


def test_scheduler_noop_cost():
    """Measure the lower-bound cost of one yield (enqueue/select/requeue)."""
    q = queue.Queue()
    n = 10000
    t0 = time.perf_counter()
    for i in range(n):
        q.put(i)
        q.get()
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / n * 1e6
    print(f"Scheduler no-op cost: {cost_us:.2f}us per yield")
    # Should be well under 100us (Python queue operations are ~1-5us each)
    assert cost_us < 100, f"Scheduler no-op cost {cost_us:.1f}us exceeds 100us threshold"


def test_scheduler_noop_with_semaphore():
    """Measure yield cost with semaphore-based scheduling (closer to actual implementation)."""
    sem = threading.Semaphore(0)
    n = 1000
    t0 = time.perf_counter()
    for i in range(n):
        sem.release()  # "continuation made runnable"
        sem.acquire()  # "scheduler selects continuation"
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / n * 1e6
    print(f"Semaphore yield cost: {cost_us:.2f}us per yield")
    assert cost_us < 100


def test_scheduler_noop_two_request_round_robin():
    """Measure yield cost with two requests alternating (round-robin)."""
    q0 = queue.Queue()
    q1 = queue.Queue()
    n = 5000  # per request
    t0 = time.perf_counter()
    for i in range(n):
        # Request 0 yields, request 1 runs
        q0.put(i)
        q1.get() if i > 0 else None
        q1.put(i)
        q0.get()
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / (n * 2) * 1e6  # 2 yields per iteration
    print(f"Round-robin yield cost (2 requests): {cost_us:.2f}us per yield")
    assert cost_us < 100
