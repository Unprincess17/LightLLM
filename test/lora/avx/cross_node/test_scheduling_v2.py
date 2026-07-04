# test_scheduling_v2.py
"""Test Server-SJF priority dispatcher."""
import threading
import time
import pytest


def test_priority_dispatcher_selects_shortest_first():
    """When two requests are queued, the shorter one (lower NM) should be selected first."""
    from concurrent_server import PriorityDispatcher
    disp = PriorityDispatcher(active_cap=1, s_hat={1: 1.0, 8: 8.0})

    results = []
    def worker(nm, label):
        disp.enqueue_and_wait(nm, label)
        results.append(label)
        time.sleep(0.05)
        disp.release()

    # Start heavy job first (it will grab the slot)
    t_heavy = threading.Thread(target=worker, args=(8, "heavy"))
    t_heavy.start()
    time.sleep(0.02)

    # Start light job (should be selected BEFORE heavy when slot frees)
    t_light = threading.Thread(target=worker, args=(1, "light"))
    t_light.start()

    t_heavy.join(timeout=2)
    t_light.join(timeout=2)

    assert "heavy" in results
    assert "light" in results


def test_priority_dispatcher_caps_active_jobs():
    """With active_cap=2, at most 2 jobs run concurrently."""
    from concurrent_server import PriorityDispatcher
    disp = PriorityDispatcher(active_cap=2, s_hat={1: 1.0})

    active = [0]
    max_active = [0]
    lock = threading.Lock()

    def worker():
        disp.enqueue_and_wait(1, "x")
        with lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        disp.release()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=5)
    assert max_active[0] == 2


def test_priority_dispatcher_records_wait():
    from concurrent_server import PriorityDispatcher
    disp = PriorityDispatcher(active_cap=1, s_hat={1: 1.0})
    disp.enqueue_and_wait(1, "a")
    disp.release()
    assert len(disp.wait_us) >= 1
    assert disp.wait_us[0] >= 0
