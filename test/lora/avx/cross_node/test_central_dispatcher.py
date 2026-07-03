"""Test the central admission dispatcher that gates executor admission."""
import threading
import time
import pytest


def test_dispatcher_caps_active_jobs():
    """With active_cap=2, at most 2 jobs run concurrently."""
    from concurrent_server import CentralDispatcher
    disp = CentralDispatcher(active_cap=2)
    active = [0]
    max_active = [0]
    lock = threading.Lock()

    def worker():
        disp.acquire()
        with lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        disp.release()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max_active[0] == 2


def test_dispatcher_records_wait():
    from concurrent_server import CentralDispatcher
    disp = CentralDispatcher(active_cap=1)
    disp.acquire()
    t = threading.Thread(target=lambda: (disp.acquire(), disp.release()))
    t.start()
    time.sleep(0.02)
    disp.release()
    t.join()
    assert len(disp.wait_us) >= 1
    assert disp.wait_us[0] > 0
