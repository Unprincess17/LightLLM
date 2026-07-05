# test_heavy_lane_v2.py
"""Test HeavyLaneDispatcher with class-aware heavy sub-cap H."""
import threading, time, pytest

def test_heavy_lane_caps_heavy_concurrency():
    """With N=8, H=2, at most 2 heavy jobs run concurrently."""
    from concurrent_server import HeavyLaneDispatcher
    disp = HeavyLaneDispatcher(active_cap=8, heavy_cap=2)
    active_heavy = [0]
    max_heavy = [0]
    lock = threading.Lock()
    def worker(is_heavy):
        disp.acquire(is_heavy)
        with lock:
            if is_heavy:
                active_heavy[0] += 1
                max_heavy[0] = max(max_heavy[0], active_heavy[0])
        time.sleep(0.05)
        with lock:
            if is_heavy:
                active_heavy[0] -= 1
        disp.release(is_heavy)
    # 6 heavy + 4 light = 10 jobs
    threads = [threading.Thread(target=worker, args=(i < 6,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=5)
    assert max_heavy[0] <= 2

def test_light_bypasses_heavy_cap():
    """Light jobs are not limited by H."""
    from concurrent_server import HeavyLaneDispatcher
    disp = HeavyLaneDispatcher(active_cap=8, heavy_cap=1)
    # All 8 slots can be light even with H=1
    for _ in range(8):
        disp.acquire(False)
    for _ in range(8):
        disp.release(False)

def test_heavy_lane_records_wait():
    from concurrent_server import HeavyLaneDispatcher
    disp = HeavyLaneDispatcher(active_cap=1, heavy_cap=1)
    disp.acquire(True)
    disp.release(True)
    assert len(disp.wait_us) >= 1
    assert disp.wait_us[0] >= 0
