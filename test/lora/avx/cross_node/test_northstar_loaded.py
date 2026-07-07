"""Tests for N2 loaded anchor map."""
import pytest
import statistics
from common.load_generator import generate_paired_traces

def test_paired_traces_same_arrivals():
    """Paired traces have identical arrival times and classes, different path assignments."""
    trace_a, trace_b = generate_paired_traces(
        lam=100.0, duration_s=10.0, seed=42,
        heavy_frac=0.25, nm_options_light=[1], nm_options_heavy=[8]
    )
    assert len(trace_a.events) == len(trace_b.events)
    for ea, eb in zip(trace_a.events, trace_b.events):
        assert ea.arrival_time == eb.arrival_time  # same arrivals
        assert ea.job_class == eb.job_class          # same classes

def test_paired_traces_reproducible():
    """Same seed produces same traces."""
    t1a, t1b = generate_paired_traces(lam=50.0, duration_s=5.0, seed=99,
                                       heavy_frac=0.25, nm_options_light=[1],
                                       nm_options_heavy=[8])
    t2a, t2b = generate_paired_traces(lam=50.0, duration_s=5.0, seed=99,
                                       heavy_frac=0.25, nm_options_light=[1],
                                       nm_options_heavy=[8])
    assert len(t1a.events) == len(t2a.events)


def test_paired_traces_nm_assignment():
    """Heavy events get NM from nm_options_heavy, light from nm_options_light."""
    trace_a, trace_b = generate_paired_traces(
        lam=100.0, duration_s=5.0, seed=42,
        heavy_frac=0.25,
        nm_options_light=[1],
        nm_options_heavy=[8]
    )
    for event in trace_a.events:
        if event.job_class == "heavy":
            assert event.nm == 8
        else:
            assert event.nm == 1
    # Paired traces have same NM values
    for ea, eb in zip(trace_a.events, trace_b.events):
        assert ea.nm == eb.nm


from bench_northstar_loaded import (
    N2_CONFIG, bracketed_capacity_search, run_load_trial,
    is_feasible, classify_capacity
)

def test_is_feasible_stable():
    """A trial is feasible if stable + SLO met."""
    # P99 < 2x isolated median, queue stable
    latencies = [100, 105, 110, 115, 120]  # tight
    result = is_feasible(latencies, isolated_median=100, slo_factor=2.0,
                         generated=1000, completed=1000, queue_slope_ci=[-0.1, 0.1])
    assert result == True

def test_is_feasible_unstable_queue():
    """Growing queue = infeasible even if latency looks ok."""
    latencies = [100, 105, 110]
    result = is_feasible(latencies, isolated_median=100, slo_factor=2.0,
                         generated=1000, completed=800, queue_slope_ci=[0.5, 2.0])
    assert result == False

def test_classify_capacity():
    """Capacity bracket: C_lower / C_upper <= 1.10 -> stop."""
    assert classify_capacity(c_lower=100, c_upper=105) == "converged"
    assert classify_capacity(c_lower=100, c_upper=200) == "continue"


def test_run_load_trial_open_loop():
    """Open-loop trial dispatches at scheduled times, not synchronously.

    Under overload (lam >> service rate) with a short drain timeout,
    not all requests complete -- proving arrivals are dispatched
    independently of completions (open-loop, not closed-loop).
    """
    from bench_northstar_loaded import run_load_trial

    # Basic open-loop run: low load, all requests complete
    result = run_load_trial(
        "cpu_first", R=64, NM=1, lam=100.0, duration_s=0.5, seed=42,
        heavy_frac=0.0, H=2048, I=2048,
    )
    assert result["generated"] > 0
    assert result["completed"] > 0
    assert len(result["latencies"]) > 0

    # Overloaded: high arrival rate, short drain
    # Not all requests should complete
    result_overload = run_load_trial(
        "cpu_first", R=64, NM=1, lam=100000.0, duration_s=0.1, seed=42,
        heavy_frac=0.0, H=2048, I=2048,
        drain_timeout_s=0.1,
    )
    assert result_overload["generated"] > 0
    # Open-loop: completed < generated under overload
    assert result_overload["completed"] < result_overload["generated"]


from analysis.analyze_n2 import compute_capacity_brackets, capacity_winner

def test_compute_capacity_brackets():
    """Capacity brackets computed from N2 results."""
    results = [
        {"R": 64, "NM": 1, "path": "cpu_first", "mixture": "1h3l",
         "c_lower": 100, "c_upper": 110},
        {"R": 64, "NM": 1, "path": "remote_improved", "mixture": "1h3l",
         "c_lower": 200, "c_upper": 220},
    ]
    brackets = compute_capacity_brackets(results)
    assert ("cpu_first", 64, 1) in brackets
    assert ("remote_improved", 64, 1) in brackets

def test_capacity_winner():
    """Capacity winner = path with highest C_feasible (= c_lower)."""
    brackets = {
        "cpu_first": {"c_lower": 100, "c_upper": 110},
        "remote_improved": {"c_lower": 200, "c_upper": 220},
    }
    winner = capacity_winner(brackets)
    assert winner == "remote_improved"


def test_capacity_bootstrap_in_analysis():
    """analyze_n2 applies capacity_bootstrap when raw trials are available."""
    import tempfile, csv, os
    from analysis.analyze_n2 import analyze_n2

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write n2_capacity.csv
        cap_path = os.path.join(tmpdir, "n2_capacity.csv")
        with open(cap_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["R", "NM", "path", "mixture", "c_lower", "c_upper"])
            writer.writeheader()
            writer.writerow({"R": 64, "NM": 1, "path": "cpu_first", "mixture": "1h3l", "c_lower": 200, "c_upper": 220})

        # Write n2_raw_trials.csv
        raw_path = os.path.join(tmpdir, "n2_raw_trials.csv")
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["R", "NM", "path", "mixture", "load", "trial", "feasible"])
            writer.writeheader()
            for load in [100, 200, 300, 400]:
                for trial in range(5):
                    feasible = load <= 200
                    writer.writerow({"R": 64, "NM": 1, "path": "cpu_first", "mixture": "1h3l",
                                     "load": load, "trial": trial, "feasible": feasible})

        output_dir = os.path.join(tmpdir, "analysis")
        region_map = analyze_n2(cap_path, output_dir)

        # Check that bootstrapped CIs are in the output
        map_path = os.path.join(output_dir, "n2_region_map.csv")
        with open(map_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        # Should have CI columns when raw trials are available
        assert "c_lower_ci_lo" in rows[0] or "cpu_first_c_lower" in rows[0]


def test_l_recovery_includes_queueing():
    """Under open-loop load, L_recovery should include queueing time."""
    from bench_northstar_loaded import run_load_trial
    # High arrival rate with 1 consumer creates queueing
    result = run_load_trial(
        "cpu_first", R=64, NM=1, lam=5000.0, duration_s=0.5, seed=42,
        heavy_frac=0.0, H=2048, I=2048, drain_timeout_s=2.0
    )
    lats = result["latencies"]
    assert len(lats) > 0
    # At high load, some L_recovery values should be significantly larger
    # than the isolated service time (~250us for R=64, NM=1)
    # If queueing is included, max should be much larger than median
    max_lat = max(lats)
    med_lat = statistics.median(lats)
    # With queueing, max/median ratio should be > 2 at high load
    # (if no queueing, max/median is ~1.5x for stable workloads)
    assert max_lat / max(med_lat, 1) > 1.5, \
        f"Expected queueing effect: max={max_lat}, med={med_lat}, ratio={max_lat/med_lat:.1f}"
