"""Tests for north-star statistical methods."""
import numpy as np
from common.stats import classify_pair, holm_correct, capacity_bootstrap


def test_classify_pair_a_wins():
    """When CI lies entirely below -delta, A practically wins."""
    result = classify_pair(diff_point=-10.0, ci_lo=-12.0, ci_hi=-8.0, delta=5.0)
    assert result == "A_wins"


def test_classify_pair_b_wins():
    """When CI lies entirely above +delta, B practically wins."""
    result = classify_pair(diff_point=10.0, ci_lo=8.0, ci_hi=12.0, delta=5.0)
    assert result == "B_wins"


def test_classify_pair_equivalent():
    """When CI lies entirely within [-delta, +delta], practically equivalent."""
    result = classify_pair(diff_point=0.5, ci_lo=-1.0, ci_hi=2.0, delta=5.0)
    assert result == "equivalent"


def test_classify_pair_unresolved():
    """When CI crosses the delta boundary, unresolved."""
    result = classify_pair(diff_point=4.0, ci_lo=-2.0, ci_hi=10.0, delta=5.0)
    assert result == "unresolved"


def test_holm_correct_basic():
    """Holm-Bonferroni step-down: smallest p gets alpha/n, next alpha/(n-1), etc."""
    pvalues = [0.01, 0.015, 0.03, 0.04]
    rejected = holm_correct(pvalues, alpha=0.05)
    # 0.01  <= 0.05/4=0.0125  -> reject
    # 0.015 <= 0.05/3=0.01667 -> reject
    # 0.03  >  0.05/2=0.025   -> stop (do not reject)
    assert rejected == [True, True, False, False]

def test_holm_correct_all_pass():
    pvalues = [0.001, 0.002, 0.003]
    rejected = holm_correct(pvalues, alpha=0.05)
    assert all(rejected)

def test_holm_correct_none_pass():
    pvalues = [0.1, 0.2, 0.3]
    rejected = holm_correct(pvalues, alpha=0.05)
    assert not any(rejected)


def test_capacity_bootstrap_basic():
    """Capacity bootstrap replays the full adaptive search per replicate."""
    rng = np.random.default_rng(42)
    trial_results = {
        100: [True]*5,
        200: [True]*5,
        300: [True]*4 + [False],
        400: [False]*5,
    }
    brackets = capacity_bootstrap(trial_results, n_resamples=500, rng=rng)
    assert brackets["c_lower_median"] >= 100
    assert brackets["c_upper_median"] <= 500
    assert "c_lower_ci_lo" in brackets
    assert "c_upper_ci_hi" in brackets


from common.instrumentation import NorthstarTimeline

def test_northstar_timeline_basic():
    """T0/T1 outer boundary with L_recovery = T1 - T0."""
    tl = NorthstarTimeline()
    tl.set("T0", 100.0)  # microseconds
    tl.set("T1", 250.0)
    assert tl.l_recovery_us() == 150.0

def test_northstar_timeline_cf_stages():
    """cpu_first inner stages: cf0..cf7, with T1 = cf7."""
    tl = NorthstarTimeline()
    tl.set("T0", 100.0)
    tl.set("cf0", 110.0)  # activation pack start
    tl.set("cf2", 120.0)  # D2H complete
    tl.set("cf4", 180.0)  # AVX compute complete
    tl.set("cf6", 230.0)  # H2D complete
    tl.set("cf7", 240.0)  # consumer-stream visibility
    tl.set("T1", 240.0)   # T1 = cf7
    assert tl.l_recovery_us() == 140.0
    # Stage intervals (host domain)
    assert tl.stage_interval("cf0", "cf2") == 10.0   # D2H
    assert tl.stage_interval("cf2", "cf4") == 60.0   # AVX compute
    assert tl.stage_interval("cf4", "cf6") == 50.0   # H2D

def test_northstar_timeline_lt_stages():
    """load_then_run inner stages: lt0..lt4, with T1 = lt4."""
    tl = NorthstarTimeline()
    tl.set("T0", 100.0)
    tl.set("lt0", 105.0)  # A/B H2D enqueue
    tl.set("lt1", 150.0)  # H2D complete
    tl.set("lt3", 200.0)  # GPU compute complete
    tl.set("lt4", 210.0)  # consumer-stream visibility
    tl.set("T1", 210.0)
    assert tl.l_recovery_us() == 110.0
    assert tl.stage_interval("lt0", "lt1") == 45.0   # H2D
    assert tl.stage_interval("lt1", "lt3") == 50.0   # GPU compute

def test_northstar_timeline_instrumentation_gap():
    """instrumentation_gap = L_recovery - sum(stage_intervals)."""
    tl = NorthstarTimeline()
    tl.set("T0", 100.0)
    tl.set("cf0", 110.0)
    tl.set("cf2", 120.0)
    tl.set("cf4", 180.0)
    tl.set("cf6", 230.0)
    tl.set("cf7", 240.0)
    tl.set("T1", 240.0)
    # L_recovery = 140, stages: cf0-cf2=10, cf2-cf4=60, cf4-cf6=50 = 120
    # gap = 140 - 120 = 20
    gap = tl.instrumentation_gap([10.0, 60.0, 50.0])
    assert gap == 20.0

def test_northstar_timeline_gap_none_when_missing():
    """instrumentation_gap returns None if T0 or T1 absent."""
    tl = NorthstarTimeline()
    tl.set("cf0", 110.0)
    assert tl.instrumentation_gap([10.0]) is None

def test_should_flag_gap_true():
    """Gap flagged when > 50us AND > 5% of L_recovery."""
    tl = NorthstarTimeline()
    tl.set("T0", 0.0)
    tl.set("T1", 100.0)  # L_recovery = 100
    # stages sum to 10, gap = 90 (> 50us and > 5%)
    assert tl.should_flag_gap([10.0]) == True

def test_should_flag_gap_false_small_gap():
    """Gap not flagged when < 50us."""
    tl = NorthstarTimeline()
    tl.set("T0", 0.0)
    tl.set("T1", 100.0)
    # stages sum to 95, gap = 5 (< 50us)
    assert tl.should_flag_gap([95.0]) == False

def test_should_flag_gap_zero_recovery():
    """No crash when L_recovery = 0."""
    tl = NorthstarTimeline()
    tl.set("T0", 100.0)
    tl.set("T1", 100.0)  # L_recovery = 0
    assert tl.should_flag_gap([0.0]) == False
