"""Tests for common.stats — statistical methods for confound-isolation experiments."""
import numpy as np
import pytest
from common.stats import (
    trial_ci, percentile_ci, tost_equivalence, paired_diff_ci, block_bootstrap_ci
)


def test_trial_ci_basic():
    trials = [10.0, 12.0, 11.0, 10.5, 11.5]
    lo, hi = trial_ci(trials, confidence=0.95)
    assert lo < 11.0 < hi
    assert hi - lo > 0


def test_trial_ci_single_trial():
    lo, hi = trial_ci([10.0])
    assert lo == 10.0 and hi == 10.0


def test_tost_equivalent():
    a = [10.0, 10.1, 9.9, 10.0, 10.2]
    b = [10.1, 10.0, 10.1, 9.9, 10.0]
    assert tost_equivalence(a, b, margin=1.0, confidence=0.95)


def test_tost_not_equivalent():
    a = [10.0, 10.1, 9.9]
    b = [15.0, 15.1, 14.9]
    assert not tost_equivalence(a, b, margin=1.0, confidence=0.95)


def test_paired_diff_ci():
    a = [10.0, 11.0, 12.0, 10.5, 11.5]
    b = [9.0, 10.0, 11.0, 9.5, 10.5]
    lo, hi = paired_diff_ci(a, b)
    assert lo > 0 and hi > 0  # a consistently > b by 1.0


def test_block_bootstrap_ci():
    rng = np.random.default_rng(42)
    samples = rng.normal(100, 10, size=1000)
    lo, hi = block_bootstrap_ci(samples, statistic=np.mean, block_size=50,
                                n_resamples=500, rng=rng)
    assert lo < 100 < hi
