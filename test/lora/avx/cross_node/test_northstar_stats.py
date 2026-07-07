"""Tests for north-star statistical methods."""
import numpy as np
from common.stats import classify_pair, holm_correct


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
