"""Tests for north-star statistical methods."""
import numpy as np
from common.stats import classify_pair


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
