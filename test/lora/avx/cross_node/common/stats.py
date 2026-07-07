"""Statistical methods for confound-isolation experiments.

Per spec section "Statistical methods": trial-level CIs (treat trial as unit
for P99), block bootstrap for long runs, TOST for equivalence claims, paired
analysis across repeated traces.
"""
import math
import numpy as np
from typing import Callable, Sequence
from scipy import stats as sp_stats


def trial_ci(values: Sequence[float], confidence: float = 0.95) -> tuple:
    """CI of the mean across trials (t-distribution). Single trial -> point."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(values[0]), float(values[0]))
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(n)
    tcrit = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return (float(mean - tcrit * se), float(mean + tcrit * se))


def percentile_ci(samples: Sequence[float], percentile: float,
                  confidence: float = 0.95, n_bootstrap: int = 2000,
                  rng: np.random.Generator = None) -> tuple:
    """Bootstrap CI for a percentile over pooled request samples."""
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_vals.append(np.percentile(arr[idx], percentile))
    alpha = (1 - confidence) / 2
    return (float(np.percentile(boot_vals, 100 * alpha)),
            float(np.percentile(boot_vals, 100 * (1 - alpha))))


def tost_equivalence(a: Sequence[float], b: Sequence[float], margin: float,
                     confidence: float = 0.95) -> bool:
    """Two One-Sided Tests for equivalence. True if a and b are equivalent
    within +/- margin at the given confidence level."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    na, nb = len(aa), len(bb)
    diff = aa.mean() - bb.mean()
    se = math.sqrt(aa.var(ddof=1) / na + bb.var(ddof=1) / nb)
    if se == 0:
        return abs(diff) <= margin
    tcrit = sp_stats.t.ppf(confidence, df=min(na, nb) - 1)
    lower = diff - tcrit * se
    upper = diff + tcrit * se
    return lower >= -margin and upper <= margin


def paired_diff_ci(a: Sequence[float], b: Sequence[float],
                   confidence: float = 0.95) -> tuple:
    """CI of the paired difference a - b (same trace, different policies)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return trial_ci(d.tolist(), confidence)


def block_bootstrap_ci(samples: Sequence[float], statistic: Callable = np.mean,
                       block_size: int = 50, n_resamples: int = 500,
                       rng: np.random.Generator = None) -> tuple:
    """Block bootstrap CI that respects temporal correlation in long runs."""
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    n_blocks = max(1, n // block_size)
    max_start = n - block_size  # last valid start index for contiguous block
    boot_vals = []
    for _ in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        idx = idx[:n]
        boot_vals.append(statistic(arr[idx]))
    alpha = 0.025  # 95% CI
    return (float(np.percentile(boot_vals, 100 * alpha)),
            float(np.percentile(boot_vals, 100 * (1 - alpha))))


def classify_pair(diff_point, ci_lo, ci_hi, delta):
    """Classify a paired comparison using simultaneous CI.

    Mutually exclusive categories:
      A_wins:      CI lies entirely below -delta
      B_wins:      CI lies entirely above +delta
      equivalent:  CI lies entirely within [-delta, +delta]
      unresolved:  all other cases
    """
    if ci_hi < -delta:
        return "A_wins"
    if ci_lo > delta:
        return "B_wins"
    if ci_lo >= -delta and ci_hi <= delta:
        return "equivalent"
    return "unresolved"


def holm_correct(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction.

    Args:
        pvalues: list of p-values
        alpha: family-wise error rate

    Returns: list of booleans, True if the corresponding null is rejected
    """
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    rejected = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            break
    return rejected


def capacity_bootstrap(trial_results, n_resamples=2000, rng=None):
    """Bootstrap the full adaptive capacity-estimation procedure.

    For each replicate: resample trials at each load, recompute feasibility
    (majority vote), recompute C_lower (highest feasible) and C_upper
    (lowest infeasible).

    Args:
        trial_results: dict mapping load -> list of bool (True=feasible)
        n_resamples: number of bootstrap replicates
        rng: numpy random Generator

    Returns: dict with c_lower/c_upper median and CI bounds
    """
    if rng is None:
        rng = np.random.default_rng()
    loads = sorted(trial_results.keys())

    c_lowers = []
    c_uppers = []
    for _ in range(n_resamples):
        feasible = {}
        for ld in loads:
            trials = trial_results[ld]
            resampled = [trials[i] for i in rng.integers(0, len(trials), size=len(trials))]
            feasible[ld] = sum(resampled) > len(resampled) / 2

        c_lower = 0
        c_upper = float("inf")
        for ld in loads:
            if feasible[ld]:
                c_lower = ld
            else:
                c_upper = ld
                break

        c_lowers.append(c_lower)
        c_uppers.append(c_upper if c_upper != float("inf") else max(loads) * 2)

    return {
        "c_lower_median": float(np.median(c_lowers)),
        "c_lower_ci_lo": float(np.percentile(c_lowers, 2.5)),
        "c_lower_ci_hi": float(np.percentile(c_lowers, 97.5)),
        "c_upper_median": float(np.median(c_uppers)),
        "c_upper_ci_lo": float(np.percentile(c_uppers, 2.5)),
        "c_upper_ci_hi": float(np.percentile(c_uppers, 97.5)),
    }
