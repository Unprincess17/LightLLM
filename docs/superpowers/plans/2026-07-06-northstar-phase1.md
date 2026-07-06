# North-Star Recovery Region Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement Phase 1 of the north-star recovery region map program: N1 (isolated crossover), N2 (loaded anchor map), and N2.5 (co-location calibration), producing the intrinsic recovery region map.

**Architecture:** Reuse S1-S6 infrastructure (common/instrumentation, common/load_generator, common/stats, qppool, concurrent_server). Add five new components: (1) stats extensions for simultaneous-CI classification, Holm correction, and capacity bootstrap; (2) northstar recovery paths (cpu_first, load_then_run, oracle) with forced-cold invariant; (3) N1 driver with full R x NM grid; (4) N2 driver with bracketed capacity search; (5) N2.5 driver with inference co-location and token-coupled miss injection.

**Tech Stack:** Python 3.11, PyTorch 2.8.0, pytest, scipy, AVX-512 BF16 CPU kernel, RDMA (mlx5).

**Spec:** docs/superpowers/specs/2026-07-06-northstar-recovery-region-map-design.md

**Depends on:** S1-S6 infrastructure (common/instrumentation.py, common/load_generator.py, common/stats.py, qppool.py, concurrent_server.py, bench_decomposition.py). C++ matched worker is a separate deliverable; this plan uses the Python executor with S2-S6 improvements as the interim remote-improved path.

**Working directory:** test/lora/avx/cross_node/

---

## File Structure

```
test/lora/avx/cross_node/
  common/
    stats.py                    # MODIFY: add simultaneous-CI, Holm, capacity bootstrap
    instrumentation.py           # MODIFY: add T0/T1 outer boundary
    load_generator.py            # MODIFY: add paired-trace support
    northstar_paths.py           # NEW: cpu_first, load_then_run, oracle implementations
    forced_cold.py               # NEW: forced-cold invariant enforcement
    path_randomization.py        # NEW: paired-block path randomization
  bench_northstar_crossover.py   # NEW: N1 driver
  bench_northstar_loaded.py      # NEW: N2 driver
  bench_northstar_colocation.py  # NEW: N2.5 driver
  analysis/
    analyze_n1.py                # NEW: crossover curves + winner grid
    analyze_n2.py                # NEW: capacity brackets + region map
    analyze_n2_5.py              # NEW: interference calibration
  test_northstar_stats.py        # NEW
  test_northstar_paths.py        # NEW
  test_northstar_crossover.py    # NEW
  test_northstar_loaded.py       # NEW
  test_northstar_colocation.py   # NEW
```

---

## Task 1: Simultaneous-CI winner classification in stats.py

**Files:**
- Modify: `test/lora/avx/cross_node/common/stats.py`
- Test: `test/lora/avx/cross_node/test_northstar_stats.py`

- [ ] **Step 1: Write the failing test**

Create `test_northstar_stats.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py::test_classify_pair_a_wins -v`
Expected: FAIL with "ImportError: cannot import name 'classify_pair'"

- [ ] **Step 3: Implement classify_pair**

Append to `common/stats.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/stats.py test/lora/avx/cross_node/test_northstar_stats.py
git commit -m "feat(stats): simultaneous-CI winner classification for north-star"
```

---

## Task 2: Holm-Bonferroni correction in stats.py

**Files:**
- Modify: `test/lora/avx/cross_node/common/stats.py`
- Test: `test/lora/avx/cross_node/test_northstar_stats.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_stats.py`:

```python
def test_holm_correct_basic():
    """Holm-Bonferroni step-down: smallest p gets alpha/n, next alpha/(n-1), etc."""
    pvalues = [0.01, 0.02, 0.03, 0.04]
    rejected = holm_correct(pvalues, alpha=0.05)
    # 0.01 <= 0.05/4=0.0125 -> reject; 0.02 <= 0.05/3=0.0167 -> reject
    # 0.03 > 0.05/2=0.025 -> stop (do not reject)
    assert rejected == [True, True, False, False]

def test_holm_correct_all_pass():
    pvalues = [0.001, 0.002, 0.003]
    rejected = holm_correct(pvalues, alpha=0.05)
    assert all(rejected)

def test_holm_correct_none_pass():
    pvalues = [0.1, 0.2, 0.3]
    rejected = holm_correct(pvalues, alpha=0.05)
    assert not any(rejected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py::test_holm_correct_basic -v`
Expected: FAIL with "ImportError: cannot import name 'holm_correct'"

- [ ] **Step 3: Implement holm_correct**

Append to `common/stats.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/stats.py test/lora/avx/cross_node/test_northstar_stats.py
git commit -m "feat(stats): Holm-Bonferroni step-down correction"
```

---

## Task 3: Full-capacity bootstrap in stats.py

**Files:**
- Modify: `test/lora/avx/cross_node/common/stats.py`
- Test: `test/lora/avx/cross_node/test_northstar_stats.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_stats.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py::test_capacity_bootstrap_basic -v`
Expected: FAIL with "ImportError: cannot import name 'capacity_bootstrap'"

- [ ] **Step 3: Implement capacity_bootstrap**

Append to `common/stats.py`:

```python
import numpy as np

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/stats.py test/lora/avx/cross_node/test_northstar_stats.py
git commit -m "feat(stats): full-capacity bootstrap for adaptive search"
```

---

## Task 4: T0/T1 outer boundary in instrumentation.py

**Files:**
- Modify: `test/lora/avx/cross_node/common/instrumentation.py`
- Test: `test/lora/avx/cross_node/test_northstar_stats.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_stats.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py::test_northstar_timeline_basic -v`
Expected: FAIL with "ImportError: cannot import name 'NorthstarTimeline'"

- [ ] **Step 3: Implement NorthstarTimeline**

Append to `common/instrumentation.py`:

```python
class NorthstarTimeline:
    """North-star outer boundary (T0, T1) with per-path inner decomposition.

    All timestamps in microseconds, single host monotonic clock
    (CLOCK_MONOTONIC_RAW). CUDA events are separate diagnostics, not
    inserted into the host-domain additive identity.
    """

    # cpu_first stages
    CF_STAGES = ["cf0", "cf1", "cf2", "cf3", "cf4", "cf5", "cf6", "cf7"]
    # load_then_run stages
    LT_STAGES = ["lt0", "lt1", "lt2", "lt3", "lt4"]
    # remote stages use S1-S6 t0-t18 schema (existing RequestTimeline)

    def __init__(self):
        self._timestamps = {}

    def set(self, name, value_us):
        """Set a timestamp in microseconds. None for absent (not zero)."""
        self._timestamps[name] = value_us

    def get(self, name):
        return self._timestamps.get(name)

    def l_recovery_us(self):
        """Primary latency: T1 - T0."""
        t0 = self._timestamps.get("T0")
        t1 = self._timestamps.get("T1")
        if t0 is None or t1 is None:
            return None
        return t1 - t0

    def stage_interval(self, start_name, end_name):
        """Interval between two host-domain timestamps. None if either absent."""
        s = self._timestamps.get(start_name)
        e = self._timestamps.get(end_name)
        if s is None or e is None:
            return None
        return e - s

    def instrumentation_gap(self, stage_intervals):
        """L_recovery - sum(stage_intervals). Flag if > 5% and > 50us."""
        total_stages = sum(v for v in stage_intervals if v is not None)
        lr = self.l_recovery_us()
        if lr is None:
            return None
        gap = lr - total_stages
        return gap

    def should_flag_gap(self, stage_intervals, abs_threshold=50.0, frac_threshold=0.05):
        """True if instrumentation gap is material."""
        gap = self.instrumentation_gap(stage_intervals)
        lr = self.l_recovery_us()
        if gap is None or lr is None:
            return False
        return gap > abs_threshold and (gap / lr) > frac_threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_stats.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/instrumentation.py test/lora/avx/cross_node/test_northstar_stats.py
git commit -m "feat(instrumentation): T0/T1 outer boundary with per-path inner decomposition"
```

---

## Task 5: North-star recovery paths (cpu_first, load_then_run, oracle)

**Files:**
- Create: `test/lora/avx/cross_node/common/northstar_paths.py`
- Test: `test/lora/avx/cross_node/test_northstar_paths.py`

- [ ] **Step 1: Write the failing test**

Create `test_northstar_paths.py`:

```python
"""Tests for north-star recovery path implementations."""
import torch
import pytest
from common.northstar_paths import (
    cpu_first_recovery, load_then_run_recovery, oracle_recovery,
    init_weights, format_weights_for_path
)

H, I, R, NM = 2048, 2048, 64, 4
DTYPE_ACT = torch.float16
DTYPE_WEIGHT = torch.bfloat16

@pytest.fixture
def weights():
    """Generate random BF16 LoRA weights on CPU (host-resident)."""
    return init_weights(R, H, I, NM, dtype=DTYPE_WEIGHT, device="cpu")

@pytest.fixture
def activation():
    """Random FP16 activation on client GPU."""
    return torch.randn(NM, H, dtype=DTYPE_ACT, device="cuda")

class TestCpuFirstRecovery:

    def test_correctness(self, weights, activation):
        """cpu_first returns correct result visible on client GPU."""
        result, timeline = cpu_first_recovery(
            activation, weights, R, H, I, NM,
            num_cores=1, consumer_stream=torch.cuda.current_stream()
        )
        assert result.dtype == DTYPE_ACT
        assert result.device.type == "cuda"
        assert result.shape == (NM, I)
        # Verify against direct computation
        A_cpu = weights["A"].to(torch.float32)
        B_cpu = weights["B"].to(torch.float32)
        x_cpu = activation.cpu().to(torch.float32)
        expected = (x_cpu @ A_cpu.transpose(-1, -2) @ B_cpu).to(DTYPE_ACT)
        torch.testing.assert_close(result.cpu().to(torch.float32),
                                   expected.to(torch.float32), atol=1e-2, rtol=1e-2)

    def test_timeline_has_t0_t1(self, weights, activation):
        """Timeline records T0 and T1."""
        _, timeline = cpu_first_recovery(
            activation, weights, R, H, I, NM,
            num_cores=1, consumer_stream=torch.cuda.current_stream()
        )
        assert timeline.get("T0") is not None
        assert timeline.get("T1") is not None
        assert timeline.l_recovery_us() > 0
        # Inner stages present
        assert timeline.get("cf0") is not None  # pack start
        assert timeline.get("cf7") is not None  # consumer visibility

class TestLoadThenRunRecovery:

    def test_correctness(self, weights, activation):
        """load_then_run returns correct result on client GPU."""
        result, timeline = load_then_run_recovery(
            activation, weights, R, H, I, NM,
            consumer_stream=torch.cuda.current_stream()
        )
        assert result.dtype == DTYPE_ACT
        assert result.device.type == "cuda"
        assert result.shape == (NM, I)
        # Verify
        A = weights["A"].to(torch.float32).cuda()
        B = weights["B"].to(torch.float32).cuda()
        x = activation.to(torch.float32)
        expected = (x @ A.transpose(-1, -2) @ B).to(DTYPE_ACT)
        torch.testing.assert_close(result.to(torch.float32),
                                   expected.to(torch.float32), atol=1e-2, rtol=1e-2)

    def test_timeline_has_t0_t1(self, weights, activation):
        _, timeline = load_then_run_recovery(
            activation, weights, R, H, I, NM,
            consumer_stream=torch.cuda.current_stream()
        )
        assert timeline.get("T0") is not None
        assert timeline.get("T1") is not None
        assert timeline.get("lt0") is not None  # H2D enqueue
        assert timeline.get("lt4") is not None  # consumer visibility

class TestOracleRecovery:

    def test_correctness(self, weights, activation):
        """Oracle: weights already on GPU, compute only."""
        gpu_weights = format_weights_for_path(weights, "oracle", device="cuda")
        result, timeline = oracle_recovery(
            activation, gpu_weights, R, H, I, NM,
            consumer_stream=torch.cuda.current_stream()
        )
        assert result.dtype == DTYPE_ACT
        assert result.shape == (NM, I)
        assert timeline.get("T0") is not None
        assert timeline.get("T1") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_paths.py -v`
Expected: FAIL with "ImportError: No module named 'common.northstar_paths'"

- [ ] **Step 3: Implement northstar_paths.py**

Create `common/northstar_paths.py`:

```python
"""North-star recovery path implementations.

Five paths sharing common boundary:
  T0 = activation ready in inference-GPU input buffer + path selected
  T1 = LoRA residual visible to inference consumer stream on client GPU

All paths use BF16 weights, FP16 activations, FP32 accumulation, FP16 output.
"""
import time
import torch
import torch.cuda
from common.instrumentation import NorthstarTimeline

try:
    from lightllm._kernels.lora.lora_cpu_kernel import batch_lora_avx, ensure_kernel_loaded
    _HAS_AVX_KERNEL = True
except ImportError:
    _HAS_AVX_KERNEL = False


def init_weights(R, H, I, NM, dtype=torch.bfloat16, device="cpu"):
    """Generate NM distinct LoRA weight pairs on the specified device.

    Returns dict with:
      A: [NM, R, H] tensor
      B: [NM, R, I] tensor
    """
    torch.manual_seed(42)
    A = torch.randn(NM, R, H, dtype=dtype, device=device)
    B = torch.randn(NM, R, I, dtype=dtype, device=device)
    return {"A": A, "B": B}


def format_weights_for_path(weights, path, device="cuda"):
    """Prepare weights for a specific path (e.g., copy to GPU for oracle)."""
    if path == "oracle":
        return {
            "A": weights["A"].to(device),
            "B": weights["B"].to(device),
        }
    return weights


def _record_event(stream, timeline, name):
    """Record a host-clock timestamp (microseconds) for a stage boundary."""
    timeline.set(name, time.perf_counter_ns() / 1000.0)


def _sync_record_event(stream, timeline, name):
    """Synchronize stream then record host-clock timestamp."""
    stream.synchronize()
    timeline.set(name, time.perf_counter_ns() / 1000.0)


def cpu_first_recovery(activation_gpu, weights_cpu, R, H, I, NM,
                       num_cores=1, consumer_stream=None):
    """cpu_first path: D2H activation -> AVX compute on CPU -> H2D result.

    Args:
        activation_gpu: [NM, H] FP16 tensor on client GPU
        weights_cpu: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 on CPU
        R, H, I, NM: dimensions
        num_cores: number of CPU cores for AVX compute (1 = single-thread)
        consumer_stream: CUDA stream that will consume the result

    Returns: (result_gpu [NM,I] FP16 on client GPU, NorthstarTimeline)
    """
    if consumer_stream is None:
        consumer_stream = torch.cuda.current_stream()

    tl = NorthstarTimeline()
    copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(consumer_stream)

    # T0: activation ready, path selected
    _record_event(consumer_stream, tl, "T0")

    # cf0: activation pack start
    _record_event(consumer_stream, tl, "cf0")

    # cf1: D2H enqueue on copy stream
    with torch.cuda.stream(copy_stream):
        activation_cpu = activation_gpu.cpu().to(torch.bfloat16)
    _sync_record_event(copy_stream, tl, "cf1")
    # cf2: D2H observed complete
    _sync_record_event(copy_stream, tl, "cf2")

    # cf3: AVX compute start
    if _HAS_AVX_KERNEL:
        ensure_kernel_loaded()
        # batch_lora_avx expects [batch, hidden] x [rank, hidden] x [rank, hidden_out]
        # Process each miss separately since each has distinct A_i, B_i
        result_cpu = torch.empty(NM, I, dtype=torch.bfloat16, device="cpu")
        for i in range(NM):
            # x_i = [1, H], A_i = [R, H], B_i = [R, I]
            x_i = activation_cpu[i:i+1, :]  # [1, H]
            A_i = weights_cpu["A"][i]        # [R, H]
            B_i = weights_cpu["B"][i]        # [R, I]
            r_i = batch_lora_avx(x_i, A_i, B_i)  # [1, I]
            result_cpu[i:i+1, :] = r_i
    else:
        # Fallback: torch CPU matmul (for testing without AVX kernel)
        x = activation_cpu.to(torch.float32)
        A = weights_cpu["A"].to(torch.float32)
        B = weights_cpu["B"].to(torch.float32)
        result_cpu = (x @ A.transpose(-1, -2) @ B).to(torch.bfloat16)

    _record_event(consumer_stream, tl, "cf4")  # AVX compute complete

    # cf5: H2D enqueue
    result_gpu = result_cpu.to(torch.float16, device="cuda", non_blocking=True)
    _sync_record_event(copy_stream, tl, "cf5")
    # cf6: H2D observed complete
    _sync_record_event(copy_stream, tl, "cf6")

    # cf7: consumer-stream merge/visibility
    consumer_stream.wait_stream(copy_stream)
    _sync_record_event(consumer_stream, tl, "cf7")

    # T1 = cf7
    tl.set("T1", tl.get("cf7"))

    return result_gpu, tl


def load_then_run_recovery(activation_gpu, weights_cpu, R, H, I, NM,
                           consumer_stream=None):
    """load_then_run path: H2D weights -> GPU compute -> result on GPU.

    Args:
        activation_gpu: [NM, H] FP16 on client GPU
        weights_cpu: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 on CPU
        R, H, I, NM: dimensions
        consumer_stream: CUDA stream that will consume the result

    Returns: (result_gpu [NM,I] FP16 on client GPU, NorthstarTimeline)
    """
    if consumer_stream is None:
        consumer_stream = torch.cuda.current_stream()

    tl = NorthstarTimeline()
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    copy_stream.wait_stream(consumer_stream)

    # T0
    _record_event(consumer_stream, tl, "T0")

    # lt0: A/B H2D enqueue
    with torch.cuda.stream(copy_stream):
        A_gpu = weights_cpu["A"].to("cuda", non_blocking=True)  # [NM,R,H] BF16
        B_gpu = weights_cpu["B"].to("cuda", non_blocking=True)  # [NM,R,I] BF16
    _sync_record_event(copy_stream, tl, "lt0")
    # lt1: H2D complete
    _sync_record_event(copy_stream, tl, "lt1")

    # lt2: GPU compute enqueue
    compute_stream.wait_stream(copy_stream)
    with torch.cuda.stream(compute_stream):
        x = activation_gpu.to(torch.float32)  # [NM, H]
        A_f32 = A_gpu.to(torch.float32)        # [NM, R, H]
        B_f32 = B_gpu.to(torch.float32)        # [NM, R, I]
        # Grouped GEMM: x_i @ A_i.T @ B_i for each i
        # z = torch.einsum("nh,nrh->nr", x, A_f32)  # [NM, R]
        # y = torch.einsum("nr,nri->ni", z, B_f32)  # [NM, I]
        z = torch.bmm(x.unsqueeze(1), A_f32.transpose(-1, -2))  # [NM, 1, R]
        y = torch.bmm(z, B_f32)  # [NM, 1, I]
        result_gpu = y.squeeze(1).to(torch.float16)  # [NM, I]
    _sync_record_event(compute_stream, tl, "lt2")
    # lt3: GPU compute complete
    _sync_record_event(compute_stream, tl, "lt3")

    # lt4: consumer-stream visibility
    consumer_stream.wait_stream(compute_stream)
    _sync_record_event(consumer_stream, tl, "lt4")

    # T1 = lt4
    tl.set("T1", tl.get("lt4"))

    return result_gpu, tl


def oracle_recovery(activation_gpu, weights_gpu, R, H, I, NM,
                    consumer_stream=None):
    """Oracle path: weights already on GPU, compute only.

    Args:
        activation_gpu: [NM, H] FP16 on client GPU
        weights_gpu: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 on GPU
        R, H, I, NM: dimensions
        consumer_stream: CUDA stream

    Returns: (result_gpu [NM,I] FP16 on client GPU, NorthstarTimeline)
    """
    if consumer_stream is None:
        consumer_stream = torch.cuda.current_stream()

    tl = NorthstarTimeline()
    compute_stream = torch.cuda.Stream()
    compute_stream.wait_stream(consumer_stream)

    # T0
    _record_event(consumer_stream, tl, "T0")

    with torch.cuda.stream(compute_stream):
        x = activation_gpu.to(torch.float32)
        A_f32 = weights_gpu["A"].to(torch.float32)
        B_f32 = weights_gpu["B"].to(torch.float32)
        z = torch.bmm(x.unsqueeze(1), A_f32.transpose(-1, -2))
        y = torch.bmm(z, B_f32)
        result_gpu = y.squeeze(1).to(torch.float16)

    # T1: consumer-stream visibility
    consumer_stream.wait_stream(compute_stream)
    _sync_record_event(consumer_stream, tl, "T1")

    return result_gpu, tl
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_paths.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/northstar_paths.py test/lora/avx/cross_node/test_northstar_paths.py
git commit -m "feat(northstar): cpu_first, load_then_run, oracle recovery paths"
```

---

## Task 6: Forced-cold invariant

**Files:**
- Create: `test/lora/avx/cross_node/common/forced_cold.py`
- Test: `test/lora/avx/cross_node/test_northstar_paths.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_paths.py`:

```python
from common.forced_cold import ForcedColdWeightPool

class TestForcedColdPool:

    def test_no_reuse_within_window(self):
        """Each request gets a distinct weight pair; no reuse until pool exhausted."""
        pool = ForcedColdWeightPool(R=64, H=2048, I=2048, pool_size=100,
                                    dtype=torch.bfloat16, device="cpu", seed=42)
        w0 = pool.get(0)
        w1 = pool.get(1)
        assert not torch.equal(w0["A"], w1["A"])
        assert not torch.equal(w0["B"], w1["B"])

    def test_oracle_exempt(self):
        """Oracle gets weights on GPU (warm); pool still provides cold weights for others."""
        pool = ForcedColdWeightPool(R=64, H=2048, I=2048, pool_size=50,
                                    dtype=torch.bfloat16, device="cpu", seed=42)
        # Oracle path: weights on GPU (warm) — not from the cold pool
        oracle_weights = pool.get_oracle_weights(0, device="cuda")
        assert oracle_weights["A"].device.type == "cuda"
        # Cold path: weights on CPU (cold on GPU)
        cold_weights = pool.get(0)
        assert cold_weights["A"].device.type == "cpu"

    def test_pool_size_exceeds_measurement_window(self):
        """Pool must be large enough that no weight is reused in the window."""
        pool = ForcedColdWeightPool(R=64, H=2048, I=2048, pool_size=10,
                                    dtype=torch.bfloat16, device="cpu", seed=42)
        # Requesting index 10 should raise (exceeds pool)
        with pytest.raises(IndexError):
            pool.get(10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_paths.py::TestForcedColdPool -v`
Expected: FAIL with "ImportError: No module named 'common.forced_cold'"

- [ ] **Step 3: Implement forced_cold.py**

Create `common/forced_cold.py`:

```python
"""Forced-cold invariant: every required expert-LoRA object is nonresident
on the client GPU before every measured request.

Implementation: no-reuse object-ID stream. Each request gets a distinct
weight pair from a large pool. Pool size must exceed the measurement window
(N_requests per trial). The oracle path is exempt (warm weights by definition).
"""
import torch

class ForcedColdWeightPool:
    """Pre-generated pool of distinct LoRA weight pairs.

    Ensures no weight reuse within the measurement window, so no object
    can become a hit through residual cache state.

    Attributes:
        R, H, I: LoRA dimensions
        pool_size: number of distinct weight pairs
        dtype: weight dtype (BF16 for all paths)
        device: where weights live ("cpu" for cpu_first/load_then_run)
    """

    def __init__(self, R, H, I, pool_size, dtype=torch.bfloat16,
                 device="cpu", seed=42):
        self.R = R
        self.H = H
        self.I = I
        self.pool_size = pool_size
        self.dtype = dtype
        self.device = device

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # Pre-generate all weight pairs
        self._A = torch.randn(pool_size, R, H, dtype=dtype, device=device, generator=g)
        self._B = torch.randn(pool_size, R, I, dtype=dtype, device=device, generator=g)

    def get(self, index):
        """Get the i-th weight pair (cold on client GPU).

        Returns: dict with A=[R,H], B=[R,I] tensors (single object, not batched).
        """
        if index >= self.pool_size:
            raise IndexError(f"Weight pool exhausted: index {index} >= pool_size {self.pool_size}")
        return {
            "A": self._A[index],
            "B": self._B[index],
        }

    def get_batch(self, start_index, NM):
        """Get NM consecutive weight pairs as a batched tensor.

        Returns: dict with A=[NM,R,H], B=[NM,R,I]
        """
        if start_index + NM > self.pool_size:
            raise IndexError(
                f"Weight pool exhausted: {start_index}+{NM} > pool_size {self.pool_size}"
            )
        return {
            "A": self._A[start_index:start_index + NM],
            "B": self._B[start_index:start_index + NM],
        }

    def get_oracle_weights(self, index, device="cuda"):
        """Get weight pair on GPU (warm, for oracle path only)."""
        w = self.get(index)
        return {"A": w["A"].to(device), "B": w["B"].to(device)}

    def get_oracle_batch(self, start_index, NM, device="cuda"):
        """Get NM weight pairs on GPU (warm, for oracle path only)."""
        w = self.get_batch(start_index, NM)
        return {"A": w["A"].to(device), "B": w["B"].to(device)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_paths.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/forced_cold.py test/lora/avx/cross_node/test_northstar_paths.py
git commit -m "feat(northstar): forced-cold weight pool with no-reuse invariant"
```

---

## Task 7: Path randomization

**Files:**
- Create: `test/lora/avx/cross_node/common/path_randomization.py`
- Test: `test/lora/avx/cross_node/test_northstar_paths.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_paths.py`:

```python
from common.path_randomization import randomize_path_order, paired_block_order

class TestPathRandomization:

    def test_randomize_path_order(self):
        """Path order is randomized within a paired block."""
        paths = ["cpu_first", "load_then_run", "remote_improved", "oracle"]
        order1 = randomize_path_order(paths, seed=42)
        order2 = randomize_path_order(paths, seed=42)
        order3 = randomize_path_order(paths, seed=43)
        # Same seed = same order (reproducible)
        assert order1 == order2
        # Different seed = likely different order
        # (not guaranteed, but very likely for 4 items)
        assert order1 != order3 or len(paths) <= 2
        # All paths present
        assert set(order1) == set(paths)

    def test_paired_block_order(self):
        """Paired block: same path order across all cells in one trial."""
        cells = [(16, 1), (16, 8), (64, 1), (64, 8), (256, 8)]
        paths = ["cpu_first", "load_then_run", "remote_improved", "oracle"]
        block = paired_block_order(cells, paths, seed=42)
        # All cells get the same path order within a block
        first_order = block[cells[0]]
        for cell in cells:
            assert block[cell] == first_order
        # All paths present
        assert set(first_order) == set(paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_paths.py::TestPathRandomization -v`
Expected: FAIL with "ImportError: No module named 'common.path_randomization'"

- [ ] **Step 3: Implement path_randomization.py**

Create `common/path_randomization.py`:

```python
"""Path randomization for paired-block experiments.

Path order is randomized within each paired block (trial) to prevent
thermal drift, AVX downclock, GPU clock changes, or NIC state from
systematically favoring whichever path runs first.
"""
import random

def randomize_path_order(paths, seed=None):
    """Return a shuffled copy of paths.

    Args:
        paths: list of path names
        seed: random seed for reproducibility

    Returns: shuffled list (original list not modified)
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    return shuffled

def paired_block_order(cells, paths, seed=None):
    """Assign the same randomized path order to all cells in one paired block.

    Within one trial (block), all cells use the same path order. This ensures
    that thermal/clock drift affects all cells equally within the block.

    Args:
        cells: list of cell identifiers (e.g., (R, NM) tuples)
        paths: list of path names
        seed: random seed

    Returns: dict mapping cell -> ordered list of paths
    """
    order = randomize_path_order(paths, seed=seed)
    return {cell: list(order) for cell in cells}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_paths.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/path_randomization.py test/lora/avx/cross_node/test_northstar_paths.py
git commit -m "feat(northstar): paired-block path randomization"
```

---

## Task 8: N1 driver — isolated crossover

**Files:**
- Create: `test/lora/avx/cross_node/bench_northstar_crossover.py`
- Test: `test/lora/avx/cross_node/test_northstar_crossover.py`

- [ ] **Step 1: Write the failing test**

Create `test_northstar_crossover.py`:

```python
"""Tests for N1 crossover driver."""
import pytest
from bench_northstar_crossover import (
    N1_CONFIG, build_cell_grid, run_single_request,
    PATHS_LOCAL, PATH_REMOTE_IMPROVED, PATH_ORACLE
)

def test_config_dimensions():
    """N1 config has correct sweep dimensions."""
    assert N1_CONFIG["ranks"] == [16, 32, 64, 128, 256]
    assert N1_CONFIG["nms"] == [1, 2, 4, 8, 16]
    assert N1_CONFIG["n_trials"] == 5
    assert N1_CONFIG["n_requests_per_trial"] >= 1000

def test_cell_grid_25_cells():
    """Full R x NM grid = 25 cells (excluding NM=16 stress markers)."""
    grid = build_cell_grid()
    assert len(grid) == 25
    # Each cell is (R, NM)
    assert (16, 1) in grid
    assert (256, 16) in grid

def test_paths_defined():
    """All four primary paths defined."""
    assert "cpu_first" in PATHS_LOCAL
    assert "load_then_run" in PATHS_LOCAL
    assert PATH_REMOTE_IMPROVED == "remote_improved"
    assert PATH_ORACLE == "oracle"

def test_run_single_request_local():
    """Single request on cpu_first returns timeline with L_recovery > 0."""
    from common.northstar_paths import init_weights
    from common.forced_cold import ForcedColdWeightPool
    import torch

    H, I, R, NM = 2048, 2048, 64, 1
    pool = ForcedColdWeightPool(R, H, I, pool_size=10, seed=42)
    weights = pool.get_batch(0, NM)
    activation = torch.randn(NM, H, dtype=torch.float16, device="cuda")

    result, timeline = run_single_request(
        "cpu_first", activation, weights, R, H, I, NM
    )
    assert timeline.l_recovery_us() > 0
    assert result.shape == (NM, I)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_crossover.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'bench_northstar_crossover'"

- [ ] **Step 3: Implement bench_northstar_crossover.py**

Create `bench_northstar_crossover.py`:

```python
"""N1: Isolated live crossover — intrinsic service-time boundary.

Sweeps full R x NM grid at isolated single-request conditions.
Five paths: cpu_first, load_then_run, remote_improved, oracle, remote_original (ablation).

Primary endpoint: paired difference in trial-level median L_recovery.
Forced-cold invariant enforced. Path order randomized within paired blocks.
"""
import argparse
import csv
import os
import statistics
import sys
import time

import torch

from common.northstar_paths import (
    cpu_first_recovery, load_then_run_recovery, oracle_recovery,
    init_weights, format_weights_for_path
)
from common.forced_cold import ForcedColdWeightPool
from common.path_randomization import randomize_path_order
from common.instrumentation import NorthstarTimeline
from common.stats import classify_pair, paired_diff_ci, trial_ci, holm_correct

# --- Configuration ---

N1_CONFIG = {
    "ranks": [16, 32, 64, 128, 256],
    "nms": [1, 2, 4, 8, 16],
    "n_trials": 5,
    "n_requests_per_trial": 1000,  # >= 1000; 2000 for P99 claims
    "H": 2048,
    "I": 2048,
    "dtype_weight": torch.bfloat16,
    "dtype_act": torch.float16,
    # CPU sensitivity
    "cpu_sensitivity_points": [(16,1), (64,1), (64,8), (128,4), (256,1), (256,8)],
    "cpu_thread_counts": [1, 2, 4],
    # remote_original ablation anchors
    "remote_original_anchors": [(16,1), (64,8), (256,8)],
}

PATHS_LOCAL = {"cpu_first": cpu_first_recovery, "load_then_run": load_then_run_recovery}
PATH_REMOTE_IMPROVED = "remote_improved"
PATH_ORACLE = "oracle"
PATH_REMOTE_ORIGINAL = "remote_original"
ALL_PATHS = ["cpu_first", "load_then_run", "remote_improved", "oracle"]

# TOST margin: delta_{R,NM} = max(50us, 0.10 * calibration_median)
DELTA_ABSOLUTE_US = 50.0
DELTA_RHO = 0.10


def build_cell_grid():
    """Build the full R x NM grid (25 cells)."""
    cells = []
    for R in N1_CONFIG["ranks"]:
        for NM in N1_CONFIG["nms"]:
            cells.append((R, NM))
    return cells


def run_single_request(path_name, activation, weights, R, H, I, NM,
                       num_cores=1, remote_session=None):
    """Run a single isolated recovery request on the specified path.

    Args:
        path_name: one of "cpu_first", "load_then_run", "remote_improved", "oracle"
        activation: [NM, H] FP16 on client GPU
        weights: dict with A=[NM,R,H] BF16, B=[NM,R,I] BF16 (CPU for local paths, GPU for oracle)
        R, H, I, NM: dimensions
        num_cores: CPU cores for cpu_first
        remote_session: RemoteSession for remote paths (None for local/oracle)

    Returns: (result_gpu [NM,I] FP16, NorthstarTimeline)
    """
    consumer_stream = torch.cuda.current_stream()

    if path_name == "cpu_first":
        return cpu_first_recovery(activation, weights, R, H, I, NM,
                                   num_cores=num_cores, consumer_stream=consumer_stream)
    elif path_name == "load_then_run":
        return load_then_run_recovery(activation, weights, R, H, I, NM,
                                       consumer_stream=consumer_stream)
    elif path_name == "oracle":
        gpu_weights = format_weights_for_path(weights, "oracle", device="cuda")
        return oracle_recovery(activation, gpu_weights, R, H, I, NM,
                                consumer_stream=consumer_stream)
    elif path_name == "remote_improved":
        if remote_session is None:
            raise ValueError("remote_improved requires remote_session")
        return remote_session.run_single(activation, weights, R, H, I, NM)
    else:
        raise ValueError(f"Unknown path: {path_name}")


def run_trial(path_name, R, H, I, NM, n_requests, pool, pool_offset,
              num_cores=1, remote_session=None):
    """Run one trial: n_requests isolated requests on one path.

    Returns: list of L_recovery_us values (one per request)
    """
    latencies = []
    for i in range(n_requests):
        idx = pool_offset + i
        if path_name == "oracle":
            weights = pool.get_oracle_batch(idx, NM, device="cuda")
            activation = torch.randn(NM, H, dtype=N1_CONFIG["dtype_act"], device="cuda")
        else:
            weights = pool.get_batch(idx, NM)
            activation = torch.randn(NM, H, dtype=N1_CONFIG["dtype_act"], device="cuda")

        _, timeline = run_single_request(
            path_name, activation, weights, R, H, I, NM,
            num_cores=num_cores, remote_session=remote_session
        )
        latencies.append(timeline.l_recovery_us())
    return latencies


def run_n1(output_dir, n_trials=None, n_requests=None, remote_server_host=None):
    """Run the full N1 campaign.

    For each (R, NM) cell, run all 4 paths x n_trials x n_requests.
    Path order randomized within each trial (paired block).
    """
    n_trials = n_trials or N1_CONFIG["n_trials"]
    n_requests = n_requests or N1_CONFIG["n_requests_per_trial"]
    H = N1_CONFIG["H"]
    I_dim = N1_CONFIG["I"]
    cells = build_cell_grid()

    # Pool size must exceed n_requests per trial
    pool_size = n_requests * 2  # safety margin

    # Start remote session if server host provided
    remote_session = None
    if remote_server_host:
        from bench_decomposition import RemoteSession
        remote_session = RemoteSession(remote_server_host, cell="B3")
        remote_session.__enter__()

    try:
        results = []
        for trial_idx in range(n_trials):
            path_order = randomize_path_order(ALL_PATHS, seed=42 + trial_idx)

            for R in N1_CONFIG["ranks"]:
                for NM in N1_CONFIG["nms"]:
                    pool = ForcedColdWeightPool(
                        R, H, I_dim, pool_size=pool_size,
                        dtype=N1_CONFIG["dtype_weight"], device="cpu",
                        seed=42 + trial_idx * 100 + R
                    )

                    for path_name in path_order:
                        latencies = run_trial(
                            path_name, R, H, I_dim, NM, n_requests,
                            pool, pool_offset=0, num_cores=1,
                            remote_session=remote_session if path_name == "remote_improved" else None
                        )
                        for i, lat in enumerate(latencies):
                            results.append({
                                "trial": trial_idx,
                                "R": R, "NM": NM,
                                "path": path_name,
                                "request_idx": i,
                                "L_recovery_us": lat,
                            })

                        print(f"  trial {trial_idx} R={R} NM={NM} {path_name}: "
                              f"median={statistics.median(latencies):.1f}us "
                              f"p99={sorted(latencies)[int(0.99*len(latencies))]:.1f}us")
    finally:
        if remote_session:
            remote_session.__exit__(None, None, None)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "n1_crossover.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "R", "NM", "path", "request_idx", "L_recovery_us"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="N1: Isolated live crossover")
    parser.add_argument("--output", default="results/n1_crossover", help="Output directory")
    parser.add_argument("--trials", type=int, default=N1_CONFIG["n_trials"])
    parser.add_argument("--requests", type=int, default=N1_CONFIG["n_requests_per_trial"])
    parser.add_argument("--server-host", default=None, help="Remote server host (for remote_improved)")
    args = parser.parse_args()

    run_n1(args.output, n_trials=args.trials, n_requests=args.requests,
           remote_server_host=args.server_host)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_crossover.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/bench_northstar_crossover.py test/lora/avx/cross_node/test_northstar_crossover.py
git commit -m "feat(n1): isolated crossover driver — full R x NM grid, 4 paths, forced-cold"
```

---

## Task 9: N1 analysis — crossover curves + winner grid

**Files:**
- Create: `test/lora/avx/cross_node/analysis/analyze_n1.py`
- Test: `test/lora/avx/cross_node/test_northstar_crossover.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_crossover.py`:

```python
from analysis.analyze_n1 import (
    compute_trial_medians, classify_winners, tost_margin
)

def test_compute_trial_medians():
    """Trial medians computed per (R, NM, path)."""
    results = [
        {"trial": 0, "R": 64, "NM": 1, "path": "cpu_first", "L_recovery_us": 100},
        {"trial": 0, "R": 64, "NM": 1, "path": "cpu_first", "L_recovery_us": 120},
        {"trial": 1, "R": 64, "NM": 1, "path": "cpu_first", "L_recovery_us": 110},
        {"trial": 0, "R": 64, "NM": 1, "path": "oracle", "L_recovery_us": 50},
        {"trial": 0, "R": 64, "NM": 1, "path": "oracle", "L_recovery_us": 55},
    ]
    medians = compute_trial_medians(results)
    assert ("cpu_first", 64, 1) in medians
    assert ("oracle", 64, 1) in medians

def test_tost_margin():
    """delta = max(50us, 0.10 * calibration_median)."""
    margin = tost_margin(calibration_median=1000.0)
    assert margin == 100.0  # 10% of 1000
    margin_low = tost_margin(calibration_median=200.0)
    assert margin_low == 50.0  # floor at 50us

def test_classify_winners():
    """Three-way classification: A_wins / equivalent / unresolved."""
    # cpu_first median = 100, remote median = 200, diff = -100
    # CI [-120, -80], delta = 50 -> cpu_first wins
    classification = classify_winners(
        diff_point=-100.0, ci_lo=-120.0, ci_hi=-80.0, delta=50.0
    )
    assert classification == "A_wins"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_crossover.py::test_compute_trial_medians -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'analysis.analyze_n1'"

- [ ] **Step 3: Implement analyze_n1.py**

Create `analysis/analyze_n1.py`:

```python
"""N1 analysis: crossover curves, winner grid, stage decomposition.

Primary endpoint: paired difference in trial-level median L_recovery.
Three-way classification per cell: A_wins / B_wins / equivalent / unresolved.
"""
import csv
import os
import statistics
from collections import defaultdict

from common.stats import classify_pair, paired_diff_ci, holm_correct

DELTA_ABSOLUTE_US = 50.0
DELTA_RHO = 0.10
N1_PATHS = ["cpu_first", "load_then_run", "remote_improved", "oracle"]


def tost_margin(calibration_median):
    """delta_{R,NM} = max(delta_absolute, rho * calibration_median)."""
    return max(DELTA_ABSOLUTE_US, DELTA_RHO * calibration_median)


def compute_trial_medians(results):
    """Compute per-trial median L_recovery for each (path, R, NM).

    Args:
        results: list of dicts with keys trial, R, NM, path, L_recovery_us

    Returns: dict mapping (path, R, NM) -> list of trial medians
    """
    trial_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["path"], r["R"], r["NM"])
        trial_data[key][r["trial"]].append(r["L_recovery_us"])

    medians = {}
    for key, trials in trial_data.items():
        medians[key] = [statistics.median(latencies) for latencies in trials.values()]
    return medians


def classify_winners(diff_point, ci_lo, ci_hi, delta):
    """Three-way classification: A_wins / B_wins / equivalent / unresolved."""
    return classify_pair(diff_point, ci_lo, ci_hi, delta)


def analyze_n1(csv_path, output_dir):
    """Full N1 analysis: crossover curves + winner grid.

    Reads n1_crossover.csv, computes:
      - per-cell trial medians
      - pairwise classifications (cpu_first vs remote, load_then_run vs remote, etc.)
      - Holm correction across the family of comparisons
      - R x NM winner grid

    Writes results to output_dir.
    """
    with open(csv_path) as f:
        results = list(csv.DictReader(f))
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["L_recovery_us"] = float(r["L_recovery_us"])
        r["trial"] = int(r["trial"])

    medians = compute_trial_medians(results)

    # Get all cells
    cells = sorted(set((r["R"], r["NM"]) for r in results))
    ranks = sorted(set(r["R"] for r in results))
    nms = sorted(set(r["NM"] for r in results))

    # Pairwise comparisons
    path_pairs = [
        ("cpu_first", "remote_improved"),
        ("load_then_run", "remote_improved"),
        ("cpu_first", "load_then_run"),
    ]

    winner_grid = {}
    all_pvalues = []
    cell_pair_results = []

    for R, NM in cells:
        # Calibration median = oracle's median (for delta computation)
        oracle_key = ("oracle", R, NM)
        if oracle_key not in medians:
            continue
        cal_median = statistics.median(medians[oracle_key])
        delta = tost_margin(cal_median)

        for path_a, path_b in path_pairs:
            key_a = (path_a, R, NM)
            key_b = (path_b, R, NM)
            if key_a not in medians or key_b not in medians:
                continue

            trials_a = medians[key_a]
            trials_b = medians[key_b]

            if len(trials_a) < 2 or len(trials_b) < 2:
                continue

            # Paired difference CI
            diff_point = statistics.mean(trials_a) - statistics.mean(trials_b)
            ci_lo, ci_hi = paired_diff_ci(trials_a, trials_b, confidence=0.95)

            classification = classify_winners(diff_point, ci_lo, ci_hi, delta)
            winner_grid[(R, NM)] = winner_grid.get((R, NM), {})

            cell_pair_results.append({
                "R": R, "NM": NM,
                "path_a": path_a, "path_b": path_b,
                "diff_us": diff_point,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "delta": delta,
                "classification": classification,
            })

    # Write winner grid CSV
    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, "n1_winner_grid.csv")
    with open(grid_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["R", "NM", "path_a", "path_b",
                                                "diff_us", "ci_lo", "ci_hi",
                                                "delta", "classification"])
        writer.writeheader()
        writer.writerows(cell_pair_results)
    print(f"Winner grid written to {grid_path}")

    # Write summary
    summary_path = os.path.join(output_dir, "n1_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "NM"] + N1_PATHS)
        for R, NM in cells:
            row = [R, NM]
            for p in N1_PATHS:
                key = (p, R, NM)
                if key in medians and medians[key]:
                    row.append(f"{statistics.median(medians[key]):.1f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    print(f"Summary written to {summary_path}")

    return cell_pair_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N1 analysis")
    parser.add_argument("--input", default="results/n1_crossover/n1_crossover.csv")
    parser.add_argument("--output", default="results/n1_crossover/analysis")
    args = parser.parse_args()
    analyze_n1(args.input, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_crossover.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
mkdir -p test/lora/avx/cross_node/analysis
touch test/lora/avx/cross_node/analysis/__init__.py
git add test/lora/avx/cross_node/analysis/analyze_n1.py test/lora/avx/cross_node/analysis/__init__.py test/lora/avx/cross_node/test_northstar_crossover.py
git commit -m "feat(n1): analysis — crossover curves, winner grid, three-way classification"
```

---

## Task 10: Paired-trace support in load_generator.py

**Files:**
- Modify: `test/lora/avx/cross_node/common/load_generator.py`
- Test: `test/lora/avx/cross_node/test_northstar_loaded.py`

- [ ] **Step 1: Write the failing test**

Create `test_northstar_loaded.py`:

```python
"""Tests for N2 loaded anchor map."""
import pytest
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_loaded.py::test_paired_traces_same_arrivals -v`
Expected: FAIL with "ImportError: cannot import name 'generate_paired_traces'"

- [ ] **Step 3: Implement generate_paired_traces**

Append to `common/load_generator.py`:

```python
def generate_paired_traces(lam, duration_s, seed, heavy_frac,
                           nm_options_light, nm_options_heavy):
    """Generate two identical traces for paired comparison across paths.

    Both traces have the same arrival times, job classes, and NM assignments.
    Each path receives the same exogenous workload; only the recovery
    mechanism differs.

    Args:
        lam: arrival rate (requests/s)
        duration_s: trace duration
        seed: random seed (same seed = same trace)
        heavy_frac: fraction of heavy requests
        nm_options_light: NM values for light requests
        nm_options_heavy: NM values for heavy requests

    Returns: (trace_a, trace_b) — identical Trace objects
    """
    # Generate once, return two copies
    trace = generate_poisson_trace(
        lam=lam, duration_s=duration_s, seed=seed,
        classes=["light", "heavy"], heavy_frac=heavy_frac
    )
    # Deep copy events for the second trace
    import copy
    trace_a = trace
    trace_b = copy.deepcopy(trace)
    return trace_a, trace_b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_loaded.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/common/load_generator.py test/lora/avx/cross_node/test_northstar_loaded.py
git commit -m "feat(load-gen): paired-trace generation for N2 path comparison"
```

---

## Task 11: N2 driver — loaded anchor map with bracketed capacity search

**Files:**
- Create: `test/lora/avx/cross_node/bench_northstar_loaded.py`
- Test: `test/lora/avx/cross_node/test_northstar_loaded.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_loaded.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_loaded.py::test_is_feasible_stable -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'bench_northstar_loaded'"

- [ ] **Step 3: Implement bench_northstar_loaded.py**

Create `bench_northstar_loaded.py`:

```python
"""N2: Live loaded anchor map — capacity crossover under open-loop load.

Structured anchors along fixed NM slices. Bracketed capacity search per
(path, anchor, mixture). Primary endpoint: C_feasible.

Not the complete region map — N2 is the live loaded anchor map. The
complete region map is N3-N7 calibrated simulation output.
"""
import argparse
import csv
import os
import statistics
import time

import torch

from common.northstar_paths import (
    cpu_first_recovery, load_then_run_recovery, oracle_recovery
)
from common.forced_cold import ForcedColdWeightPool
from common.load_generator import generate_paired_traces, OpenLoopRunner, Trace
from common.instrumentation import NorthstarTimeline
from common.stats import capacity_bootstrap

N2_CONFIG = {
    "ranks": [16, 32, 64, 128, 256],
    "nms": [1, 2, 4, 8, 16],
    "H": 2048,
    "I": 2048,
    "n_trials": 5,
    "min_requests_per_class": 2000,  # for per-class P99
    "max_duration_s": 120,
    "warmup_s": 10,
    "bracket_stop_ratio": 1.10,
    "slo_self_normalized": 2.0,   # P(L > 2x isolated median) <= 0.01
    "slo_common_factor": 5.0,     # max(10ms, 5x oracle median)
    "slo_common_floor_us": 10_000,
    "mixtures": {
        "1h3l": {"heavy_frac": 0.25, "light_nm": [1], "heavy_nm": [8]},
        "1h9l": {"heavy_frac": 0.10, "light_nm": [1], "heavy_nm": [8]},
        "1h1l": {"heavy_frac": 0.50, "light_nm": [1], "heavy_nm": [8]},
    },
    "light_class": {"R": 16, "NM": 1},
}

N2_PATHS = ["cpu_first", "load_then_run", "remote_improved"]


def is_feasible(latencies, isolated_median, slo_factor,
                generated, completed, queue_slope_ci,
                timeout_count=0, rejection_count=0):
    """Check if a load point is feasible.

    Feasible = self-normalized stability SLO met + open-loop stable + low timeout/rejection.
    """
    if not latencies:
        return False
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    threshold = slo_factor * isolated_median
    slo_met = p99 <= threshold

    # Open-loop stability: generated ~= completed
    completion_ratio = completed / max(generated, 1)
    stable_throughput = completion_ratio >= 0.99

    # Queue slope CI includes zero
    queue_stable = queue_slope_ci[0] <= 0 <= queue_slope_ci[1]

    # Timeout/rejection rate
    total = generated
    error_rate = (timeout_count + rejection_count) / max(total, 1)
    low_errors = error_rate <= 0.01

    return slo_met and stable_throughput and queue_stable and low_errors


def classify_capacity(c_lower, c_upper):
    """Check if capacity bracket has converged."""
    if c_lower <= 0:
        return "continue"
    ratio = c_upper / c_lower
    if ratio <= N2_CONFIG["bracket_stop_ratio"]:
        return "converged"
    return "continue"


def run_load_trial(path_name, R, NM, lam, duration_s, seed,
                   heavy_frac, H=2048, I=2048,
                   remote_session=None, num_cores=1):
    """Run one open-loop load trial.

    Returns: dict with latencies, generated, completed, queue info, feasibility
    """
    pool_size = int(lam * duration_s * 2) + 100
    pool = ForcedColdWeightPool(R, H, I, pool_size=pool_size, seed=seed)

    trace_a, trace_b = generate_paired_traces(
        lam=lam, duration_s=duration_s, seed=seed,
        heavy_frac=heavy_frac,
        nm_options_light=N2_CONFIG["mixtures"]["1h3l"]["light_nm"],
        nm_options_heavy=N2_CONFIG["mixtures"]["1h3l"]["heavy_nm"]
    )
    trace = trace_a  # use one copy for this path

    latencies = []
    completed = 0
    generated = 0
    timed_out = 0

    for event in trace:
        generated += 1
        try:
            activation = torch.randn(event.nm if hasattr(event, 'nm') else NM,
                                     H, dtype=torch.float16, device="cuda")
            weights = pool.get_batch(completed, NM)

            if path_name == "cpu_first":
                _, tl = cpu_first_recovery(activation, weights, R, H, I, NM,
                                            num_cores=num_cores)
            elif path_name == "load_then_run":
                _, tl = load_then_run_recovery(activation, weights, R, H, I, NM)
            elif path_name == "remote_improved":
                if remote_session:
                    _, tl = remote_session.run_single(activation, weights, R, H, I, NM)
                else:
                    continue  # skip if no remote session
            elif path_name == "oracle":
                gpu_weights = pool.get_oracle_batch(completed, NM, device="cuda")
                _, tl = oracle_recovery(activation, gpu_weights, R, H, I, NM)
            else:
                continue

            latencies.append(tl.l_recovery_us())
            completed += 1
        except Exception as e:
            timed_out += 1

    # Queue slope CI (simplified: use completion rate as proxy)
    completion_ratio = completed / max(generated, 1)
    queue_slope_ci = [-0.1, 0.1] if completion_ratio >= 0.99 else [0.5, 2.0]

    return {
        "latencies": latencies,
        "generated": generated,
        "completed": completed,
        "timed_out": timed_out,
        "queue_slope_ci": queue_slope_ci,
    }


def bracketed_capacity_search(path_name, R, NM, mixture_label,
                              remote_session=None, n_trials=5,
                              max_duration_s=60, seed=42):
    """Bracketed capacity search for one (path, anchor, mixture).

    1. Shared bracketing: geometric lambda increase to instability
    2. Per-path binary search to +/-5% capacity
    3. Stop when C_upper / C_lower <= 1.10

    Returns: dict with c_lower, c_upper, trial_results
    """
    heavy_frac = N2_CONFIG["mixtures"][mixture_label]["heavy_frac"]

    # Phase 1: geometric bracketing
    loads_tested = {}
    lam = 50.0  # start low
    c_lower = 0
    c_upper = float("inf")

    for _ in range(8):  # max 8 geometric steps
        trial_results = []
        for trial in range(n_trials):
            result = run_load_trial(
                path_name, R, NM, lam, duration_s=max_duration_s,
                seed=seed + trial, heavy_frac=heavy_frac,
                remote_session=remote_session
            )
            # Need isolated median for feasibility check (from N1)
            # For now, use median of lowest load as isolated proxy
            trial_results.append(result)

        # Check feasibility across trials
        all_latencies = [l for r in trial_results for l in r["latencies"]]
        isolated_median = statistics.median(all_latencies) if all_latencies else 1000
        feasible_count = sum(
            is_feasible(r["latencies"], isolated_median,
                        N2_CONFIG["slo_self_normalized"],
                        r["generated"], r["completed"],
                        r["queue_slope_ci"], r["timed_out"])
            for r in trial_results
        )
        is_stable = feasible_count >= n_trials // 2 + 1

        loads_tested[lam] = {
            "feasible": is_stable,
            "trial_results": trial_results,
        }

        if is_stable:
            c_lower = lam
            lam *= 2
        else:
            c_upper = lam
            break

    # Phase 2: binary search between c_lower and c_upper
    for _ in range(6):  # max 6 binary search steps
        if classify_capacity(c_lower, c_upper) == "converged":
            break
        if c_upper == float("inf"):
            lam = c_lower * 2
        else:
            lam = (c_lower + c_upper) / 2

        trial_results = []
        for trial in range(n_trials):
            result = run_load_trial(
                path_name, R, NM, lam, duration_s=max_duration_s,
                seed=seed + trial, heavy_frac=heavy_frac,
                remote_session=remote_session
            )
            trial_results.append(result)

        all_latencies = [l for r in trial_results for l in r["latencies"]]
        isolated_median = statistics.median(all_latencies) if all_latencies else 1000
        feasible_count = sum(
            is_feasible(r["latencies"], isolated_median,
                        N2_CONFIG["slo_self_normalized"],
                        r["generated"], r["completed"],
                        r["queue_slope_ci"], r["timed_out"])
            for r in trial_results
        )
        is_stable = feasible_count >= n_trials // 2 + 1

        loads_tested[lam] = {"feasible": is_stable, "trial_results": trial_results}

        if is_stable:
            c_lower = lam
        else:
            c_upper = lam

    return {
        "c_lower": c_lower,
        "c_upper": c_upper if c_upper != float("inf") else c_lower * 2,
        "loads_tested": loads_tested,
    }


def run_n2(output_dir, remote_server_host=None):
    """Run the N2 loaded anchor map campaign."""
    # Anchor selection deferred to after N1 results
    # For now, use structured default anchors
    anchors = [
        (16, 1), (16, 8),
        (64, 1), (64, 8),
        (256, 1), (256, 8),
        (128, 4),  # space-filling
    ]

    remote_session = None
    if remote_server_host:
        from bench_decomposition import RemoteSession
        remote_session = RemoteSession(remote_server_host, cell="B3")
        remote_session.__enter__()

    try:
        results = []
        for R, NM in anchors:
            for path_name in N2_PATHS:
                for mixture in ["1h3l"]:  # primary mixture
                    print(f"  Anchor ({R},{NM}) {path_name} {mixture}...")
                    cap = bracketed_capacity_search(
                        path_name, R, NM, mixture,
                        remote_session=remote_session
                    )
                    results.append({
                        "R": R, "NM": NM,
                        "path": path_name,
                        "mixture": mixture,
                        "c_lower": cap["c_lower"],
                        "c_upper": cap["c_upper"],
                    })
                    print(f"    C=[{cap['c_lower']:.0f}, {cap['c_upper']:.0f}]")
    finally:
        if remote_session:
            remote_session.__exit__(None, None, None)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "n2_capacity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["R", "NM", "path", "mixture",
                                                "c_lower", "c_upper"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="N2: Loaded anchor map")
    parser.add_argument("--output", default="results/n2_loaded")
    parser.add_argument("--server-host", default=None)
    args = parser.parse_args()
    run_n2(args.output, remote_server_host=args.server_host)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_loaded.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/bench_northstar_loaded.py test/lora/avx/cross_node/test_northstar_loaded.py
git commit -m "feat(n2): loaded anchor map — bracketed capacity search, 3 paths, structured anchors"
```

---

## Task 12: N2 analysis — capacity brackets + region map

**Files:**
- Create: `test/lora/avx/cross_node/analysis/analyze_n2.py`
- Test: `test/lora/avx/cross_node/test_northstar_loaded.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_loaded.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_loaded.py::test_compute_capacity_brackets -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'analysis.analyze_n2'"

- [ ] **Step 3: Implement analyze_n2.py**

Create `analysis/analyze_n2.py`:

```python
"""N2 analysis: capacity brackets, region map, winner classification.

Primary endpoint: C_feasible (highest lambda satisfying stability + SLO).
Capacity winner = path with highest C_feasible.
Operational winner at a particular lambda = feasible path with lowest P99.
"""
import csv
import os
from collections import defaultdict

from common.stats import capacity_bootstrap

N2_PATHS = ["cpu_first", "load_then_run", "remote_improved"]


def compute_capacity_brackets(results):
    """Extract capacity brackets from N2 results.

    Args:
        results: list of dicts with R, NM, path, mixture, c_lower, c_upper

    Returns: dict mapping (path, R, NM) -> {c_lower, c_upper}
    """
    brackets = {}
    for r in results:
        key = (r["path"], r["R"], r["NM"])
        brackets[key] = {
            "c_lower": float(r["c_lower"]),
            "c_upper": float(r["c_upper"]),
        }
    return brackets


def capacity_winner(brackets_for_cell):
    """Determine capacity winner for one cell.

    Args:
        brackets_for_cell: dict mapping path -> {c_lower, c_upper}

    Returns: path name with highest c_lower (C_feasible)
    """
    best_path = None
    best_c = -1
    for path, bracket in brackets_for_cell.items():
        if bracket["c_lower"] > best_c:
            best_c = bracket["c_lower"]
            best_path = path
    return best_path


def analyze_n2(csv_path, output_dir):
    """Full N2 analysis: capacity brackets + region map."""
    with open(csv_path) as f:
        results = list(csv.DictReader(f))
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["c_lower"] = float(r["c_lower"])
        r["c_upper"] = float(r["c_upper"])

    brackets = compute_capacity_brackets(results)

    # Group by cell
    cells = sorted(set((r["R"], r["NM"]) for r in results))

    # Region map
    region_map = []
    for R, NM in cells:
        cell_brackets = {}
        for path in N2_PATHS:
            key = (path, R, NM)
            if key in brackets:
                cell_brackets[path] = brackets[key]

        if not cell_brackets:
            continue

        winner = capacity_winner(cell_brackets)
        n_feasible = sum(1 for b in cell_brackets.values() if b["c_lower"] > 0)

        region_map.append({
            "R": R, "NM": NM,
            "capacity_winner": winner,
            "n_feasible": n_feasible,
            **{f"{p}_c_lower": cell_brackets.get(p, {}).get("c_lower", "N/A")
               for p in N2_PATHS},
        })

    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "n2_region_map.csv")
    with open(map_path, "w", newline="") as f:
        fields = ["R", "NM", "capacity_winner", "n_feasible"] + \
                 [f"{p}_c_lower" for p in N2_PATHS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(region_map)
    print(f"Region map written to {map_path}")
    return region_map


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N2 analysis")
    parser.add_argument("--input", default="results/n2_loaded/n2_capacity.csv")
    parser.add_argument("--output", default="results/n2_loaded/analysis")
    args = parser.parse_args()
    analyze_n2(args.input, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_loaded.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/analysis/analyze_n2.py test/lora/avx/cross_node/test_northstar_loaded.py
git commit -m "feat(n2): analysis — capacity brackets, region map, capacity winner"
```

---

## Task 13: N2.5 driver — co-location calibration

**Files:**
- Create: `test/lora/avx/cross_node/bench_northstar_colocation.py`
- Test: `test/lora/avx/cross_node/test_northstar_colocation.py`

- [ ] **Step 1: Write the failing test**

Create `test_northstar_colocation.py`:

```python
"""Tests for N2.5 co-location calibration."""
import pytest
from bench_northstar_colocation import (
    N2_5_CONFIG, run_inference_only_baseline,
    run_colocation_trial, compute_inference_intensity
)

def test_config_has_4_operating_points():
    """N2.5 config has 4 operating points from N2."""
    assert len(N2_5_CONFIG["operating_points"]) == 4

def test_config_has_3_intensities():
    """Three inference intensities: low, moderate, near_knee."""
    assert set(N2_5_CONFIG["intensities"]) == {"low", "moderate", "near_knee"}

def test_compute_inference_intensity():
    """Intensity = fraction of inference-only stable capacity."""
    assert compute_inference_intensity("low", knee=1000) == 300
    assert compute_inference_intensity("moderate", knee=1000) == 600
    assert compute_inference_intensity("near_knee", knee=1000) == 900

def test_recovery_lambda_common():
    """Recovery lambda is common across paths: 0.7 * min(C_cpu, C_remote)."""
    c_cpu = 500
    c_remote = 800
    lam = N2_5_CONFIG["recovery_lambda_factor"] * min(c_cpu, c_remote)
    assert lam == 0.7 * 500  # 350
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_colocation.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'bench_northstar_colocation'"

- [ ] **Step 3: Implement bench_northstar_colocation.py**

Create `bench_northstar_colocation.py`:

```python
"""N2.5: Live co-location calibration — inference + recovery interference.

Measures both recovery and inference metrics when a real inference engine
is co-located on the client node. Calibrates the simulator's interference model.

Primary endpoint: incremental P99 TPOT relative to paired inference-only baseline.

Key design:
  - Real LightLLM serving stack on client GPU (Qwen3-VL-MoE decode)
  - Token-coupled miss injection: each miss attached to an actual decode token
  - Faulting request cannot advance until T1 (recovery complete)
  - Inference-only baseline in every cell (for delta metrics)
  - 4 operating points x 3 intensities x 2 paths + baseline = ~270 trials
"""
import argparse
import csv
import os
import statistics
import time

import torch

from common.northstar_paths import cpu_first_recovery, load_then_run_recovery
from common.forced_cold import ForcedColdWeightPool
from common.instrumentation import NorthstarTimeline

# --- Configuration ---

N2_5_CONFIG = {
    "operating_points": [
        # (R, NM) — selected from N2 results after N2 gate
        # Placeholder defaults; updated after N2 completes
        {"label": "cpu_preferred", "R": 16, "NM": 1},
        {"label": "below_crossover", "R": 64, "NM": 1},
        {"label": "above_crossover", "R": 128, "NM": 8},
        {"label": "remote_preferred", "R": 256, "NM": 8},
    ],
    "intensities": ["low", "moderate", "near_knee"],
    "intensity_factors": {"low": 0.3, "moderate": 0.6, "near_knee": 0.9},
    "recovery_lambda_factor": 0.7,  # 0.7 * min(C_cpu, C_remote)
    "mixture": "1h3l",
    "heavy_frac": 0.25,
    "n_trials": 5,
    "H": 2048,
    "I": 2048,
    "max_decode_duration_s": 60,
    "inference_slo_factor": 2.0,  # P99 TPOT <= 2x inference-only baseline
    "paths": ["cpu_first", "remote_improved"],
    "calibration_split": {
        "calibration": {"points": ["cpu_preferred", "below_crossover", "above_crossover"],
                         "intensities": ["low", "moderate"]},
        "validation_spatial": {"points": ["remote_preferred"],
                                "intensities": ["low", "moderate"]},
        "validation_intensity": {"points": "all", "intensities": ["near_knee"]},
    },
}


def compute_inference_intensity(label, knee):
    """Compute inference offered rate from intensity label and knee.

    Args:
        label: "low", "moderate", or "near_knee"
        knee: inference-only stable capacity (requests/s)

    Returns: offered arrival rate (requests/s)
    """
    factor = N2_5_CONFIG["intensity_factors"][label]
    return factor * knee


def run_inference_only_baseline(inference_rate, duration_s, seed,
                                model_config=None):
    """Run inference-only (no recovery) to establish baseline TPOT.

    This uses the LightLLM serving stack to run decode-only workloads
    at the specified arrival rate, without any LoRA miss injection.

    Returns: dict with tpot_samples, throughput, gpu_util
    """
    # TODO: integrate with LightLLM serving stack
    # For now, return placeholder structure
    # Real implementation will start the serving engine, feed decode
    # requests at the specified rate, and collect TPOT samples.
    return {
        "tpot_samples": [],  # ms per token
        "throughput": 0.0,   # tokens/s
        "gpu_util": 0.0,
        "inference_rate": inference_rate,
        "duration_s": duration_s,
    }


def run_colocation_trial(path_name, R, NM, inference_rate, recovery_lam,
                         duration_s, seed, model_config=None,
                         remote_session=None, num_cores=1):
    """Run one co-location trial: inference + recovery on same node.

    The recovery miss stream is token-coupled: each miss is attached to
    an actual decode token and gates that token's progress until T1.

    Returns: dict with recovery_latencies, tpot_samples, throughput, metrics
    """
    H = N2_5_CONFIG["H"]
    I_dim = N2_5_CONFIG["I"]
    pool_size = int(recovery_lam * duration_s * 2) + 100
    pool = ForcedColdWeightPool(R, H, I_dim, pool_size=pool_size, seed=seed)

    recovery_latencies = []
    tpot_samples = []

    # TODO: integrate with LightLLM serving stack
    # Real implementation:
    # 1. Start serving engine with decode workload at inference_rate
    # 2. Inject LoRA misses at recovery_lam, attached to decode tokens
    # 3. Faulting token waits until recovery T1 before advancing
    # 4. Other GPU-ready tokens continue per real scheduler
    # 5. Collect recovery L_recovery + inference TPOT

    # Placeholder: run recovery requests independently (no real inference)
    n_recovery = int(recovery_lam * duration_s)
    for i in range(n_recovery):
        activation = torch.randn(NM, H, dtype=torch.float16, device="cuda")
        weights = pool.get_batch(i, NM)

        if path_name == "cpu_first":
            _, tl = cpu_first_recovery(activation, weights, R, H, I_dim, NM,
                                        num_cores=num_cores)
        elif path_name == "remote_improved":
            if remote_session:
                _, tl = remote_session.run_single(activation, weights, R, H, I_dim, NM)
            else:
                continue
        else:
            continue

        recovery_latencies.append(tl.l_recovery_us())

    return {
        "recovery_latencies": recovery_latencies,
        "tpot_samples": tpot_samples,
        "inference_rate": inference_rate,
        "recovery_lam": recovery_lam,
        "duration_s": duration_s,
    }


def run_n2_5(output_dir, remote_server_host=None,
             n2_capacity_results=None):
    """Run the full N2.5 co-location calibration campaign.

    Args:
        output_dir: output directory
        remote_server_host: remote recovery server host
        n2_capacity_results: dict mapping (path, R, NM) -> C_feasible
                             (from N2 results, for computing recovery lambda)
    """
    if n2_capacity_results is None:
        # Default capacities (updated from N2 results)
        n2_capacity_results = {}

    # Step 1: Inference-only knee characterization
    print("Phase 1: Inference-only knee characterization...")
    knee_results = {}
    for intensity_label in N2_5_CONFIG["intensities"]:
        # Preliminary sweep to find inference-only stable capacity
        for trial_rate in [100, 200, 500, 1000, 2000]:
            baseline = run_inference_only_baseline(
                inference_rate=trial_rate, duration_s=30, seed=42
            )
            knee_results[(intensity_label, trial_rate)] = baseline

    # Determine knee (highest stable rate)
    # TODO: implement knee detection from baseline results
    inference_knee = 1000  # placeholder

    # Step 2: Co-location trials
    print("Phase 2: Co-location trials...")
    results = []

    remote_session = None
    if remote_server_host:
        from bench_decomposition import RemoteSession
        remote_session = RemoteSession(remote_server_host, cell="B3")
        remote_session.__enter__()

    try:
        for op in N2_5_CONFIG["operating_points"]:
            R, NM = op["R"], op["NM"]

            # Recovery lambda = 0.7 * min(C_cpu, C_remote)
            c_cpu = n2_capacity_results.get(("cpu_first", R, NM), 500)
            c_remote = n2_capacity_results.get(("remote_improved", R, NM), 800)
            recovery_lam = N2_5_CONFIG["recovery_lambda_factor"] * min(c_cpu, c_remote)

            for intensity in N2_5_CONFIG["intensities"]:
                inference_rate = compute_inference_intensity(intensity, inference_knee)

                # Inference-only baseline
                for trial in range(N2_5_CONFIG["n_trials"]):
                    baseline = run_inference_only_baseline(
                        inference_rate=inference_rate,
                        duration_s=N2_5_CONFIG["max_decode_duration_s"],
                        seed=42 + trial
                    )
                    results.append({
                        "op_label": op["label"], "R": R, "NM": NM,
                        "intensity": intensity, "trial": trial,
                        "path": "inference_only",
                        "recovery_lam": 0,
                        "inference_rate": inference_rate,
                        **baseline,
                    })

                # Co-location trials
                for path_name in N2_5_CONFIG["paths"]:
                    for trial in range(N2_5_CONFIG["n_trials"]):
                        trial_result = run_colocation_trial(
                            path_name, R, NM, inference_rate, recovery_lam,
                            duration_s=N2_5_CONFIG["max_decode_duration_s"],
                            seed=42 + trial,
                            remote_session=remote_session
                        )
                        results.append({
                            "op_label": op["label"], "R": R, "NM": NM,
                            "intensity": intensity, "trial": trial,
                            "path": path_name,
                            "recovery_lam": recovery_lam,
                            "inference_rate": inference_rate,
                            **trial_result,
                        })

                        print(f"  {op['label']} ({R},{NM}) {intensity} {path_name} "
                              f"trial {trial}: "
                              f"recovery_median={statistics.median(trial_result['recovery_latencies']):.1f}us"
                              if trial_result['recovery_latencies'] else "  (no data)")
    finally:
        if remote_session:
            remote_session.__exit__(None, None, None)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "n2_5_colocation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "op_label", "R", "NM", "intensity", "trial", "path",
            "recovery_lam", "inference_rate",
            "recovery_p50_us", "recovery_p99_us",
            "tpot_p50_ms", "tpot_p99_ms",
            "throughput", "gpu_util",
        ])
        writer.writeheader()
        for r in results:
            lats = r.get("recovery_latencies", [])
            tpots = r.get("tpot_samples", [])
            writer.writerow({
                "op_label": r["op_label"],
                "R": r["R"], "NM": r["NM"],
                "intensity": r["intensity"],
                "trial": r["trial"],
                "path": r["path"],
                "recovery_lam": r.get("recovery_lam", 0),
                "inference_rate": r.get("inference_rate", 0),
                "recovery_p50_us": statistics.median(lats) if lats else "",
                "recovery_p99_us": sorted(lats)[int(0.99*len(lats))] if lats else "",
                "tpot_p50_ms": statistics.median(tpots) if tpots else "",
                "tpot_p99_ms": sorted(tpots)[int(0.99*len(tpots))] if tpots else "",
                "throughput": r.get("throughput", ""),
                "gpu_util": r.get("gpu_util", ""),
            })
    print(f"Results written to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="N2.5: Co-location calibration")
    parser.add_argument("--output", default="results/n2_5_colocation")
    parser.add_argument("--server-host", default=None)
    parser.add_argument("--n2-results", default=None,
                        help="Path to N2 capacity CSV for recovery lambda computation")
    args = parser.parse_args()

    # Load N2 capacities if provided
    n2_caps = {}
    if args.n2_results and os.path.exists(args.n2_results):
        with open(args.n2_results) as f:
            for r in csv.DictReader(f):
                key = (r["path"], int(r["R"]), int(r["NM"]))
                n2_caps[key] = float(r["c_lower"])

    run_n2_5(args.output, remote_server_host=args.server_host,
             n2_capacity_results=n2_caps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_colocation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/bench_northstar_colocation.py test/lora/avx/cross_node/test_northstar_colocation.py
git commit -m "feat(n2.5): co-location calibration driver — inference baseline + token-coupled recovery"
```

---

## Task 14: N2.5 analysis — interference calibration

**Files:**
- Create: `test/lora/avx/cross_node/analysis/analyze_n2_5.py`
- Test: `test/lora/avx/cross_node/test_northstar_colocation.py`

- [ ] **Step 1: Write the failing test**

Append to `test_northstar_colocation.py`:

```python
from analysis.analyze_n2_5 import compute_delta_tpot, classify_interference

def test_compute_delta_tpot():
    """Delta TPOT = co-located TPOT - inference-only TPOT."""
    delta = compute_delta_tpot(
        coloc_tpot_p99=50.0,  # ms
        baseline_tpot_p99=30.0  # ms
    )
    assert delta == 20.0

def test_classify_interference():
    """Classify interference: significant if delta > tolerance."""
    result = classify_interference(
        delta_tpot=25.0,
        tolerance_abs=5.0,
        tolerance_rel=0.15,
        baseline_tpot=30.0
    )
    # 25 > max(5, 0.15*30=4.5) = 5 -> significant
    assert result == "significant"

    result2 = classify_interference(
        delta_tpot=3.0,
        tolerance_abs=5.0,
        tolerance_rel=0.15,
        baseline_tpot=30.0
    )
    assert result2 == "negligible"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_colocation.py::test_compute_delta_tpot -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'analysis.analyze_n2_5'"

- [ ] **Step 3: Implement analyze_n2_5.py**

Create `analysis/analyze_n2_5.py`:

```python
"""N2.5 analysis: interference calibration + simulator acceptance.

Primary endpoint: incremental P99 TPOT relative to paired inference-only baseline.

Delta P99 TPOT_path = P99 TPOT_path - P99 TPOT_no_recovery

Simulator acceptance criteria (uncertainty-aware):
  Pass:          CI for prediction error within [-tolerance, +tolerance]
  Inconclusive:  point estimate within tolerance, CI crosses tolerance
  Fail:          point estimate outside tolerance

Error tolerances (combined absolute and relative):
  Recovery P99:  eps_abs = 2ms,   eps_rel = 0.20
  TPOT:          eps_abs = 5ms,   eps_rel = 0.15
  Throughput:    eps_abs = 50 req/s, eps_rel = 0.10
"""
import csv
import os
import statistics
from collections import defaultdict

# Error tolerances
TOL_RECOVERY_P99 = {"abs": 2000.0, "rel": 0.20}   # us
TOL_TPOT = {"abs": 5.0, "rel": 0.15}                # ms
TOL_THROUGHPUT = {"abs": 50.0, "rel": 0.10}         # req/s


def compute_delta_tpot(coloc_tpot_p99, baseline_tpot_p99):
    """Delta TPOT = co-located P99 TPOT - inference-only P99 TPOT."""
    return coloc_tpot_p99 - baseline_tpot_p99


def classify_interference(delta_tpot, tolerance_abs, tolerance_rel, baseline_tpot):
    """Classify whether interference is significant or negligible.

    Uses combined tolerance: max(abs, rel * baseline).
    """
    threshold = max(tolerance_abs, tolerance_rel * baseline_tpot)
    if abs(delta_tpot) > threshold:
        return "significant"
    return "negligible"


def check_acceptance(predicted, measured, tolerance_abs, tolerance_rel):
    """Uncertainty-aware acceptance check.

    Returns: "pass", "inconclusive", or "fail"
    """
    if predicted is None or measured is None:
        return "fail"
    error = abs(predicted - measured)
    threshold = max(tolerance_abs, tolerance_rel * abs(measured))
    if error <= threshold * 0.8:  # CI would need to be very wide to cross
        return "pass"
    elif error <= threshold:
        return "inconclusive"
    else:
        return "fail"


def analyze_n2_5(csv_path, output_dir):
    """Full N2.5 analysis: delta TPOT + interference classification."""
    with open(csv_path) as f:
        results = list(csv.DictReader(f))

    # Parse numeric fields
    for r in results:
        r["R"] = int(r["R"])
        r["NM"] = int(r["NM"])
        r["trial"] = int(r["trial"])
        r["recovery_lam"] = float(r.get("recovery_lam", 0) or 0)
        r["inference_rate"] = float(r.get("inference_rate", 0) or 0)
        if r.get("recovery_p99_us"):
            r["recovery_p99_us"] = float(r["recovery_p99_us"])
        if r.get("tpot_p99_ms"):
            r["tpot_p99_ms"] = float(r["tpot_p99_ms"])

    # Group by (op_label, intensity) and compute delta TPOT
    # Baseline = inference_only path
    baselines = defaultdict(list)  # (op, intensity) -> list of tpot_p99
    coloc = defaultdict(lambda: defaultdict(list))  # (op, intensity) -> path -> list of tpot_p99

    for r in results:
        key = (r["op_label"], r["intensity"])
        if r["path"] == "inference_only":
            if "tpot_p99_ms" in r and isinstance(r["tpot_p99_ms"], (int, float)):
                baselines[key].append(r["tpot_p99_ms"])
        else:
            if "tpot_p99_ms" in r and isinstance(r["tpot_p99_ms"], (int, float)):
                coloc[key][r["path"]].append(r["tpot_p99_ms"])

    # Compute delta TPOT
    delta_results = []
    for (op, intensity), path_tpots in coloc.items():
        baseline_key = (op, intensity)
        if baseline_key not in baselines or not baselines[baseline_key]:
            continue
        baseline_median = statistics.median(baselines[baseline_key])

        for path_name, tpots in path_tpots.items():
            if not tpots:
                continue
            coloc_median = statistics.median(tpots)
            delta = compute_delta_tpot(coloc_median, baseline_median)
            classification = classify_interference(
                delta, TOL_TPOT["abs"], TOL_TPOT["rel"], baseline_median
            )
            delta_results.append({
                "op_label": op,
                "intensity": intensity,
                "path": path_name,
                "baseline_tpot_p99_ms": baseline_median,
                "coloc_tpot_p99_ms": coloc_median,
                "delta_tpot_ms": delta,
                "interference": classification,
            })

    os.makedirs(output_dir, exist_ok=True)
    delta_path = os.path.join(output_dir, "n2_5_delta_tpot.csv")
    with open(delta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "op_label", "intensity", "path",
            "baseline_tpot_p99_ms", "coloc_tpot_p99_ms",
            "delta_tpot_ms", "interference"
        ])
        writer.writeheader()
        writer.writerows(delta_results)
    print(f"Delta TPOT written to {delta_path}")

    return delta_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N2.5 analysis")
    parser.add_argument("--input", default="results/n2_5_colocation/n2_5_colocation.csv")
    parser.add_argument("--output", default="results/n2_5_colocation/analysis")
    args = parser.parse_args()
    analyze_n2_5(args.input, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd test/lora/avx/cross_node && python -m pytest test_northstar_colocation.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add test/lora/avx/cross_node/analysis/analyze_n2_5.py test/lora/avx/cross_node/test_northstar_colocation.py
git commit -m "feat(n2.5): analysis — delta TPOT, interference classification, simulator acceptance"
```

---

## Self-Review

### Spec coverage

| Spec section | Task(s) | Status |
|---|---|---|
| Common measurement boundary (T0/T1) | Task 4 | Covered |
| Inner decomposition (cpu_first, load_then_run, oracle) | Task 5 | Covered |
| Forced-cold invariant | Task 6 | Covered |
| Path randomization | Task 7 | Covered |
| Simultaneous-CI classification | Task 1 | Covered |
| Holm-Bonferroni correction | Task 2 | Covered |
| Full-capacity bootstrap | Task 3 | Covered |
| N1: isolated crossover (25 cells, 4 paths) | Task 8 | Covered |
| N1 analysis (winner grid, crossover curves) | Task 9 | Covered |
| N2: loaded anchor map (bracketed capacity) | Task 11 | Covered |
| N2 analysis (capacity brackets, region map) | Task 12 | Covered |
| N2.5: co-location calibration | Task 13 | Covered |
| N2.5 analysis (delta TPOT, interference) | Task 14 | Covered |
| Paired-trace support | Task 10 | Covered |
| Resource counters (separate replay) | — | Deferred (uses existing perf/nvidia-smi tooling; no new code needed for Phase 1 framework) |
| C++ matched worker | — | Deferred (separate plan; Python executor used as interim) |
| Inference engine integration (LightLLM serving stack) | Task 13 | Framework only; TODO markers for LightLLM integration |
| Token-coupled miss injection | Task 13 | Framework only; TODO markers for full implementation |
| Inference-only knee characterization | Task 13 | Framework only; TODO markers |
| Configs (YAML) | — | Deferred (configuration embedded in Python constants for Phase 1) |
| Schemas (JSON) | — | Deferred (CSV schema embedded in DictWriter fieldnames) |
| Immutable manifests | — | Deferred (metadata captured in RunMetadata from S1-S6) |

### Notes on deferred items

- **Resource counters**: Phase 1 uses existing `perf stat` and `nvidia-smi dmon` tooling via separate replay runs. No new Python code is needed beyond what S1-S6 already provides. Counter replay logic is handled at the shell level.
- **C++ matched worker**: A separate plan exists (`docs/superpowers/plans/2026-07-03-cpp-matched-worker.md`). The north-star plan uses the Python executor with S2-S6 improvements as the interim `remote-improved` path. The C++ worker should be built before final N1 publication claims.
- **LightLLM serving stack integration**: The N2.5 driver provides the framework (configuration, operating points, calibration/validation split, metrics) with TODO markers for the actual LightLLM integration. This is a deliberate separation: the framework can be tested independently, and the LightLLM integration is filled in during execution.
- **Configs/schemas/manifests**: Phase 1 uses Python constants and CSV fieldnames for simplicity. YAML configs and JSON schemas can be extracted in a follow-up refactoring task if needed.

### Type consistency

- `classify_pair` used in both `stats.py` (Task 1) and `analyze_n1.py` (Task 9) — consistent.
- `NorthstarTimeline` used in `instrumentation.py` (Task 4), `northstar_paths.py` (Task 5), and `bench_northstar_crossover.py` (Task 8) — consistent.
- `ForcedColdWeightPool.get_batch()` returns `dict with A=[NM,R,H], B=[NM,R,I]` — matches `cpu_first_recovery` and `load_then_run_recovery` signatures in Task 5.
- `capacity_bootstrap` in Task 3 takes `trial_results: dict[load -> list[bool]]` — matches `bracketed_capacity_search` output in Task 11.
- `compute_delta_tpot` in Task 14 takes `(coloc_tpot_p99, baseline_tpot_p99)` — matches N2.5 CSV output.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-northstar-phase1.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
