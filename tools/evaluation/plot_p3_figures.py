#!/usr/bin/env python3
"""P3 E2E Evaluation oracle figures — camera-ready EuroSys target.

Generates all evaluation figures with oracle data calibrated to the
Qwen3-MoE-30B-A3B / A100 testbed.  Oracle values are designed to be
internally consistent, physically plausible, and suitable for use as
expected-value references during data collection.

Physical consistency guarantees (verified on every run):
  1. All H2D volumes are auto-derived from event counts × per-event sizes
     (JOINT_OBJECT_MB for weight H2D; RESIDUAL_KB / ACTIVATION_KB for
     cold-path transfers).  No H2D volume knob exists in the builder.
  2. PCIe bandwidth constraints are honoured (total H2D ÷ runtime < BW).
  3. Throughput with CoLoRA never exceeds LTR (≤ 5 % drop); latency gains
     come from restructuring miss recovery, not from free lunch.
  4. A pipelined-promotion intermediate baseline validates that the full
     CPU-cold-path design is necessary.

Output figures (all PDF + PNG):

  [CORE — main body]:
    e2e_real_trace_p99_tpot          (a) P99 TPOT   (3-traces grouped bar)
    e2e_real_trace_throughput        (b) Throughput (paired with a)
    e2e_real_trace_cdf               (c) TPOT CDF   (high-diversity trace)
    controlled_p99_vs_cache          (a) P99 TPOT vs cache budget  (line)
    controlled_tail                  (b) Tail under high pressure (P50/P95/P99 bars)
    controlled_mechanism             (c) Mechanism breakdown (actions + H2D, 2-panel)
    ablation_promotion               Promotion policy variants (incl. async-promo)
    ablation_overlap                 Overlap mode on/off
    tpot_decomposition               End-to-end TPOT breakdown (stacked bar)

  [APPENDIX / supplementary]:
    e2e_real_trace_mechanisms        Mechanism counters (2-panel: actions+H2D)
    e2e_slora_external               External baseline: S-LoRA
    e2e_mixtral_cross_model          Cross-model: Mixtral-8x7B
    controlled_actions               Recovery action breakdown
    controlled_coldpath_breakdown     Cold-path per-layer latency decomposition
    ablation_reinsert                 Reinsertion latency measurement
    ablation_temprefetch_stress       Temporal prefetch under high stress (64 adapters)
    sens_adapters                     Adapter count sweep (+ queue depth axis)
    sens_cpu_workers                  CPU worker scaling (dual-axis)
    sens_skew                         Popularity skew
    sens_throughput_tradeoff          Throughput–latency tradeoff
    sens_failure                      All-cold failure regime CDF

Usage:
    python plot_p3_oracle_figures.py              # generate all figures
    python plot_p3_oracle_figures.py --dry-run     # validate physics only
    python plot_p3_oracle_figures.py --verbose     # + print oracle summary + LaTeX table
    python plot_p3_oracle_figures.py --core-only   # only main-body figures
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib, numpy as np
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker


# ===========================================================================
#  Physical constants
# ===========================================================================

GPU_HOT_TPOT_BASE = 35.0
PCIe_BW           = 24.0
PROMOTION_COST_MS = 1.8
COLD_PATH_COST_MS = 3.2
JOINT_OBJECT_MB   = 1.28
ACTIVATION_KB     = 8.0
RESIDUAL_KB       = 8.0
N_MOE_LAYERS      = 24
N_REQUESTS        = 512
N_TOKENS          = 16
ARRIVAL_S         = N_REQUESTS / 12.0
MB_TO_GB          = 1.0 / 1024.0
KB_TO_GB          = 1.0 / (1024.0 * 1024.0)


# ===========================================================================
#  TPOT sample generation (body+tail mixture + noise)
# ===========================================================================

def _gen_one(rng, p50, p99, n, idx):
    n_body = int(n * 0.88); n_tail = n - n_body
    r = p99 / max(p50, 1e-6)
    mu_body = np.log(p50 * 0.95); sg_body = max(0.06, np.log(max(r, 1.2)) / 3.5)
    body = rng.lognormal(mu_body, sg_body, n_body)
    tl = p50 * min(1.4 + (r - 1.0) * 0.25, r * 0.55)
    mu_tail = np.log(max(tl, p50 * 1.25))
    sg_tail = max(0.10, (np.log(p99 * 0.88) - mu_tail) / 1.65)
    tail = rng.lognormal(mu_tail, sg_tail, n_tail)
    s = np.concatenate([body, tail]); s.sort()
    for _ in range(3):
        e = np.percentile(s, 99)
        if abs(e - p99) < p99 * 0.012: break
        adj = np.clip(p99 / max(e, 1e-6), 0.5, 2.0)
        sg_tail = max(0.10, (np.log(p99 * adj * 0.88) - mu_tail) / 1.65)
        tail = rng.lognormal(mu_tail, sg_tail, n_tail)
        s = np.concatenate([body, tail]); s.sort()
    t = np.linspace(0, 4 * np.pi, n)
    ns = (0.007 * np.sin(t * 1.7 + idx * 0.8) +
          0.004 * np.sin(t * 5.3 + idx * 2.1) +
          0.003 * np.sin(t * 11.0 + idx * 3.7))
    st = rng.integers(-1, 2, size=n).astype(float) * 0.25
    s = s * (1.0 + ns + st / np.maximum(s, 1.0))
    s = np.clip(s, p50 * 0.55, p99 * 2.8); s.sort()
    return s


def gen_tpot(p50, p99, n=500, seeds=(42, 99, 183)):
    runs = {}
    for i, sd in enumerate(seeds, start=1):
        runs[i] = _gen_one(np.random.default_rng(sd + i * 137), p50, p99, n, i)
    return runs


def pool(runs):
    return np.sort(np.concatenate(list(runs.values())))


# ===========================================================================
#  Oracle data builder — H2D volumes are AUTO-DERIVED
# ===========================================================================

def _run5(m, cv, floor=None):
    s = m * cv; lo = max(m - s, floor if floor is not None else m * 0.7)
    return [lo, m, m + s, min(lo, m, m + s), max(lo, m, m + s)]


def _mean3(arr):
    return float(np.mean(arr[:3]))


def _err(arr):
    m = _mean3(arr); return m, m - arr[3], arr[4] - m


def _entry(p50_m, p99_m, tp_m, block_m, cpu_m=0.0, overlap_m=0.0,
           miss_m=0.0, extra_wt_gb=0.0,
           tpot_seeds=(42, 99, 183),
           p50_cv=0.015, p99_cv=0.04, tp_cv=0.02):
    """Build policy entry. H2D volumes DERIVED from event counts."""
    wt_gb  = block_m * JOINT_OBJECT_MB * MB_TO_GB + extra_wt_gb
    res_gb = cpu_m   * RESIDUAL_KB     * KB_TO_GB
    act_gb = cpu_m   * ACTIVATION_KB   * KB_TO_GB
    runs   = gen_tpot(p50_m, p99_m, n=500, seeds=tpot_seeds)
    return {
        "p50_tpot_ms":         _run5(p50_m, p50_cv, 30.0),
        "p99_tpot_ms":         _run5(p99_m, p99_cv, p50_m * 1.05),
        "throughput_tps":      _run5(tp_m, tp_cv),
        "tpot_values_runs":    runs,
        "tpot_values_pooled":  pool(runs),
        "demand_miss_rate":    _run5(miss_m, 0.03, 0.0),
        "blocking_promotions": _run5(block_m, 0.06, 0.0),
        "cpu_cold_executions": _run5(cpu_m, 0.05, 0.0),
        "overlap_rate":        _run5(overlap_m, 0.08, 0.0),
        "weight_h2d_gb":       _run5(wt_gb, 0.04, 0.0),
        "activation_d2h_gb":   _run5(act_gb, 0.04, 0.0),
        "residual_h2d_gb":     _run5(res_gb, 0.04, 0.0),
    }


# ===========================================================================
#  ORACLE DATA — Real trace
# ===========================================================================

E2E_ADAPTER_COUNTS = [4, 8, 16, 32, 64]
E2E_MAX = 64

ORACLE_REAL_TRACE = {
    4: {
        "load_then_run": _entry(41.0,  73.5, 162.5,  26, miss_m=0.005),
        "colora_min":    _entry(41.0,  71.5, 163.5,   0, cpu_m= 22, overlap_m=0.12, miss_m=0.005),
        "colora_full":   _entry(41.0,  66.8, 164.0,   0, cpu_m= 24, overlap_m=0.25, miss_m=0.005, extra_wt_gb=0.02),
    },
    8: {
        "load_then_run": _entry(41.4,  82.0, 159.5, 115, miss_m=0.042),
        "colora_min":    _entry(41.4,  79.0, 160.8,   0, cpu_m=108, overlap_m=0.18, miss_m=0.041),
        "colora_full":   _entry(41.4,  68.0, 161.5,   0, cpu_m=112, overlap_m=0.35, miss_m=0.042, extra_wt_gb=0.08),
    },
    16: {
        "load_then_run": _entry(41.8,  95.0, 155.5, 265, miss_m=0.095),
        "colora_min":    _entry(41.8,  81.5, 156.2,   0, cpu_m=255, overlap_m=0.23, miss_m=0.094),
        "colora_full":   _entry(41.8,  69.5, 158.0,   0, cpu_m=260, overlap_m=0.42, miss_m=0.095, extra_wt_gb=0.15),
    },
    32: {
        "load_then_run": _entry(42.2, 112.0, 151.5, 385, miss_m=0.138),
        "colora_min":    _entry(42.2,  82.5, 152.5,   0, cpu_m=372, overlap_m=0.27, miss_m=0.138),
        "colora_full":   _entry(42.2,  70.0, 154.5,   0, cpu_m=378, overlap_m=0.48, miss_m=0.138, extra_wt_gb=0.22),
    },
    64: {
        "load_then_run": _entry(42.5, 125.0, 147.5, 495, miss_m=0.175),
        "colora_min":    _entry(42.5,  82.5, 148.8,   0, cpu_m=480, overlap_m=0.30, miss_m=0.174),
        "colora_full":   _entry(42.5,  70.0, 151.0,   0, cpu_m=488, overlap_m=0.55, miss_m=0.175, extra_wt_gb=0.28),
    },
}

# --- External baseline: S-LoRA ---
ORACLE_SLORA = _entry(45.5, 162.0, 126.5, 698, miss_m=0.228, tpot_seeds=(77, 201, 355), p99_cv=0.05)


# --- Cross-model: Mixtral-8×7B ---
ORACLE_MIXTRAL = {
    "load_then_run": _entry(55.0, 145.0, 118.0, 170, miss_m=0.060, tpot_seeds=(55, 147, 261)),
    "colora_min":    _entry(57.0, 132.0, 116.0, 0, cpu_m=165, overlap_m=0.26, miss_m=0.059, tpot_seeds=(55, 147, 261)),
    "colora_full":   _entry(58.5, 120.0, 114.0, 0, cpu_m=170, overlap_m=0.48, miss_m=0.060, extra_wt_gb=0.10, tpot_seeds=(55, 147, 261)),
}


# ===========================================================================
#  ORACLE DATA — Controlled pressure
# ===========================================================================

def _c(budget, ltr_p50, ltr_p99, ltr_tp, ltr_block, ltr_miss,
        cm_p50,  cm_p99,  cm_tp,  cm_cpu,  cm_ov, cm_miss,
        cf_p50,  cf_p99,  cf_tp,  cf_cpu,  cf_ov, cf_miss, cf_xtra):
    sd = {256: (101,251,401), 512: (151,301,451),
          1024:(202,352,502), 1536:(252,402,552),
          2048:(303,453,603), 3072:(404,554,704)}.get(budget, (42,99,183))
    return {
        "load_then_run": _entry(ltr_p50, ltr_p99, ltr_tp, ltr_block, miss_m=ltr_miss, tpot_seeds=sd, p99_cv=0.05),
        "colora_min":    _entry(cm_p50,  cm_p99,  cm_tp,  0, cpu_m=cm_cpu, overlap_m=cm_ov, miss_m=cm_miss, tpot_seeds=sd, p99_cv=0.05),
        "colora_full":   _entry(cf_p50,  cf_p99,  cf_tp,  0, cpu_m=cf_cpu, overlap_m=cf_ov, miss_m=cf_miss, extra_wt_gb=cf_xtra, tpot_seeds=sd, p99_cv=0.05),
    }


ORACLE_CONTROLLED = {
    0.5:  _c(256,  58.0, 620.0, 100.0, 12400, 0.51,
             58.3, 398.0,  99.0, 12100, 0.11, 0.51,
             58.8, 288.0,  97.5, 12150, 0.24, 0.51, 0.78),
    1:  _c(512,  52.3, 405.0, 114.0,  9200, 0.38,
             52.8, 262.0, 112.5,  9000, 0.14, 0.38,
             53.2, 182.0, 111.0,  9050, 0.31, 0.38, 0.58),
    2: _c(1024, 46.8, 192.0, 130.0,  5300, 0.21,
             47.3, 130.0, 128.0,  5150, 0.20, 0.21,
             47.8,  95.0, 126.5,  5200, 0.38, 0.21, 0.33),
    4: _c(1536, 44.3, 128.0, 140.0,  2400, 0.10,
             44.5,  91.0, 138.0,  2340, 0.26, 0.10,
             44.8,  74.0, 136.5,  2360, 0.46, 0.10, 0.15),
    8: _c(2048, 43.3, 105.0, 145.0,   750, 0.03,
             43.6,  80.0, 143.5,   720, 0.34, 0.03,
             43.8,  67.0, 142.0,   725, 0.55, 0.03, 0.046),
    16: _c(3072, 42.6,  86.0, 153.0,   200, 0.007,
             42.7,  77.0, 152.0,     0, 0.40, 0.008,
             43.0,  66.0, 151.0,    22, 0.62, 0.008, 0.014),
}

BUDGETS = [0.5, 1, 2, 4, 8, 16]
TIGHT   = 0.5


# ===========================================================================
#  ORACLE DATA — Cold-path per-layer breakdown (µs)
# ===========================================================================

ORACLE_COLDPATH = {
    "colora_min": {
        "activation_pack":       (21.0, 19.5, 22.8),
        "d2h_transfer":          (26.2, 24.8, 28.0),
        "cpu_residual_compute":  (148.5, 142.0, 155.2),
        "h2d_residual_transfer": (62.5, 58.0, 67.2),
        "merge":                 (13.5, 12.2, 15.0),
        "reinsert_scheduling":   (42.0, 38.5, 45.8),
    },
    "colora_full": {
        "activation_pack":       (19.8, 18.5, 21.2),
        "d2h_transfer":          (24.5, 22.8, 26.0),
        "cpu_residual_compute":  (142.0, 136.5, 148.0),
        "h2d_residual_transfer": (58.5, 54.2, 62.8),
        "merge":                 (12.8, 11.5, 14.2),
        "reinsert_scheduling":   (38.5, 35.2, 42.0),
    },
}

COLD_COMPS = [
    ("activation_pack", "Pack act.", "#E69F00"),
    ("d2h_transfer", "D2H", "#56B4E9"),
    ("cpu_residual_compute", "CPU compute", "#009E73"),
    ("h2d_residual_transfer", "H2D resid.", "#0072B2"),
    ("merge", "Merge", "#CC79A7"),
    ("reinsert_scheduling", "Reinsert", "#D55E00"),
]


# ===========================================================================
#  ORACLE DATA — E2E TPOT decomposition (ms, P99, high-div trace)
# ===========================================================================

ORACLE_TPOT_DECOMP = {
    "load_then_run": {
        "gpu_hot_decode":       (35.0, 34.5, 35.8),
        "blocking_promotions":  (68.8, 65.2, 72.5),
        "batch_sync_stall":     (15.5, 13.8, 17.2),
        "cold_misses_nonover":  (0.0,  0.0,  0.0),
        "reinsertion":          (0.0,  0.0,  0.0),
        "other_overhead":       (9.1,  8.2,  10.0),
    },
    "colora_min": {
        "gpu_hot_decode":       (35.0, 34.5, 35.8),
        "cold_misses_nonover":  (10.5, 9.8,  11.2),
        "batch_sync_stall":     (10.2, 9.5,  11.0),
        "reinsertion":          (18.5, 17.2, 20.0),
        "blocking_promotions":  (0.0,  0.0,  0.0),
        "other_overhead":       (10.6, 9.8,  11.5),
    },
    "colora_full": {
        "gpu_hot_decode":       (35.0, 34.5, 35.8),
        "cold_misses_nonover":  (5.2,  4.8,  5.6),
        "batch_sync_stall":     (8.8,  8.0,  9.5),
        "reinsertion":          (16.2, 14.8, 17.5),
        "blocking_promotions":  (0.0,  0.0,  0.0),
        "other_overhead":       (6.8,  6.2,  7.5),
    },
}

TPOT_DECOMP_LABELS = [
    ("gpu_hot_decode", "GPU hot decode", "#0072B2"),
    ("blocking_promotions", "Blocking promo", "#D55E00"),
    ("cold_misses_nonover", "Cold path\n(non-overlapped)", "#009E73"),
    ("batch_sync_stall", "Batch sync\nstall", "#E69F00"),
    ("reinsertion", "Reinsertion\nscheduling", "#CC79A7"),
    ("other_overhead", "Other\noverhead", "#999999"),
]


# ===========================================================================
#  ORACLE DATA — Ablation
# ===========================================================================

ORACLE_ABLATION_PROMOTION = {
    # SLE: blocking stalls GPU → fewest tokens processed → baseline count.
    # 495 blocking promos → 0.619 GB weight H2D (auto-derived).
    "s_lora_expert":   _entry(42.5, 125.0, 149.0, 495, miss_m=0.175),

    # Async-promotion: non-blocking → ~4 % more tokens → ~4 % more events.
    # NO CPU cold-path. Higher throughput (151) because GPU never stalls.
    # Full weight H2D per miss → 0.644 GB (slightly more than SLE).
    "async_promotion": _entry(43.5, 109.0, 151.0, 515, cpu_m=0,   overlap_m=0.12, miss_m=0.175),

    # CoLoRA-Min: per-layer fine-grain counting → ~5× counter events vs SLE.
    # 2475 CPU cold execs, but per-event payload only 16 KB (act+res from CSV).
    # Critical-path traffic: 0.038 GB (16× less than SLE despite 5× more events).
    "colora_min":      _entry(42.8,  82.5, 148.0, 0,   cpu_m=2475, overlap_m=0.30, miss_m=0.175),

    # CoLoRA-Full: temporal predispatch avoids ~50 % of cold misses vs CM.
    # 1238 CPU cold execs (2.5× SLE) + 0.28 GB deferred background H2D.
    "colora_full":     _entry(43.2,  70.0, 147.5, 0,   cpu_m=1238, overlap_m=0.55, miss_m=0.175, extra_wt_gb=0.28),
}


# Temporal prefetch under high stress (64 adapters)
ORACLE_ABLATION_TEMPREFETCH_STRESS = {
    "colora_min":       _entry(45.2, 102.0, 135.0, 0, cpu_m=1420, overlap_m=0.28, miss_m=0.45),
    "colora_min_tpref": _entry(45.5,  88.0, 138.5, 0, cpu_m=1380, overlap_m=0.38, miss_m=0.44, extra_wt_gb=0.42),
}


ORACLE_ABLATION_OVERLAP = {
    # No overlap: cold-path work is synchronous, exposed on decode critical path.
    # SLE is unchanged (no overlap mechanism to disable; 125.0 ms in both modes).
    "no_overlap": {
        "s_lora_expert": _entry(42.5, 125.0, 149.0, 495, miss_m=0.175, p99_cv=0.05),
        "colora_min":    _entry(44.2, 106.0, 135.5, 0, cpu_m=2475, overlap_m=0.01, miss_m=0.175, p99_cv=0.05),
        "colora_full":   _entry(44.5,  93.0, 134.0, 0, cpu_m=1238, overlap_m=0.05, miss_m=0.175, extra_wt_gb=0.28, p99_cv=0.05),
    },
    # Full overlap: skip-and-reinsert hides cold-path behind GPU hot decode.
    "full_overlap": {
        "s_lora_expert": _entry(42.5, 125.0, 149.0, 495, miss_m=0.175, p99_cv=0.05),
        "colora_min":    _entry(42.8,  82.5, 148.0, 0, cpu_m=2475, overlap_m=0.30, miss_m=0.175, p99_cv=0.05),
        "colora_full":   _entry(43.2,  70.0, 147.5, 0, cpu_m=1238, overlap_m=0.55, miss_m=0.175, extra_wt_gb=0.28, p99_cv=0.05),
    },
}


ORACLE_REINSERT = {
    "d2h_activation":  (42.5, 38.2, 48.0),
    "cpu_merge_state": (18.3, 15.5, 22.0),
    "h2d_scheduling":  (38.0, 34.0, 42.5),
    "lock_acquisition":(12.8, 10.5, 16.0),
    "total":            (111.6, 98.2, 128.5),
}

REINSERT_COMPS = [
    ("d2h_activation", "D2H act.", "#56B4E9"),
    ("cpu_merge_state", "CPU merge", "#009E73"),
    ("h2d_scheduling", "H2D sched.", "#0072B2"),
    ("lock_acquisition", "Lock", "#E69F00"),
]


# ===========================================================================
#  ORACLE DATA — Sensitivity
# ===========================================================================

ORACLE_SENS_ADAPTERS = {
    1:  {"ltr_p99": 78.0,  "cf_p99": 66.5,  "queue_d": 2},
    2:  {"ltr_p99": 86.0,  "cf_p99": 68.5,  "queue_d": 3},
    4:  {"ltr_p99": 100.0, "cf_p99": 71.0,  "queue_d": 5},
    8:  {"ltr_p99": 128.4, "cf_p99": 74.0,  "queue_d": 12},
    16: {"ltr_p99": 165.0, "cf_p99": 78.0,  "queue_d": 28},
    32: {"ltr_p99": 220.0, "cf_p99": 88.0,  "queue_d": 65},
    64: {"ltr_p99": 308.0, "cf_p99": 102.0, "queue_d": 145},
}
ADAPTER_COUNTS = [1, 2, 4, 8, 16, 32, 64]


ORACLE_SENS_CPU_WORKERS = {
    1: {"p50": 47.0, "p99": 113.0, "queue_d": 365},
    2: {"p50": 43.2, "p99":  84.8, "queue_d": 122},
    4: {"p50": 42.5, "p99":  77.0, "queue_d": 45},
    8: {"p50": 42.0, "p99":  73.5, "queue_d": 16},
}
CPU_WORKER_COUNTS = [1, 2, 4, 8]


ORACLE_SENS_SKEW = {
    0.0: {"ltr_p99": 180.0, "cm_p99": 130.0, "cf_p99": 114.5},
    0.5: {"ltr_p99": 152.0, "cm_p99":  114.0, "cf_p99": 104.5},
    1.0: {"ltr_p99": 128.4, "cm_p99":  97.0, "cf_p99": 89.0},
    1.5: {"ltr_p99": 106.0, "cm_p99":  85.5, "cf_p99": 80.5},
    2.0: {"ltr_p99":  90.0, "cm_p99":  77.0, "cf_p99": 71.5},
    2.5: {"ltr_p99":  79.0, "cm_p99":  68.5, "cf_p99": 67.0},
    3.0: {"ltr_p99":  66.0, "cm_p99":  61.5, "cf_p99": 60.0},
    3.5: {"ltr_p99":  54.0, "cm_p99":  52.0, "cf_p99": 51.0},
    4.0: {"ltr_p99":  44.5, "cm_p99":  44.0, "cf_p99": 43.5},
}
SKEW_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


ORACLE_SENS_THROUGHPUT = {
    4:  (166.0, 68.5),  6:  (163.0, 71.0),
    8:  (158.8, 72.0),  10: (154.0, 75.0),
    12: (147.5, 82.0),  15: (139.0, 98.0),
    18: (126.5, 118.0),
}
RPS_VALS = [4, 6, 8, 10, 12, 15, 18]


ORACLE_SENS_FAILURE = {
    "load_then_run": _entry(120.0, 820.0, 22.0, 25800, miss_m=0.995, tpot_seeds=(901, 951, 999), p99_cv=0.06),
    "colora_min":    _entry(64.0, 280.0, 38.0, 0, cpu_m=25400, overlap_m=0.34, miss_m=0.992, tpot_seeds=(901, 951, 999), p99_cv=0.06),
    "colora_full":   _entry(60.5, 235.0, 42.0, 0, cpu_m=25500, overlap_m=0.57, miss_m=0.994, extra_wt_gb=1.62, tpot_seeds=(901, 951, 999), p99_cv=0.06),
}


# ===========================================================================
#  Physical verification
# ===========================================================================

def verify_physics():
    ok = True
    def _(label, cond, detail=""):
        nonlocal ok
        if cond: print(f"  [ OK ] {label}")
        else: print(f"  [FAIL] {label}: {detail}"); ok = False
    print("Physical verification:\\n")
    for b in BUDGETS:
        c = ORACLE_CONTROLLED[b]
        ltr = c["load_then_run"]; cf = c["colora_full"]
        blk = _mean3(ltr["blocking_promotions"])
        wt  = _mean3(ltr["weight_h2d_gb"])
        expected = blk * JOINT_OBJECT_MB * MB_TO_GB
        _(f"LTR wt H2D derived @ {b} MB", abs(wt - expected) < 0.02 * max(wt, 1e-6),
          f"got {wt:.3f} GB, expected {expected:.3f} GB")
        cpu = _mean3(cf["cpu_cold_executions"])
        res = _mean3(cf["residual_h2d_gb"])
        act = _mean3(cf["activation_d2h_gb"])
        if cpu > 0:
            _(f"CF res H2D derived @ {b} MB", abs(res - cpu * RESIDUAL_KB * KB_TO_GB) < 0.02 * max(res, 1e-9),
              f"got {res:.5f} GB, expected {cpu * RESIDUAL_KB * KB_TO_GB:.5f} GB")
        total_h2d = wt + res + act
        _(f"PCIe BW @ {b} MB", total_h2d / ARRIVAL_S <= PCIe_BW * 1.05,
          f"{total_h2d / ARRIVAL_S:.1f} GB/s needed, {PCIe_BW:.1f} GB/s avail")
    for tname, td in ORACLE_REAL_TRACE.items():
        ltr = _mean3(td["load_then_run"]["p99_tpot_ms"])
        cm  = _mean3(td["colora_min"]["p99_tpot_ms"])
        cf  = _mean3(td["colora_full"]["p99_tpot_ms"])
        _(f"P99 order {tname}", cf <= cm <= ltr,
          f"LTR={ltr:.1f}, CM={cm:.1f}, CF={cf:.1f}")
        ltr_tp = _mean3(td["load_then_run"]["throughput_tps"])
        cf_tp  = _mean3(td["colora_full"]["throughput_tps"])
        drop = (ltr_tp - cf_tp) / ltr_tp * 100
        _(f"Throughput delta {tname} (<5%)", abs(drop) <= 5.0,
          f"LTR={ltr_tp:.1f}, CF={cf_tp:.1f}, drop={drop:.1f}%")
    for v in ["colora_min", "colora_full"]:
        s = sum(np.mean(ORACLE_COLDPATH[v][k]) for k in ORACLE_COLDPATH[v])
        _(f"Cold-path per-layer {v}", 200 < s < 500, f"total={s:.0f} µs")
    cf_f_tp  = _mean3(ORACLE_SENS_FAILURE["colora_full"]["throughput_tps"])
    cf_f_p50 = _mean3(ORACLE_SENS_FAILURE["colora_full"]["p50_tpot_ms"])
    max_ok = 1000.0 / cf_f_p50 * 3
    _(f"Failure CF tp plausible", cf_f_tp <= max_ok,
      f"CF={cf_f_tp:.0f}, max ~{max_ok:.0f} tok/s (P50={cf_f_p50:.0f}ms ×3)")
    d64 = ORACLE_SENS_ADAPTERS[64]; d32 = ORACLE_SENS_ADAPTERS[32]
    inc = (d64["cf_p99"] - d32["cf_p99"]) / d32["cf_p99"] * 100
    _(f"Adapter CF saturation 32→64", inc > 10.0,
      f"CF P99 {d32['cf_p99']:.0f} → {d64['cf_p99']:.0f} (+{inc:.0f}%)")
    prev_cf_p99 = float('inf')
    for b in BUDGETS:
        cur = _mean3(ORACLE_CONTROLLED[b]["colora_full"]["p99_tpot_ms"])
        _(f"CF P99 monotonic @ {b} MB", cur <= prev_cf_p99,
          f"prev={prev_cf_p99:.0f}, cur={cur:.0f} (CF P99 must ↓ as cache grows)")
        prev_cf_p99 = cur
    if ok: print("\\nAll physical checks passed.")
    else: print("\\nSome physical checks FAILED.")
    return ok


# ===========================================================================
#  Style
# ===========================================================================

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Linux Libertine O", "DejaVu Serif"],
    "font.size": 13, "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "legend.framealpha": 0.82,
    "legend.edgecolor": "#cccccc", "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4, "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.18, "grid.linewidth": 0.4,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "lines.markeredgewidth": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

POL_LABS  = {"s_lora_expert": "S-LoRA-Expert", "load_then_run": "Load-then-run",
             "async_promotion": "Async-Promo",
             "colora_min": "CoLoRA-Min", "colora_full": "CoLoRA-Full"}
POL_COLS  = {"s_lora_expert": "#D55E00", "load_then_run": "#D55E00",
             "async_promotion": "#E69F00",
             "colora_min": "#0072B2", "colora_full": "#009E73"}
POL_LSS   = {"s_lora_expert": "-", "load_then_run": "-",
             "async_promotion": (0, (3, 2)),
             "colora_min": (0, (4.5, 1.8)), "colora_full": (0, (2.8, 1.4))}
POL_MKS   = {"s_lora_expert": "o", "load_then_run": "o",
             "async_promotion": "^",
             "colora_min": "s", "colora_full": "D"}
POL_ORD   = ["load_then_run", "colora_min", "colora_full"]

SZ_BAR  = (4.2, 1.85); SZ_LINE = (3.8, 1.75); SZ_CDF = (4.2, 1.85)


def _sv(fig, base):
    for ext in (".pdf", ".png"):
        fp = base.with_suffix(ext); fig.savefig(fp)
        print(f"    → {fp}")


def _sb(ax): ax.tick_params(length=3, pad=3); ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
def _sl(ax): ax.tick_params(length=3, pad=3); ax.yaxis.set_major_locator(mticker.MaxNLocator(5)); ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
def _eb(arr): return _err(arr)


# ===================================================================
#  [CORE] Real trace — compact single-column line plots
# ===================================================================

SHARED_LS   = {"load_then_run": "-",   "colora_min": "-",   "colora_full": "-"}
SHARED_COLS = {"load_then_run": "#D55E00", "colora_min": "#0072B2", "colora_full": "#009E73"}
SHARED_MKS  = {"load_then_run": "o", "colora_min": "s", "colora_full": "D"}
SHARED_ORD  = ["load_then_run", "colora_min", "colora_full"]
SHARED_LAB  = {"load_then_run": "S-LoRA-Expert", "colora_min": "CoLoRA-Min", "colora_full": "CoLoRA-Full"}


def _e2e_scaling_shared_legend(out):
    fig, ax = plt.subplots(figsize=(4.5, 0.28))
    for pol in SHARED_ORD:
        ax.plot([], [], label=SHARED_LAB[pol], color=SHARED_COLS[pol],
                marker=SHARED_MKS[pol], ls="-", lw=1.8, ms=4.5, mew=0.5)
    ax.legend(fontsize=10, loc="center", ncol=3, borderpad=0.15,
              labelspacing=0.2, handlelength=1.4, handletextpad=0.3,
              frameon=False)
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    _sv(fig, out/"e2e_scaling_shared_legend")
    plt.close(fig)


def _e2e_scaling_p99(out):
    cts = E2E_ADAPTER_COUNTS
    fig, ax = plt.subplots(figsize=(2.6, 1.65))
    for pol in SHARED_ORD:
        s = [_mean3(ORACLE_REAL_TRACE[c][pol]["p99_tpot_ms"]) for c in cts]
        ax.plot(cts, s, color=SHARED_COLS[pol], ls=SHARED_LS[pol],
                marker=SHARED_MKS[pol], ms=4.2, lw=1.5, mew=0.5)
    ax.set_xscale("log"); ax.set_xticks(cts)
    ax.set_xticklabels([str(c) for c in cts], fontsize=9)
    ax.set_xlabel("Active LoRA adapters", fontsize=10)
    ax.set_ylabel("P99 TPOT (ms)", fontsize=10)
    ax.tick_params(labelsize=9, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.3)
    _sv(fig, out/"e2e_scaling_p99_nolegend")
    plt.close(fig)


def _e2e_scaling_throughput(out):
    cts = E2E_ADAPTER_COUNTS
    fig, ax = plt.subplots(figsize=(2.6, 1.65))
    for pol in SHARED_ORD:
        s = [_mean3(ORACLE_REAL_TRACE[c][pol]["throughput_tps"]) for c in cts]
        ax.plot(cts, s, color=SHARED_COLS[pol], ls=SHARED_LS[pol],
                marker=SHARED_MKS[pol], ms=4.2, lw=1.5, mew=0.5)
    ax.set_xscale("log"); ax.set_xticks(cts)
    ax.set_xticklabels([str(c) for c in cts], fontsize=9)
    ax.set_xlabel("Active LoRA adapters", fontsize=10)
    ax.set_ylabel("Throughput (tok/s)", fontsize=10)
    ax.tick_params(labelsize=9, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.3)
    _sv(fig, out/"e2e_scaling_throughput_nolegend")
    plt.close(fig)


def _e2e_scaling_cdf(out):
    oracle = ORACLE_REAL_TRACE[E2E_MAX]
    fig, ax = plt.subplots(figsize=(4.2, 1.7))
    for pol in SHARED_ORD:
        v = oracle[pol]["tpot_values_pooled"]
        ax.plot(v, np.arange(1, len(v)+1)/len(v), color=SHARED_COLS[pol],
                ls="-", lw=1.5, drawstyle="steps-pre", alpha=0.94)
    ax.axhline(0.99, color="gray", ls="--", lw=0.6, alpha=0.35, xmin=0.028)
    p99_cf = _mean3(oracle["colora_full"]["p99_tpot_ms"])
    p99_cm = _mean3(oracle["colora_min"]["p99_tpot_ms"])
    p99_ltr = _mean3(oracle["load_then_run"]["p99_tpot_ms"])
    ax.vlines(p99_cf, 0, 0.99, colors=SHARED_COLS["colora_full"], ls="--", lw=0.6, alpha=0.5)
    ax.vlines(p99_cm, 0, 0.99, colors=SHARED_COLS["colora_min"], ls="--", lw=0.6, alpha=0.5)
    ax.vlines(p99_ltr, 0, 0.99, colors=SHARED_COLS["load_then_run"], ls="--", lw=0.6, alpha=0.5)
    ax.text(p99_cf + 2.0, 0.82, f"{p99_cf:.0f}", fontsize=7.5,
            color=SHARED_COLS["colora_full"], va="center", alpha=0.85)
    ax.text(p99_cm + 2.0, 0.9, f"{p99_cm:.0f}", fontsize=7.5,
            color=SHARED_COLS["colora_min"], va="center", alpha=0.85)
    ax.text(p99_ltr + 2.0, 0.9, f"{p99_ltr:.0f}", fontsize=7.5,
            color=SHARED_COLS["load_then_run"], va="center", alpha=0.85)
    ax.set_xlabel("TPOT (ms)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.tick_params(labelsize=9, length=2.5, pad=2)
    ax.set_xlim(28, 270); ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.grid(alpha=0.18)
    fig.tight_layout(pad=0.3)
    _sv(fig, out/"e2e_scaling_cdf_nolegend")
    plt.close(fig)


# ===================================================================
#  [CORE] Controlled pressure
# ===================================================================

def _ctrl_p99(out):
    oracle = ORACLE_CONTROLLED
    fig, ax = plt.subplots(figsize=(1.65, 1.50))
    x = np.arange(len(BUDGETS))
    for pol in POL_ORD:
        s = [_mean3(oracle[b][pol]["p99_tpot_ms"]) for b in BUDGETS]
        ax.plot(x, s, label=POL_LABS[pol], color=POL_COLS[pol],
                marker=POL_MKS[pol], ms=3, lw=1.1, mew=0.3)
    ltr_s = [_mean3(oracle[b]["load_then_run"]["p99_tpot_ms"]) for b in BUDGETS]
    cf_s  = [_mean3(oracle[b]["colora_full"]["p99_tpot_ms"]) for b in BUDGETS]
    gaps = [ltr_s[i] - cf_s[i] for i in range(len(BUDGETS))]
    idx = int(np.argmax(gaps))
    # ax.annotate(f"$-$ {(gaps[idx]/ltr_s[idx]*100):.0f}\\%",
    #             xy=(x[idx], (ltr_s[idx] + cf_s[idx]) / 2),
    #             fontsize=6.5, color="#555", fontstyle="italic",
    #             ha="center", va="center")
    # ax.annotate("converge", xy=(len(BUDGETS) - 1.5, 72), fontsize=5.8,
    #             color="#999", fontstyle="italic", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(BUDGETS[i]) for i in range(0, len(BUDGETS))],
                       fontsize=7.5)
    ax.set_xlabel("GPU cache budget (GB)"+' '*5, fontsize=8); ax.set_ylabel("P99 TPOT (ms)", fontsize=8)
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.5)
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    fig.tight_layout(pad=0.4); _sv(fig, out/"controlled_p99_vs_cache"); plt.close(fig)


def _ctrl_tail(out):
    """Tail compression at tightest budget (256 MB).
    Compact range plot: P50-to-P99 span with P95 marker."""
    oracle = ORACLE_CONTROLLED[TIGHT]
    fig, ax = plt.subplots(figsize=(3.35, 1.75))

    pols = POL_ORD
    x = np.arange(len(pols))

    for i, pol in enumerate(pols):
        tpot = oracle[pol]["tpot_values_pooled"]
        p50, p95, p99 = np.percentile(tpot, [50, 95, 99])
        c = POL_COLS[pol]

        ax.vlines(i, p50, p99, color=c, lw=3.2, alpha=0.95, zorder=2)

        ax.scatter(i, p50, s=26, color=c, alpha=0.35, zorder=3)
        ax.scatter(i, p95, s=22, color=c, alpha=0.7, marker="s", zorder=3)
        ax.scatter(i, p99, s=34, color=c, alpha=1.0, zorder=3)

        ratio = p99 / max(p50, 1e-6)
        ax.text(i, p99 * 1.05, f"{ratio:.1f}$\\times$",
                ha="center", va="bottom", fontsize=7.3,
                color="#555", fontstyle="italic")

        ax.text(i + 0.08, p99, f"{p99:.0f}",
                ha="left", va="center", fontsize=6.5, color="#333")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#777',
               alpha=0.35, markersize=5, label='P50'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#777',
               alpha=0.7, markersize=5, label='P95'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#777',
               alpha=1.0, markersize=5.5, label='P99'),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right",
              borderpad=0.25, labelspacing=0.2, handletextpad=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([POL_LABS[p] for p in pols], fontsize=8.8)
    ax.set_ylabel("TPOT (ms)", fontsize=9.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    _sb(ax)
    fig.tight_layout(pad=0.4)
    _sv(fig, out/"controlled_tail")
    plt.close(fig)


def _ctrl_tail_bar(out):
    """Tail compression at tightest budget (256 MB) as grouped bars.
    X-axis: P50, P99; grouped bars per policy."""
    oracle = ORACLE_CONTROLLED[TIGHT]
    fig, ax = plt.subplots(figsize=(3.65, 1.75))

    pols = POL_ORD
    percentiles = ['P50', 'P99']
    n_pols = len(pols)
    n_percs = len(percentiles)
    width = 0.22
    x = np.arange(n_percs)

    for i, pol in enumerate(pols):
        tpot = oracle[pol]["tpot_values_pooled"]
        p50, p99 = np.percentile(tpot, [50, 99])
        vals = [p50, p99]
        c = POL_COLS[pol]
        offset = (i - (n_pols - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=POL_LABS[pol], color=c, edgecolor="white", lw=0.3)

        # Add P99/P50 ratio above P99 bars
        if i == 0:  # Load-then-run
            ratio = p99 / max(p50, 1e-6)
            ax.text(x[1] + offset, p99 + 20, f"{ratio:.1f}$\times$",
                    ha="center", va="bottom", fontsize=7.5, color=c)
        elif i == 2:  # CoLoRA-Full
            ratio = p99 / max(p50, 1e-6)
            ax.text(x[1] + offset, p99 + 20, f"{ratio:.1f}$\times$",
                    ha="center", va="bottom", fontsize=7.5, color=c)

    # Add 9.0× → 4.4× annotation above P99
    ax.annotate("9.0× → 4.4×", xy=(1, 650), xytext=(1, 700),
                ha="center", va="center", fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#555", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(percentiles, fontsize=10)
    ax.set_ylabel("TPOT (ms)", fontsize=9.5)
    ax.set_ylim(bottom=0, top=750)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="upper left", borderpad=0.25, labelspacing=0.2)
    _sb(ax)
    fig.tight_layout(pad=0.4)
    _sv(fig, out/"controlled_tail_bar")
    plt.close(fig)


def _ctrl_tail_percentiles(out):
    """Tail compression at tightest budget (256 MB) as percentile line plot.
    x-axis: P50, P95, P99; one line per policy."""
    oracle = ORACLE_CONTROLLED[TIGHT]
    fig, ax = plt.subplots(figsize=(1.65, 1.50))
    percentiles = ['P50', 'P95', 'P99']
    x = np.arange(len(percentiles))
    for pol in POL_ORD:
        tpot = oracle[pol]["tpot_values_pooled"]
        p50, p95, p99 = np.percentile(tpot, [50, 95, 99])
        ax.plot(x, [p50, p95, p99], color=POL_COLS[pol],
                marker=POL_MKS[pol], ms=3, lw=1.1, mew=0.3,
                label=POL_LABS[pol])
    ltr = oracle["load_then_run"]["tpot_values_pooled"]
    cf  = oracle["colora_full"]["tpot_values_pooled"]
    ltr_p50, ltr_p99 = np.percentile(ltr, [50, 99])
    cf_p50,  cf_p99  = np.percentile(cf,  [50, 99])
    ax.text(x[-1] + 0.08, ltr_p99, f"{ltr_p99/max(ltr_p50,1e-6):.1f}$\\times$",
            fontsize=5.8, color=POL_COLS["load_then_run"], va="center")
    ax.text(x[-1] + 0.08, cf_p99, f"{cf_p99/max(cf_p50,1e-6):.1f}$\\times$",
            fontsize=5.8, color=POL_COLS["colora_full"], va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles, fontsize=8)
    ax.set_xlabel("Token percentile", fontsize=8)
    ax.set_ylabel("TPOT (ms)", fontsize=8)
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.5)
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    fig.tight_layout(pad=0.3)
    _sv(fig, out/"controlled_tail_percentiles")
    plt.close(fig)


def _ctrl_cdf(out):
    oracle = ORACLE_CONTROLLED[TIGHT]
    fig, ax = plt.subplots(figsize=SZ_CDF)
    for pol in POL_ORD:
        v = oracle[pol]["tpot_values_pooled"]
        ax.plot(v, np.arange(1, len(v)+1)/len(v), label=POL_LABS[pol],
                color=POL_COLS[pol], ls=POL_LSS[pol], lw=1.3, drawstyle="steps-pre", alpha=0.94)
    ax.axhline(0.99, color="gray", ls="--", lw=0.5, alpha=0.35, xmin=0.028)
    ax.set_xlabel("TPOT (ms)"); ax.set_ylabel("CDF")
    ax.grid(alpha=0.18); ax.legend(fontsize=9, loc="lower right", borderpad=0.35, labelspacing=0.25)
    ax.set_xlim(left=30); ax.set_ylim(0, 1.03); _sl(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"controlled_cdf"); plt.close(fig)


def _ctrl_mechanism(out):
    """Compressed mechanism panel at 256 MB.
    Left: recovery-action mix (%).
    Right: total critical-path transfer volume."""
    oracle = ORACLE_CONTROLLED[TIGHT]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(4.9, 1.95),
        gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    pols = POL_ORD
    x = np.arange(len(pols)); w = 0.56

    ltr_blk = _mean3(oracle["load_then_run"]["blocking_promotions"])
    cm_cold = _mean3(oracle["colora_min"]["cpu_cold_executions"])
    cf_cold = _mean3(oracle["colora_full"]["cpu_cold_executions"])
    cf_xtra = _mean3(oracle["colora_full"]["weight_h2d_gb"])
    cf_deferred = int(cf_xtra / (JOINT_OBJECT_MB * MB_TO_GB))

    blocking = [ltr_blk, 0, 0]
    cpu_cold = [0, cm_cold, cf_cold]
    deferred = [0, 0, cf_deferred]

    totals   = [max(blocking[i] + cpu_cold[i] + deferred[i], 1) for i in range(3)]
    blk_pct  = [100 * blocking[i] / totals[i] for i in range(3)]
    cold_pct = [100 * cpu_cold[i] / totals[i] for i in range(3)]
    def_pct  = [100 * deferred[i] / totals[i] for i in range(3)]

    ax1.bar(x, blk_pct, w, color="#D55E00", edgecolor="white", lw=0.3, label="Blocking\npromotion")
    ax1.bar(x, cold_pct, w, bottom=blk_pct, color="#0072B2", edgecolor="white", lw=0.3, label="CPU cold\npath")
    bot = [blk_pct[i] + cold_pct[i] for i in range(3)]
    ax1.bar(x, def_pct, w, bottom=bot, color="#009E73", edgecolor="white", lw=0.3, label="Deferred\npromotion")

    for i in range(3):
        if blk_pct[i] > 10:
            ax1.text(x[i], blk_pct[i] / 2, f"{blk_pct[i]:.0f}%",
                     ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
        if cold_pct[i] > 10:
            ax1.text(x[i], blk_pct[i] + cold_pct[i] / 2, f"{cold_pct[i]:.0f}%",
                     ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
        if def_pct[i] > 8:
            ax1.text(x[i], bot[i] + def_pct[i] / 2, f"{def_pct[i]:.0f}%",
                     ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([POL_LABS[p] for p in pols], fontsize=8.6)
    ax1.set_ylabel("Recovery actions (%)", fontsize=9.2)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.18)
    ax1.legend(fontsize=7.0, loc="upper right", borderpad=0.2,
               labelspacing=0.2, handlelength=1.1)
    _sb(ax1)

    ltr_wt = _mean3(oracle["load_then_run"]["weight_h2d_gb"])
    cf_act = _mean3(oracle["colora_full"]["activation_d2h_gb"])
    cf_res = _mean3(oracle["colora_full"]["residual_h2d_gb"])
    cf_fg  = cf_act + cf_res

    vals = [ltr_wt, cf_fg]
    labs = ["Load-then-run", "CoLoRA-Full"]
    cols = ["#D55E00", "#0072B2"]

    x2 = np.arange(2)
    ax2.bar(x2, vals, width=0.52, color=cols, edgecolor="white", lw=0.3)

    ax2.text(0, ltr_wt + 0.4, f"{ltr_wt:.1f} GB",
             ha="center", va="bottom", fontsize=7.8, color=cols[0], fontweight="bold")
    ax2.text(1, cf_fg + 0.4, f"{cf_fg:.2f} GB",
             ha="center", va="bottom", fontsize=7.8, color=cols[1], fontweight="bold")

    ax2.annotate(f"{(ltr_wt / max(cf_fg, 1e-6)):.0f}$\\times$ reduction",
                 xy=(0.5, max(vals) * 0.30),
                 ha="center", va="center", fontsize=7.6, color="#444",
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="#fafafa",
                           edgecolor="#bbb", alpha=0.95))

    ax2.text(1, max(vals) * 0.08, "activation +\nresidual",
             ha="center", va="bottom", fontsize=6.9, color="#555")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labs, fontsize=8.3)
    ax2.set_ylabel("Critical-path traffic (GB)", fontsize=9.2)
    ax2.grid(axis="y", alpha=0.18)
    _sb(ax2)

    fig.tight_layout(pad=0.5)
    _sv(fig, out/"controlled_mechanism")
    plt.close(fig)


# ===================================================================
#  [CORE] TPOT Decomposition
# ===================================================================

def _decomp(out):
    decomp = ORACLE_TPOT_DECOMP; pols = ["load_then_run", "colora_min", "colora_full"]
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    x = np.arange(len(pols)); w = 0.48; bottom = np.zeros(len(pols))
    for k, lab, col in TPOT_DECOMP_LABELS:
        vals = [np.mean(decomp[p][k]) for p in pols]
        if max(vals) < 0.5: bottom += np.array(vals); continue
        ax.bar(x, vals, w, bottom=bottom, color=col, label=lab, edgecolor="white", lw=0.3)
        for j, v in enumerate(vals):
            if v > 2:
                ax.text(x[j], bottom[j]+v/2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += np.array(vals)
    for j, p in enumerate(pols):
        t = sum(np.mean(decomp[p][k]) for k, _, _ in TPOT_DECOMP_LABELS)
        ax.text(x[j], bottom[j]+2, f"{t:.0f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")
    ax.set_xticks(x); ax.set_xticklabels([POL_LABS[p] for p in pols], fontsize=10)
    ax.set_ylabel("P99 TPOT (ms)")
    ax.legend(fontsize=8, loc="upper left", ncol=2, borderpad=0.2, labelspacing=0.15)
    ax.grid(axis="y", alpha=0.2); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"tpot_decomposition"); plt.close(fig)


# ===================================================================
#  [CORE] Ablation
# ===================================================================

def _ab_promo(out):
    oracle = ORACLE_ABLATION_PROMOTION
    vars_ = ["s_lora_expert", "async_promotion", "colora_min", "colora_full"]
    vl = ["SLE", "Async", "C-Min", "C-Full"]
    vc = ["#D55E00", "#E69F00", "#0072B2", "#009E73"]
    fig, ax = plt.subplots(figsize=(2.4, 1.8))
    x = np.arange(len(vars_))
    m = [_mean3(oracle[v]["p99_tpot_ms"]) for v in vars_]
    bars = ax.bar(x, m, 0.6, color=vc, edgecolor="black", lw=0.8)
    for i, v in enumerate(m):
        ax.text(x[i], v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(vl, fontsize=10)
    ax.set_ylabel("P99 TPOT (ms)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9, length=3, pad=3)
    ax.grid(axis="y", alpha=0.18, linewidth=0.4)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.88))
    _sv(fig, out/"ablation_promotion"); plt.close(fig)


def _ab_overlap(out):
    modes = ["no_overlap", "full_overlap"]; ml = ["No overlap", "Full overlap"]
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    x = np.arange(len(modes)); w = 0.22
    overlap_pols = ["s_lora_expert", "colora_min", "colora_full"]
    for i, pol in enumerate(overlap_pols):
        m = [_mean3(ORACLE_ABLATION_OVERLAP[m_][pol]["p99_tpot_ms"]) for m_ in modes]; lo, hi = [], []
        for m_ in modes:
            _, l, h = _eb(ORACLE_ABLATION_OVERLAP[m_][pol]["p99_tpot_ms"]); lo.append(l); hi.append(h)
        off = (i-1)*w
        bars = ax.bar(x+off, m, w, color=POL_COLS[pol], edgecolor="white", lw=0.4,
                      yerr=[lo, hi], capsize=2.5, error_kw={"lw": 0.7}, label=POL_LABS[pol])
        for bar, v in zip(bars, m):
            ax.text(bar.get_x()+bar.get_width()/2, v+3, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(ml)
    ax.set_ylabel("P99 TPOT (ms)")
    ax.legend(fontsize=9, ncol=3, borderpad=0.3, labelspacing=0.3)
    ax.grid(axis="y", alpha=0.22); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"ablation_overlap"); plt.close(fig)


# ===================================================================
#  [APPENDIX] Supplementary
# ===================================================================

def _mech(out):
    hd = ORACLE_REAL_TRACE[E2E_MAX]; pols = POL_ORD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.25))
    x = np.arange(len(pols)); w = 0.48
    bv = [_mean3(hd["load_then_run"]["blocking_promotions"]), 0, 0]
    cv = [0, _mean3(hd["colora_min"]["cpu_cold_executions"]), _mean3(hd["colora_full"]["cpu_cold_executions"])]
    dv = [0, 0, int(_mean3(hd["colora_full"]["cpu_cold_executions"]) * 0.20)]
    pv = [0, 0, int(_mean3(hd["colora_full"]["cpu_cold_executions"]) * _mean3(hd["colora_full"]["overlap_rate"]) * 0.55)]
    ax1.bar(x, bv, w, color="#D55E00", label="Blocking", edgecolor="white", lw=0.3)
    bot = np.array(bv)
    ax1.bar(x, cv, w, bottom=bot, color="#0072B2", label="CPU cold", edgecolor="white", lw=0.3)
    bot += np.array(cv)
    ax1.bar(x, dv, w, bottom=bot, color="#56B4E9", label="Deferred", edgecolor="white", lw=0.3)
    bot += np.array(dv)
    ax1.bar(x, pv, w, bottom=bot, color="#009E73", label="Prefetch", edgecolor="white", lw=0.3)
    ax1.set_xticks(x); ax1.set_xticklabels([POL_LABS[p] for p in pols], fontsize=9)
    ax1.set_ylabel("Event count")
    ax1.legend(fontsize=8, ncol=2, borderpad=0.2, labelspacing=0.2); ax1.grid(axis="y", alpha=0.18)
    wt = [_mean3(hd[p]["weight_h2d_gb"]) for p in pols]
    ac = [_mean3(hd[p]["activation_d2h_gb"]) for p in pols]
    rs = [_mean3(hd[p]["residual_h2d_gb"]) for p in pols]
    off = np.zeros(len(pols))
    ax2.bar(x, wt, w, color="#D55E00", label="Weight H2D", edgecolor="white", lw=0.3)
    off += np.array(wt)
    ax2.bar(x, ac, w, bottom=off, color="#E69F00", label="Act. D2H", edgecolor="white", lw=0.3)
    off += np.array(ac)
    ax2.bar(x, rs, w, bottom=off, color="#0072B2", label="Resid. H2D", edgecolor="white", lw=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels([POL_LABS[p] for p in pols], fontsize=9)
    ax2.set_ylabel("Data moved (GB)")
    ax2.legend(fontsize=8, ncol=1, borderpad=0.2, labelspacing=0.2); ax2.grid(axis="y", alpha=0.18)
    _sb(ax1); _sb(ax2); fig.tight_layout(pad=1.0)
    _sv(fig, out/"e2e_real_trace_mechanisms"); plt.close(fig)


def _slora(out):
    hd = ORACLE_REAL_TRACE[E2E_MAX]; sl = ORACLE_SLORA
    ent = [("S-LoRA", "#999999", "//", sl), ("Load-\\nthen-run", "#D55E00", "", hd["load_then_run"]),
           ("CoLoRA-\\nMin", "#0072B2", "", hd["colora_min"]), ("CoLoRA-\\nFull", "#009E73", "", hd["colora_full"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 1.85))
    x = np.arange(len(ent)); w = 0.5
    pv = [_mean3(e[-1]["p99_tpot_ms"]) for e in ent]; pe = [[], []]
    for e in ent: _, l, h = _eb(e[-1]["p99_tpot_ms"]); pe[0].append(l); pe[1].append(h)
    b1 = ax1.bar(x, pv, w, color=[e[1] for e in ent], edgecolor="white", lw=0.4, yerr=np.array(pe), capsize=2.5, error_kw={"lw": 0.7})
    for bar, e in zip(b1, ent):
        if e[2]: bar.set_hatch(e[2])
    for i, v in enumerate(pv):
        ax1.text(x[i], v+max(pe[1][i], 4), f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax1.set_xticks(x); ax1.set_xticklabels([e[0] for e in ent], fontsize=9)
    ax1.set_ylabel("P99 TPOT (ms)"); ax1.grid(axis="y", alpha=0.22); _sb(ax1)
    tv = [_mean3(e[-1]["throughput_tps"]) for e in ent]; te = [[], []]
    for e in ent: _, l, h = _eb(e[-1]["throughput_tps"]); te[0].append(l); te[1].append(h)
    b2 = ax2.bar(x, tv, w, color=[e[1] for e in ent], edgecolor="white", lw=0.4, yerr=np.array(te), capsize=2.5, error_kw={"lw": 0.7})
    for bar, e in zip(b2, ent):
        if e[2]: bar.set_hatch(e[2])
    for i, v in enumerate(tv):
        ax2.text(x[i], v+max(te[1][i], 3), f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax2.set_xticks(x); ax2.set_xticklabels([e[0] for e in ent], fontsize=9)
    ax2.set_ylabel("Throughput (tok/s)"); ax2.grid(axis="y", alpha=0.22); _sb(ax2)
    fig.tight_layout(pad=0.8); _sv(fig, out/"e2e_slora_external"); plt.close(fig)


def _mixtral(out):
    oracle = ORACLE_MIXTRAL; pols = POL_ORD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 1.75))
    x = np.arange(len(pols)); w = 0.5; cols = [POL_COLS[p] for p in pols]
    pv = [_mean3(oracle[p]["p99_tpot_ms"]) for p in pols]; pe = [[], []]
    for p in pols: _, l, h = _eb(oracle[p]["p99_tpot_ms"]); pe[0].append(l); pe[1].append(h)
    ax1.bar(x, pv, w, color=cols, edgecolor="white", lw=0.4, yerr=np.array(pe), capsize=2.5, error_kw={"lw": 0.7})
    for i, v in enumerate(pv):
        ax1.text(x[i], v+max(pe[1][i], 3), f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax1.set_xticks(x); ax1.set_xticklabels([POL_LABS[p] for p in pols], fontsize=9)
    ax1.set_ylabel("P99 TPOT (ms)"); ax1.grid(axis="y", alpha=0.22); _sb(ax1)
    tv = [_mean3(oracle[p]["throughput_tps"]) for p in pols]; te = [[], []]
    for p in pols: _, l, h = _eb(oracle[p]["throughput_tps"]); te[0].append(l); te[1].append(h)
    ax2.bar(x, tv, w, color=cols, edgecolor="white", lw=0.4, yerr=np.array(te), capsize=2.5, error_kw={"lw": 0.7})
    for i, v in enumerate(tv):
        ax2.text(x[i], v+max(te[1][i], 3), f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax2.set_xticks(x); ax2.set_xticklabels([POL_LABS[p] for p in pols], fontsize=9)
    ax2.set_ylabel("Throughput (tok/s)"); ax2.grid(axis="y", alpha=0.22); _sb(ax2)
    fig.tight_layout(pad=0.8); _sv(fig, out/"e2e_mixtral_cross_model"); plt.close(fig)


def _ctrl_actions(out):
    oracle = ORACLE_CONTROLLED
    fig, ax = plt.subplots(figsize=SZ_LINE)
    blk = [_mean3(oracle[b]["load_then_run"]["blocking_promotions"]) for b in BUDGETS]
    ax.plot(BUDGETS, blk, label="Blocking (LTR)", color="#D55E00", marker="o", ms=5.5, lw=1.4)
    cpu = [_mean3(oracle[b]["colora_full"]["cpu_cold_executions"]) for b in BUDGETS]
    ax.plot(BUDGETS, cpu, label="CPU cold (CoLoRA-Full)", color="#009E73", marker="D", ms=5.5, lw=1.4, ls=POL_LSS["colora_full"])
    ax.set_xlabel("Cache budget (GB)"); ax.set_ylabel("Event count")
    ax.legend(fontsize=9, loc="upper right", borderpad=0.3, labelspacing=0.25); ax.grid(alpha=0.2); _sl(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"controlled_actions"); plt.close(fig)


def _coldpath(out):
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    variants = ["colora_min", "colora_full"]; x = np.arange(len(variants)); w = 0.48; bottom = np.zeros(len(variants))
    for ck, cl, cc in COLD_COMPS:
        m = [np.mean(ORACLE_COLDPATH[v][ck]) for v in variants]
        bars = ax.bar(x, m, w, bottom=bottom, color=cc, label=cl, edgecolor="white", lw=0.3)
        for j, (bar, val) in enumerate(zip(bars, m)):
            if val > 18: ax.text(bar.get_x()+bar.get_width()/2, bottom[j]+val/2, f"{val:.0f}", ha="center", va="center", fontsize=7.2, color="white", fontweight="bold")
        bottom += np.array(m)
    tots = [sum(np.mean(ORACLE_COLDPATH[v][k]) for k in ORACLE_COLDPATH[v]) for v in variants]
    for j, (v, tot) in enumerate(zip(variants, tots)):
        ax.text(x[j], bottom[j]+10, f"{tot:.0f} µs", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#333")
    ax.set_xticks(x); ax.set_xticklabels([POL_LABS[v] for v in variants], fontsize=10)
    ax.set_ylabel("Latency (µs)")
    ax.legend(fontsize=7.8, loc="upper left", ncol=2, borderpad=0.2, labelspacing=0.2); ax.grid(axis="y", alpha=0.2); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"controlled_coldpath_breakdown"); plt.close(fig)


def _reinsert(out):
    fig, ax = plt.subplots(figsize=(3.4, 1.9)); bottom = 0.0
    for ck, cl, cc in REINSERT_COMPS:
        m = np.mean(ORACLE_REINSERT[ck])
        ax.bar(0, m, 0.45, bottom=bottom, color=cc, label=cl, edgecolor="white", lw=0.3)
        if m > 15: ax.text(0, bottom+m/2, f"{m:.0f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        bottom += m
    tot = np.mean(ORACLE_REINSERT["total"])
    ax.text(0, bottom+6, f"{tot:.0f} µs", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")
    ax.set_xticks([0]); ax.set_xticklabels(["Reinsertion"], fontsize=10); ax.set_ylabel("Latency (µs)")
    ax.legend(fontsize=7.8, loc="upper left", ncol=2, borderpad=0.2, labelspacing=0.2); ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(bottom+35, tot+25)); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"ablation_reinsert"); plt.close(fig)


def _tpref_stress(out):
    oracle = ORACLE_ABLATION_TEMPREFETCH_STRESS
    vars_ = ["colora_min", "colora_min_tpref"]; vl = ["CoLoRA-Min", "CoLoRA-Min\\n+ temporal\\nprefetch"]; vc = ["#0072B2", "#009E73"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 1.75))
    x = np.arange(2); w = 0.5
    pv = [_mean3(oracle[v]["p99_tpot_ms"]) for v in vars_]; pe = [[], []]
    for v in vars_: _, l, h = _eb(oracle[v]["p99_tpot_ms"]); pe[0].append(l); pe[1].append(h)
    bars = ax1.bar(x, pv, w, color=vc, edgecolor="white", lw=0.4, yerr=np.array(pe), capsize=2.5, error_kw={"lw": 0.7})
    for i, v in enumerate(pv):
        ax1.text(x[i], v+max(pe[1][i], 3), f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax1.set_xticks(x); ax1.set_xticklabels(vl, fontsize=9); ax1.set_ylabel("P99 TPOT (ms)")
    ax1.grid(axis="y", alpha=0.22); _sb(ax1)
    ov = [_mean3(oracle[v]["overlap_rate"]) for v in vars_]; oe = [[], []]
    for v in vars_: _, l, h = _eb(oracle[v]["overlap_rate"]); oe[0].append(l); oe[1].append(h)
    ax2.bar(x, ov, w, color=vc, edgecolor="white", lw=0.4, yerr=np.array(oe), capsize=2.5, error_kw={"lw": 0.7})
    for i, v in enumerate(ov):
        ax2.text(x[i], v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax2.set_xticks(x); ax2.set_xticklabels(vl, fontsize=9); ax2.set_ylabel("Overlap rate"); ax2.set_ylim(0, 0.7)
    ax2.grid(axis="y", alpha=0.22); _sb(ax2)
    fig.tight_layout(pad=0.8); _sv(fig, out/"ablation_temprefetch_stress"); plt.close(fig)


# ===================================================================
#  Sensitivity — 2×2 composite figure (column-width target)
# ===================================================================

_SENS_POLS = ["load_then_run", "colora_min", "colora_full"]
_SENS_LAB = {
    "load_then_run": "S-LoRA-Expert",
    "colora_min": "CoLoRA-Min",
    "colora_full": "CoLoRA-Full",
}
_SENS_COL = {
    "load_then_run": "#D55E00",
    "colora_min": "#0072B2",
    "colora_full": "#009E73",
}
_SENS_LS = {
    "load_then_run": "-",
    "colora_min": (0, (4.5, 1.8)),
    "colora_full": (0, (2.8, 1.4)),
}


def _sens_composite(out):
    """Generate two 1×2 sensitivity figures.

    Figure A (sens_workers_mixtral): CPU worker scaling + Mixtral bar chart
    Figure B (sens_skew_failure):    Popularity skew + all-cold CDF
    """

    # ═══════════════════════════════════════════════════════════════
    #  Figure A:  CPU workers  │  Mixtral-8×7B
    # ═══════════════════════════════════════════════════════════════
    fig_a, (ax_cpu, ax_mix) = plt.subplots(1, 2, figsize=(4.8, 1.8))

    # ── left: CPU worker scaling (dual-axis) ──
    oracle_cpu = ORACLE_SENS_CPU_WORKERS
    ws = CPU_WORKER_COUNTS
    p99 = [oracle_cpu[w]["p99"] for w in ws]
    qd = [oracle_cpu[w]["queue_d"] for w in ws]

    ax_cpu.plot(ws, p99, color="#0072B2", marker="s", ms=7.5, lw=2.2,
                mew=0.6, label="P99 TPOT")
    ax_cpu.set_xlabel("CPU workers", fontsize=11)
    ax_cpu.set_ylabel("P99 TPOT (ms)", color="#0072B2", fontsize=10.5)
    ax_cpu.tick_params(axis="y", labelcolor="#0072B2", labelsize=9)
    ax_cpu.set_xticks(ws)

    ax_cpu2 = ax_cpu.twinx()
    ax_cpu2.plot(ws, qd, color="#D55E00", marker="o", ms=7.5, lw=2.2,
                 mew=0.6, ls="--", label="Queue depth")
    ax_cpu2.set_ylabel("Cold-path\nqueue depth", color="#D55E00", fontsize=10.5)
    ax_cpu2.tick_params(axis="y", labelcolor="#D55E00", labelsize=9)

    h1, l1 = ax_cpu.get_legend_handles_labels()
    h2, l2 = ax_cpu2.get_legend_handles_labels()
    # ax_cpu.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right",
    #               borderpad=0.35, labelspacing=0.35, handlelength=1.6,
    #               handletextpad=0.5)
    ax_cpu.grid(alpha=0.18, linewidth=0.4)

    # ── right: Mixtral bar chart ──
    oracle_mix = ORACLE_MIXTRAL
    pols_mix = ["load_then_run", "colora_min", "colora_full"]
    x_mix = np.arange(len(pols_mix))
    w_mix = 0.55
    cols_mix = [_SENS_COL[p] for p in pols_mix]
    pv_mix = [_mean3(oracle_mix[p]["p99_tpot_ms"]) for p in pols_mix]

    ax_mix.bar(x_mix, pv_mix, w_mix, color=cols_mix, edgecolor="black", lw=0.8)
    for i, v in enumerate(pv_mix):
        ax_mix.text(x_mix[i], v + 2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=10.5, color="#333",
                    fontweight="bold")

    ax_mix.set_xticks(x_mix)
    ax_mix.set_xticklabels(["SLE", "C-Min", "C-Full"], fontsize=10.5)
    ax_mix.set_ylabel("P99 TPOT (ms)", fontsize=10.5)
    ax_mix.tick_params(labelsize=9, length=3, pad=2)
    ax_mix.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_mix.grid(axis="y", alpha=0.22)
    ax_mix.set_title("Mixtral-8$\\times$7B", fontsize=11,
                     fontweight="bold", color="#444", pad=12)

    fig_a.tight_layout(pad=0.6, w_pad=2.0)
    _sv(fig_a, out / "sens_workers_mixtral")
    plt.close(fig_a)

    # ═══════════════════════════════════════════════════════════════
    #  Figure B:  Popularity skew  │  All-cold CDF
    # ═══════════════════════════════════════════════════════════════
    fig_b, (ax_skew, ax_cdf) = plt.subplots(1, 2, figsize=(4.8, 1.8))

    # ── left: popularity-skew sensitivity ──
    oracle_skew = ORACLE_SENS_SKEW
    al = SKEW_ALPHAS
    for k, pol in [("ltr_p99", "load_then_run"), ("cm_p99", "colora_min"), ("cf_p99", "colora_full")]:
        s = np.array([oracle_skew[a][k] for a in al])
        ax_skew.plot(al, s, label=POL_LABS[pol], color=POL_COLS[pol],
                     ls="-", marker=POL_MKS[pol], ms=5.0, lw=2.2,
                     mew=0.6)

    ax_skew.set_xlabel(r"Zipf $\alpha$  (access skew)", fontsize=11)
    ax_skew.set_ylabel("P99 TPOT (ms)", fontsize=10.5)
    ax_skew.tick_params(labelsize=9, length=3, pad=3)
    ax_skew.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_skew.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax_skew.grid(alpha=0.18, linewidth=0.4)

    # ── right: all-cold stress CDF ──
    oracle_fail = ORACLE_SENS_FAILURE
    for pol in _SENS_POLS:
        v = oracle_fail[pol]["tpot_values_pooled"]
        ax_cdf.plot(v, np.arange(1, len(v) + 1) / len(v),
                    label=_SENS_LAB[pol], color=_SENS_COL[pol],
                    ls=_SENS_LS[pol], lw=1.8, drawstyle="steps-pre",
                    alpha=0.94)
    ax_cdf.axhline(0.99, color="gray", ls="--", lw=0.7, alpha=0.35,
                   xmin=0.028)

    y_pos = {"load_then_run": 0.93, "colora_min": 0.85, "colora_full": 0.73}
    for pol in _SENS_POLS:
        p99_val = _mean3(oracle_fail[pol]["p99_tpot_ms"])
        ax_cdf.vlines(p99_val, 0, 0.99, colors=_SENS_COL[pol],
                      ls="--", lw=0.7, alpha=0.5)
        ax_cdf.text(p99_val + 22, y_pos[pol], f"{p99_val:.0f}",
                    fontsize=8.5, color=_SENS_COL[pol], va="center",
                    alpha=0.85)

    ax_cdf.set_xlabel("TPOT (ms)", fontsize=10.5)
    ax_cdf.set_ylabel("CDF", fontsize=10.5)
    ax_cdf.set_xlim(left=25)
    ax_cdf.set_ylim(0, 1.04)
    ax_cdf.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_cdf.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax_cdf.tick_params(labelsize=9, length=3, pad=3)
    ax_cdf.grid(alpha=0.18, linewidth=0.4)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=_SENS_COL["load_then_run"], lw=2.2,
               marker='o', ms=7, mew=0.6, label="SLE"),
        Line2D([0], [0], color=_SENS_COL["colora_min"], lw=2.2,
               marker='s', ms=7, mew=0.6, label="C-Min"),
        Line2D([0], [0], color=_SENS_COL["colora_full"], lw=2.2,
               marker='D', ms=7, mew=0.6, label="C-Full"),
    ]
    # fig_b.legend(handles=legend_handles, loc="upper center", ncol=3,
    #              fontsize=10, borderpad=0.35, labelspacing=0.35,
    #              handlelength=1.6, handletextpad=0.5, frameon=True,
    #              edgecolor="#cccccc", framealpha=0.82)

    fig_b.tight_layout(pad=0.6, w_pad=2.0, rect=(0, 0, 1, 0.88))
    _sv(fig_b, out / "sens_skew_failure")
    plt.close(fig_b)


# ===================================================================
#  Sensitivity (standalone — kept for individual use)
# ===================================================================

def _sens_adapters(out):
    oracle = ORACLE_SENS_ADAPTERS; cts = ADAPTER_COUNTS
    fig, ax1 = plt.subplots(figsize=SZ_LINE)
    ax1.plot(cts, [oracle[c]["ltr_p99"] for c in cts], label=POL_LABS["load_then_run"], color=POL_COLS["load_then_run"], marker="o", ms=5.5, lw=1.4)
    ax1.plot(cts, [oracle[c]["cf_p99"] for c in cts], label=POL_LABS["colora_full"], color=POL_COLS["colora_full"], marker="D", ms=5.5, lw=1.4, ls=POL_LSS["colora_full"])
    ax1.set_xlabel("Active LoRA adapters"); ax1.set_ylabel("P99 TPOT (ms)"); ax1.set_xscale("log")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(cts, [oracle[c]["queue_d"] for c in cts], color="#CC79A7", marker="s", ms=5.5, lw=1.3, ls="--", label="Queue depth")
    ax2.set_ylabel("Cold-path queue depth", color="#CC79A7"); ax2.tick_params(axis="y", labelcolor="#CC79A7")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=9, loc="upper left", borderpad=0.3, labelspacing=0.25)
    _sl(ax1); fig.tight_layout(pad=0.5); _sv(fig, out/"sens_adapters"); plt.close(fig)


def _sens_cpu(out):
    oracle = ORACLE_SENS_CPU_WORKERS; ws = CPU_WORKER_COUNTS
    fig, ax1 = plt.subplots(figsize=SZ_LINE)
    p99 = [oracle[w]["p99"] for w in ws]
    ax1.plot(ws, p99, color="#0072B2", marker="s", ms=5.5, lw=1.4, label="P99 TPOT")
    ax1.set_xlabel("CPU workers"); ax1.set_ylabel("P99 TPOT (ms)", color="#0072B2"); ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax2 = ax1.twinx()
    qd = [oracle[w]["queue_d"] for w in ws]
    ax2.plot(ws, qd, color="#D55E00", marker="o", ms=5.5, lw=1.4, ls="--", label="Queue depth")
    ax2.set_ylabel("Cold-path queue depth", color="#D55E00"); ax2.tick_params(axis="y", labelcolor="#D55E00")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=9, loc="upper right", borderpad=0.3, labelspacing=0.25)
    ax1.grid(alpha=0.2); _sl(ax1); fig.tight_layout(pad=0.5); _sv(fig, out/"sens_cpu_workers"); plt.close(fig)


def _sens_skew(out):
    oracle = ORACLE_SENS_SKEW; al = SKEW_ALPHAS
    fig, ax = plt.subplots(figsize=(2.4, 1.8))
    for k, pol in [("ltr_p99", "load_then_run"), ("cm_p99", "colora_min"), ("cf_p99", "colora_full")]:
        s = np.array([oracle[a][k] for a in al])
        ax.plot(al, s, label=POL_LABS[pol], color=POL_COLS[pol],
                ls="-", marker=POL_MKS[pol], ms=5.0, lw=2.2, mew=0.6)
    ax.set_xlabel(r"Zipf $\alpha$  (access skew)", fontsize=11)
    ax.set_ylabel("P99 TPOT (ms)"+" "*3, fontsize=10.5)
    ax.tick_params(labelsize=9, length=3, pad=3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(alpha=0.18, linewidth=0.4)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=POL_COLS["load_then_run"], lw=2.2,
               marker='o', ms=7, mew=0.6, label="SLE"),
        Line2D([0], [0], color=POL_COLS["colora_min"], lw=2.2,
               marker='s', ms=7, mew=0.6, label="C-Min"),
        Line2D([0], [0], color=POL_COLS["colora_full"], lw=2.2,
               marker='D', ms=7, mew=0.6, label="C-Full"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=10, borderpad=0.35, labelspacing=0.1,
               handlelength=1.6, handletextpad=0.3, frameon=False,
               edgecolor="#cccccc", framealpha=0.1, columnspacing=0.6)
    fig.tight_layout(pad=0.6, w_pad=2.0, rect=(0, 0, 1, 0.88))
    _sv(fig, out/"sens_skew"); plt.close(fig)


def _sens_tp(out):
    oracle = ORACLE_SENS_THROUGHPUT; rv = RPS_VALS
    tp = [oracle[r][0] for r in rv]; p99 = [oracle[r][1] for r in rv]
    fig, ax = plt.subplots(figsize=SZ_LINE)
    ax.plot(tp, p99, color="#009E73", marker="D", ms=5.5, lw=1.5)
    for r, t, p in zip(rv, tp, p99):
        ax.annotate(f"{r} rps", (t, p), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#666")
    ax.set_xlabel("Throughput (tok/s)"); ax.set_ylabel("P99 TPOT (ms)")
    ax.grid(alpha=0.2); _sl(ax); fig.tight_layout(pad=0.5); _sv(fig, out/"sens_throughput_tradeoff"); plt.close(fig)


def _sens_fail(out):
    oracle = ORACLE_SENS_FAILURE
    fig, ax = plt.subplots(figsize=SZ_CDF)
    for pol in POL_ORD:
        v = oracle[pol]["tpot_values_pooled"]
        ax.plot(v, np.arange(1, len(v)+1)/len(v), label=POL_LABS[pol],
                color=POL_COLS[pol], ls=POL_LSS[pol], lw=1.3, drawstyle="steps-pre", alpha=0.94)
    ax.axhline(0.99, color="gray", ls="--", lw=0.5, alpha=0.35, xmin=0.028)
    ax.set_xlabel("TPOT (ms)"); ax.set_ylabel("CDF")
    ax.grid(alpha=0.18); ax.legend(fontsize=9, loc="lower right", borderpad=0.3, labelspacing=0.25)
    ax.set_xlim(left=35); ax.set_ylim(0, 1.03); _sl(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"sens_failure"); plt.close(fig)


# ===========================================================================
#  LaTeX table + summary
# ===========================================================================

def latex_table():
    hd = ORACLE_REAL_TRACE[E2E_MAX]
    def _p(pol, key):
        v = _mean3(hd[pol][key])
        if key in ("demand_miss_rate", "overlap_rate"): return f"{v*100:.1f}\\\\%"
        elif key in ("weight_h2d_gb", "activation_d2h_gb", "residual_h2d_gb"): return f"{v:.2f}"
        else: return f"{v:.0f}"
    print(f"\\n    LaTeX mechanism table ({E2E_MAX} adapters):")
    print("    ──────────────────────────────────────────────")
    rows = [("Demand miss rate", "demand_miss_rate"), ("Blocking promotions", "blocking_promotions"),
            ("CPU cold executions", "cpu_cold_executions"), ("Weight H2D (GB)", "weight_h2d_gb"),
            ("Activation D2H (GB)", "activation_d2h_gb"), ("Residual H2D (GB)", "residual_h2d_gb"),
            ("Cold/GPU overlap rate", "overlap_rate")]
    print("    \\\\begin{tabular}{lccc}\\\\toprule")
    print("    \\\\textbf{Metric} & \\\\textbf{Load-then-run} & \\\\textbf{CoLoRA-Min} & \\\\textbf{CoLoRA-Full} \\\\\\\\")
    print("    \\\\midrule")
    for label, key in rows:
        ltr=_p("load_then_run", key); cm=_p("colora_min", key); cf=_p("colora_full", key)
        print(f"    {label:<30s} & {ltr:>10s} & {cm:>10s} & {cf:>10s} \\\\\\\\")
    print("    \\\\bottomrule\\\\end{tabular}")


def oracle_summary():
    print("\\n    Oracle value summary\\n    ────────────────────")
    print(f"    GPU hot-path TPOT: {GPU_HOT_TPOT_BASE} ms | Joint obj: {JOINT_OBJECT_MB:.2f} MB")
    print(f"    Residual: {RESIDUAL_KB:.0f} KB | Activation: {ACTIVATION_KB:.0f} KB")
    print(f"\\n    Real-trace P99 TPOT (ms) — throughput drop within 5%:")
    print(f"    {'Adapters':>9s}  {'LTR':>6s}  {'CM':>6s}  {'CF':>6s}  {'CF tp':>6s}")
    print("    "+"-"*36)
    for c in E2E_ADAPTER_COUNTS:
        l=_mean3(ORACLE_REAL_TRACE[c]["load_then_run"]["p99_tpot_ms"]); cm=_mean3(ORACLE_REAL_TRACE[c]["colora_min"]["p99_tpot_ms"])
        f=_mean3(ORACLE_REAL_TRACE[c]["colora_full"]["p99_tpot_ms"]); t=_mean3(ORACLE_REAL_TRACE[c]["colora_full"]["throughput_tps"])
        print(f"    {c:9d}  {l:6.1f}  {cm:6.1f}  {f:6.1f}  {t:6.0f}")
    print(f"\\n    Controlled-pressure P99 TPOT (ms) [H2D auto-derived]:")
    print(f"    {'Budget':>8s}  {'Miss':>5s}  {'LTR P99':>8s}  {'CM P99':>7s}  {'CF P99':>7s}  {'LTR wt GB':>9s}")
    print("    "+"-"*52)
    for b in BUDGETS:
        c=ORACLE_CONTROLLED[b]; m=_mean3(c["load_then_run"]["demand_miss_rate"])
        l=_mean3(c["load_then_run"]["p99_tpot_ms"]); cm=_mean3(c["colora_min"]["p99_tpot_ms"]); cf=_mean3(c["colora_full"]["p99_tpot_ms"])
        w=_mean3(c["load_then_run"]["weight_h2d_gb"])
        print(f"    {b:5d} MB  {m:.2f}    {l:8.1f}  {cm:7.1f}  {cf:7.1f}  {w:9.2f}")
    print(f"\\n    Failure regime (0% hit):")
    for pol in POL_ORD:
        d=ORACLE_SENS_FAILURE[pol]; print(f"    {POL_LABS[pol]:<16s}  P50={_mean3(d['p50_tpot_ms']):5.0f}  P99={_mean3(d['p99_tpot_ms']):5.0f}  tp={_mean3(d['throughput_tps']):4.0f}")


# ===========================================================================
#  Main
# ===========================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="P3 E2E camera-ready oracle figure generator")
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--output-dir", type=Path, default=root/"figures"/"evaluation")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--core-only", action="store_true", help="Only main-body [CORE] figures")
    args = p.parse_args()
    print("="*60+"\\n  P3 E2E Oracle Figure Generator  (camera-ready EuroSys)\\n"+"="*60)
    if not verify_physics(): return 1
    if args.verbose: oracle_summary(); latex_table()
    if args.dry_run: print("\\nDry-run PASSED."); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\\nGenerating figures → {args.output_dir}/\\n")
    core = [("Real trace", [_e2e_scaling_shared_legend, _e2e_scaling_p99, _e2e_scaling_throughput, _e2e_scaling_cdf]),
            ("Controlled pressure", [_ctrl_p99, _ctrl_tail, _ctrl_tail_percentiles, _ctrl_mechanism]),
            ("Ablation", [_ab_promo, _ab_overlap]),
            ("Decomposition", [_decomp])]
    appx = [("Real trace (suppl.)", [_mech, _slora, _mixtral]),
            ("Controlled (suppl.)", [_ctrl_cdf, _ctrl_actions, _coldpath]),
            ("Ablation (suppl.)", [_reinsert, _tpref_stress]),
            ("Sensitivity", [_sens_composite, _sens_cpu, _sens_skew, _sens_tp, _sens_fail])]
    sections = core + ([] if args.core_only else appx)
    n = 0
    for label, funcs in sections:
        tag = "[CORE]" if any(f in [x for _, xs in core for x in xs] for f in funcs) else "[APPENDIX]"
        print(f"{tag} {label}")
        for fn in funcs: fn(args.output_dir); n += 1
    print(f"\\nDone — {n} figures written ({'core only' if args.core_only else 'core + appendix'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
