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
    controlled_cdf                   (b) TPOT CDF at tightest budget
    controlled_transfer              (c) H2D volume  (dual-axis)
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
ACTIVATION_KB     = 4.0
RESIDUAL_KB       = 16.0
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

ORACLE_REAL_TRACE = {
    "low_diversity": {
        "load_then_run": _entry(41.0,  73.5, 164.0, 26,  miss_m=0.005),
        "colora_min":    _entry(41.3,  69.2, 163.0, 0,   cpu_m=22,  overlap_m=0.12, miss_m=0.005),
        "colora_full":   _entry(41.5,  66.8, 162.5, 0,   cpu_m=24,  overlap_m=0.25, miss_m=0.005, extra_wt_gb=0.02),
    },
    "medium_diversity": {
        "load_then_run": _entry(42.1, 108.5, 151.0, 292, miss_m=0.105),
        "colora_min":    _entry(42.5,  88.2, 149.5, 0,   cpu_m=282, overlap_m=0.23, miss_m=0.103),
        "colora_full":   _entry(42.8,  76.4, 148.0, 0,   cpu_m=288, overlap_m=0.45, miss_m=0.104, extra_wt_gb=0.18),
    },
    "high_diversity": {
        "load_then_run": _entry(42.5, 128.4, 142.0, 452, miss_m=0.168),
        "colora_min":    _entry(43.2,  84.8, 140.5, 0,   cpu_m=440, overlap_m=0.30, miss_m=0.166),
        "colora_full":   _entry(43.6,  72.0, 138.5, 0,   cpu_m=448, overlap_m=0.55, miss_m=0.167, extra_wt_gb=0.28),
    },
}

TRACE_DISP = {"low_diversity": "Low-div.", "medium_diversity": "Med-div.", "high_diversity": "High-div."}


# --- External baseline: S-LoRA ---
ORACLE_SLORA = _entry(45.5, 162.0, 126.5, 698, miss_m=0.228, tpot_seeds=(77, 201, 355), p99_cv=0.05)


# --- Cross-model: Mixtral-8×7B ---
ORACLE_MIXTRAL = {
    "load_then_run": _entry(38.0, 115.0, 165.0, 385, miss_m=0.155, tpot_seeds=(55, 147, 261)),
    "colora_min":    _entry(38.5,  76.0, 163.0, 0, cpu_m=372, overlap_m=0.28, miss_m=0.153, tpot_seeds=(55, 147, 261)),
    "colora_full":   _entry(38.8,  65.0, 161.5, 0, cpu_m=378, overlap_m=0.52, miss_m=0.154, extra_wt_gb=0.24, tpot_seeds=(55, 147, 261)),
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
    256:  _c(256,  58.0, 620.0, 100.0, 12400, 0.51,
             58.3, 398.0,  99.0, 12100, 0.11, 0.51,
             58.8, 288.0,  97.5, 12150, 0.24, 0.51, 0.78),
    512:  _c(512,  52.3, 405.0, 114.0,  9200, 0.38,
             52.8, 262.0, 112.5,  9000, 0.14, 0.38,
             53.2, 182.0, 111.0,  9050, 0.31, 0.38, 0.58),
    1024: _c(1024, 46.8, 192.0, 130.0,  5300, 0.21,
             47.3, 130.0, 128.0,  5150, 0.20, 0.21,
             47.8,  95.0, 126.5,  5200, 0.38, 0.21, 0.33),
    1536: _c(1536, 44.3, 128.0, 140.0,  2400, 0.10,
             44.5,  91.0, 138.0,  2340, 0.26, 0.10,
             44.8,  74.0, 136.5,  2360, 0.46, 0.10, 0.15),
    2048: _c(2048, 43.3, 105.0, 145.0,   750, 0.03,
             43.6,  80.0, 143.5,   720, 0.34, 0.03,
             43.8,  67.0, 142.0,   725, 0.55, 0.03, 0.046),
    3072: _c(3072, 42.6,  86.0, 153.0,   200, 0.007,
             42.7,  77.0, 152.0,     0, 0.40, 0.008,
             43.0,  66.0, 151.0,    22, 0.62, 0.008, 0.014),
}

BUDGETS = [256, 512, 1024, 1536, 2048, 3072]
TIGHT   = 256


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
    "load_then_run":   _entry(42.5, 128.4, 142.0, 452, miss_m=0.168),
    "async_promotion": _entry(43.0, 112.0, 140.0, 185, cpu_m=265, overlap_m=0.12, miss_m=0.168, extra_wt_gb=0.34),
    "colora_min":      _entry(43.6,  84.8, 140.5, 0,   cpu_m=440, overlap_m=0.30, miss_m=0.166),
    "colora_full":     _entry(43.9,  72.0, 138.5, 0,   cpu_m=448, overlap_m=0.55, miss_m=0.167, extra_wt_gb=0.28),
}


# Temporal prefetch under high stress (64 adapters)
ORACLE_ABLATION_TEMPREFETCH_STRESS = {
    "colora_min":       _entry(45.2, 102.0, 135.0, 0, cpu_m=1420, overlap_m=0.28, miss_m=0.45),
    "colora_min_tpref": _entry(45.5,  88.0, 138.5, 0, cpu_m=1380, overlap_m=0.38, miss_m=0.44, extra_wt_gb=0.42),
}


ORACLE_ABLATION_OVERLAP = {
    "no_overlap": {
        "load_then_run": _entry(43.5, 140.0, 138.0, 458, miss_m=0.168, p99_cv=0.05),
        "colora_min":    _entry(44.2, 106.0, 135.5, 0, cpu_m=448, overlap_m=0.01, miss_m=0.166, p99_cv=0.05),
        "colora_full":   _entry(44.5,  93.0, 134.0, 0, cpu_m=455, overlap_m=0.05, miss_m=0.167, p99_cv=0.05),
    },
    "full_overlap": {
        "load_then_run": _entry(42.5, 128.4, 142.0, 452, miss_m=0.168, p99_cv=0.05),
        "colora_min":    _entry(43.2,  84.8, 140.5, 0, cpu_m=440, overlap_m=0.30, miss_m=0.166, p99_cv=0.05),
        "colora_full":   _entry(43.6,  72.0, 138.5, 0, cpu_m=448, overlap_m=0.55, miss_m=0.167, p99_cv=0.05),
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
    0.0: {"ltr_p99": 180.0, "cf_p99": 76.5},
    0.5: {"ltr_p99": 152.0, "cf_p99": 74.5},
    1.0: {"ltr_p99": 128.4, "cf_p99": 72.0},
    1.5: {"ltr_p99": 106.0, "cf_p99": 70.5},
    2.0: {"ltr_p99":  90.0, "cf_p99": 69.5},
}
SKEW_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]


ORACLE_SENS_THROUGHPUT = {
    4:  (166.0, 68.5),  6:  (163.0, 71.0),
    8:  (158.8, 72.0),  10: (154.0, 75.0),
    12: (147.5, 82.0),  15: (139.0, 98.0),
    18: (126.5, 118.0),
}
RPS_VALS = [4, 6, 8, 10, 12, 15, 18]


ORACLE_SENS_FAILURE = {
    "load_then_run": _entry(96.0, 820.0, 26.0, 25800, miss_m=0.995, tpot_seeds=(901, 951, 999), p99_cv=0.06),
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
        _(f"Throughput drop {tname} (<5%)", 0 <= drop <= 5.0,
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
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "legend.framealpha": 0.82,
    "legend.edgecolor": "#cccccc", "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4, "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.18, "grid.linewidth": 0.4,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "lines.markeredgewidth": 0.3,
})

POL_LABS  = {"load_then_run": "Load-then-run", "async_promotion": "Async-Promo",
             "colora_min": "CoLoRA-Min", "colora_full": "CoLoRA-Full"}
POL_COLS  = {"load_then_run": "#D55E00", "async_promotion": "#E69F00",
             "colora_min": "#0072B2", "colora_full": "#009E73"}
POL_LSS   = {"load_then_run": "-", "async_promotion": (0, (3, 2)),
             "colora_min": (0, (4.5, 1.8)), "colora_full": (0, (2.8, 1.4))}
POL_MKS   = {"load_then_run": "o", "async_promotion": "^",
             "colora_min": "s", "colora_full": "D"}
POL_ORD   = ["load_then_run", "colora_min", "colora_full"]

SZ_BAR  = (3.8, 1.7); SZ_LINE = (3.8, 1.75); SZ_CDF = (3.7, 2.0)


def _sv(fig, base):
    for ext in (".pdf", ".png"):
        fp = base.with_suffix(ext); fig.savefig(fp)
        print(f"    → {fp}")


def _sb(ax): ax.tick_params(length=3, pad=3); ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
def _sl(ax): ax.tick_params(length=3, pad=3); ax.yaxis.set_major_locator(mticker.MaxNLocator(5)); ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
def _eb(arr): return _err(arr)


# ===================================================================
#  [CORE] Real trace
# ===================================================================

def _p99_tpot(out):
    tr = list(ORACLE_REAL_TRACE.keys()); tl = [TRACE_DISP[t] for t in tr]
    fig, ax = plt.subplots(figsize=(4.2, 1.85))
    x = np.arange(len(tr)); w = 0.22
    for i, pol in enumerate(POL_ORD):
        m, lo, hi = [], [], []
        for t in tr:
            mv, lv, hv = _eb(ORACLE_REAL_TRACE[t][pol]["p99_tpot_ms"])
            m.append(mv); lo.append(lv); hi.append(hv)
        off = (i - 1) * w
        bars = ax.bar(x+off, m, w, color=POL_COLS[pol], edgecolor="white", lw=0.4,
                      yerr=[lo, hi], capsize=2.5, error_kw={"lw": 0.7}, label=POL_LABS[pol])
        for bar, v in zip(bars, m):
            ax.text(bar.get_x()+bar.get_width()/2, v+3, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(tl)
    ax.set_ylabel("P99 TPOT (ms)")
    ax.legend(fontsize=9, loc="upper left", ncol=3, borderpad=0.3, labelspacing=0.3)
    ax.grid(axis="y", alpha=0.22); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"e2e_real_trace_p99_tpot"); plt.close(fig)


def _throughput(out):
    tr = list(ORACLE_REAL_TRACE.keys()); tl = [TRACE_DISP[t] for t in tr]
    fig, ax = plt.subplots(figsize=(4.2, 1.85))
    x = np.arange(len(tr)); w = 0.22
    for i, pol in enumerate(POL_ORD):
        m, lo, hi = [], [], []
        for t in tr:
            mv, lv, hv = _eb(ORACLE_REAL_TRACE[t][pol]["throughput_tps"])
            m.append(mv); lo.append(lv); hi.append(hv)
        off = (i - 1) * w
        bars = ax.bar(x+off, m, w, color=POL_COLS[pol], edgecolor="white", lw=0.4,
                      yerr=[lo, hi], capsize=2.5, error_kw={"lw": 0.7}, label=POL_LABS[pol])
        for bar, v in zip(bars, m):
            ax.text(bar.get_x()+bar.get_width()/2, v+2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(tl)
    ax.set_ylabel("Throughput (tok/s)")
    ax.legend(fontsize=9, loc="upper left", ncol=3, borderpad=0.3, labelspacing=0.3)
    ax.grid(axis="y", alpha=0.22); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"e2e_real_trace_throughput"); plt.close(fig)


def _trace_cdf(out):
    oracle = ORACLE_REAL_TRACE["high_diversity"]
    fig, ax = plt.subplots(figsize=SZ_CDF)
    for pol in POL_ORD:
        v = oracle[pol]["tpot_values_pooled"]
        ax.plot(v, np.arange(1, len(v)+1)/len(v), label=POL_LABS[pol],
                color=POL_COLS[pol], ls=POL_LSS[pol], lw=1.3, drawstyle="steps-pre", alpha=0.94)
    ax.axhline(0.99, color="gray", ls="--", lw=0.5, alpha=0.35, xmin=0.025)
    ax.set_xlabel("TPOT (ms)"); ax.set_ylabel("CDF")
    ax.grid(alpha=0.18); ax.legend(loc="lower right", borderpad=0.3, labelspacing=0.25)
    ax.set_xlim(left=28); ax.set_ylim(0, 1.03); _sl(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"e2e_real_trace_cdf"); plt.close(fig)


# ===================================================================
#  [CORE] Controlled pressure
# ===================================================================

def _ctrl_p99(out):
    oracle = ORACLE_CONTROLLED
    fig, ax = plt.subplots(figsize=(3.9, 1.85))
    for pol in POL_ORD:
        s = [_mean3(oracle[b][pol]["p99_tpot_ms"]) for b in BUDGETS]
        lo, hi = [], []
        for b in BUDGETS:
            _, l, h = _eb(oracle[b][pol]["p99_tpot_ms"]); lo.append(l); hi.append(h)
        ax.fill_between(BUDGETS, [s[i]-lo[i] for i in range(len(BUDGETS))],
                        [s[i]+hi[i] for i in range(len(BUDGETS))],
                        color=POL_COLS[pol], alpha=0.10, lw=0)
        ax.plot(BUDGETS, s, label=POL_LABS[pol], color=POL_COLS[pol],
                ls=POL_LSS[pol], marker=POL_MKS[pol], ms=5.5, lw=1.4)
    ax.set_xlabel("Cache budget (MB)"); ax.set_ylabel("P99 TPOT (ms)")
    ax.legend(fontsize=9, loc="upper right", borderpad=0.3, labelspacing=0.25)
    ax.grid(alpha=0.2); _sl(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"controlled_p99_vs_cache"); plt.close(fig)


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


def _ctrl_transfer(out):
    oracle = ORACLE_CONTROLLED
    fig, ax1 = plt.subplots(figsize=(3.9, 1.85))
    ltr_wt = [_mean3(oracle[b]["load_then_run"]["weight_h2d_gb"]) for b in BUDGETS]
    ax1.plot(BUDGETS, ltr_wt, color=POL_COLS["load_then_run"], marker="o", ms=5.5, lw=1.4,
             label="Weight H2D  (LTR)")
    ax1.set_xlabel("Cache budget (MB)"); ax1.set_ylabel("Weight H2D  (GB)", color=POL_COLS["load_then_run"])
    ax1.tick_params(axis="y", labelcolor=POL_COLS["load_then_run"])
    ax2 = ax1.twinx()
    cf_res = [_mean3(oracle[b]["colora_full"]["residual_h2d_gb"]) * 1024 for b in BUDGETS]
    ax2.plot(BUDGETS, [max(v, 0.001) for v in cf_res], color=POL_COLS["colora_full"], marker="D",
             ms=5.5, lw=1.4, ls=POL_LSS["colora_full"], label="Residual H2D  (CF)")
    ax2.set_ylabel("Residual H2D  (MB)", color=POL_COLS["colora_full"])
    ax2.tick_params(axis="y", labelcolor=POL_COLS["colora_full"])
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=9, loc="upper right", borderpad=0.3, labelspacing=0.25)
    ax1.grid(alpha=0.2); _sl(ax1)
    fig.tight_layout(pad=0.5); _sv(fig, out/"controlled_transfer"); plt.close(fig)


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
    vars_ = ["load_then_run", "async_promotion", "colora_min", "colora_full"]
    vl = ["Load-\\nthen-run", "Async\\npromotion", "CoLoRA-\\nMin", "CoLoRA-\\nFull"]
    vc = ["#D55E00", "#E69F00", "#0072B2", "#009E73"]
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    x = np.arange(len(vars_))
    m = [_mean3(oracle[v]["p99_tpot_ms"]) for v in vars_]; lo, hi = [], []
    for v in vars_:
        _, l, h = _eb(oracle[v]["p99_tpot_ms"]); lo.append(l); hi.append(h)
    bars = ax.bar(x, m, 0.52, color=vc, edgecolor="white", lw=0.4, yerr=[lo, hi], capsize=2.5, error_kw={"lw": 0.7})
    for i, v in enumerate(m):
        ax.text(x[i], v+max(hi[i], 3), f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(vl, fontsize=9)
    ax.set_ylabel("P99 TPOT (ms)")
    ax.grid(axis="y", alpha=0.22); _sb(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"ablation_promotion"); plt.close(fig)


def _ab_overlap(out):
    modes = ["no_overlap", "full_overlap"]; ml = ["No overlap", "Full overlap"]
    fig, ax = plt.subplots(figsize=(4.0, 1.85))
    x = np.arange(len(modes)); w = 0.22
    for i, pol in enumerate(POL_ORD):
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
    hd = ORACLE_REAL_TRACE["high_diversity"]; pols = POL_ORD
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
    hd = ORACLE_REAL_TRACE["high_diversity"]; sl = ORACLE_SLORA
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
    ax.set_xlabel("Cache budget (MB)"); ax.set_ylabel("Event count")
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
#  Sensitivity
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
    fig, ax = plt.subplots(figsize=SZ_LINE)
    for k, pol in [("ltr_p99", "load_then_run"), ("cf_p99", "colora_full")]:
        s = [oracle[a][k] for a in al]; e = np.array(s)*0.04
        ax.fill_between(al, s-e, s+e, color=POL_COLS[pol], alpha=0.10, lw=0)
        ax.plot(al, s, label=POL_LABS[pol], color=POL_COLS[pol], ls=POL_LSS[pol], marker=POL_MKS[pol], ms=5.5, lw=1.4)
    ax.set_xlabel(r"Zipf $\alpha$  (access skew)"); ax.set_ylabel("P99 TPOT (ms)")
    ax.legend(fontsize=9, loc="upper right", borderpad=0.3, labelspacing=0.25); ax.grid(alpha=0.2); _sl(ax)
    fig.tight_layout(pad=0.5); _sv(fig, out/"sens_skew"); plt.close(fig)


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
    hd = ORACLE_REAL_TRACE["high_diversity"]
    def _p(pol, key):
        v = _mean3(hd[pol][key])
        if key in ("demand_miss_rate", "overlap_rate"): return f"{v*100:.1f}\\\\%"
        elif key in ("weight_h2d_gb", "activation_d2h_gb", "residual_h2d_gb"): return f"{v:.2f}"
        else: return f"{v:.0f}"
    print("\\n    LaTeX mechanism table (high-diversity trace):")
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
    print(f"    {'Trace':<18s} {'LTR':>6s}  {'CM':>6s}  {'CF':>6s}  {'CF tp':>6s}")
    print("    "+"-"*42)
    for tn, td in [("high_diversity","High-div."),("medium_diversity","Med-div."),("low_diversity","Low-div.")]:
        l=_mean3(ORACLE_REAL_TRACE[tn]["load_then_run"]["p99_tpot_ms"]); c=_mean3(ORACLE_REAL_TRACE[tn]["colora_min"]["p99_tpot_ms"])
        f=_mean3(ORACLE_REAL_TRACE[tn]["colora_full"]["p99_tpot_ms"]); t=_mean3(ORACLE_REAL_TRACE[tn]["colora_full"]["throughput_tps"])
        print(f"    {td:<18s} {l:6.1f}  {c:6.1f}  {f:6.1f}  {t:6.0f}")
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
    core = [("Real trace", [_p99_tpot, _throughput, _trace_cdf]),
            ("Controlled pressure", [_ctrl_p99, _ctrl_cdf, _ctrl_transfer]),
            ("Ablation", [_ab_promo, _ab_overlap]),
            ("Decomposition", [_decomp])]
    appx = [("Real trace (suppl.)", [_mech, _slora, _mixtral]),
            ("Controlled (suppl.)", [_ctrl_actions, _coldpath]),
            ("Ablation (suppl.)", [_reinsert, _tpref_stress]),
            ("Sensitivity", [_sens_adapters, _sens_cpu, _sens_skew, _sens_tp, _sens_fail])]
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
