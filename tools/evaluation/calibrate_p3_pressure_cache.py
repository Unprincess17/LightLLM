#!/usr/bin/env python3
"""Calibrate cache budgets to joint-object miss-rate regimes for P3 pressure traces.

Simulates an LRU cache at **joint-object** granularity using the runtime's actual
cache key:

    (projection, adapter_idx, layer_id, expert_id)

Supports three routing modes:

  * ``synthetic_uniform`` – each token samples top-k experts uniformly at random.
  * ``synthetic_sticky``  – per-adapter preferred expert set with stickiness.
  * ``runtime_profiled``  – consume a real routing log (requires --runtime-routing-log).

Also computes an adapter-level proxy miss rate for comparison only.  Final
pressure labels always come from the joint-object miss rate.

Pressure regimes:

    low    -> 15-25% joint miss rate
    medium -> 35-50% joint miss rate
    high   -> >= 60% joint miss rate

Memory optimization: joint keys are packed into 32-bit ints and stored in
numpy arrays for fast vectorized generation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Packed key encoding (32-bit unsigned int)
# ---------------------------------------------------------------------------
# [proj_id:4][adapter_idx:12][layer_id:8][expert_id:8]
# Supports: 16 projections, 4096 adapters, 256 layers, 256 experts

PROJ_SHIFT = 28
ADAPTER_SHIFT = 16
LAYER_SHIFT = 8

PROJ_NAMES = ["moe_expert_gate", "moe_expert_up", "moe_expert_down"]
PROJ_IDS = {name: i for i, name in enumerate(PROJ_NAMES)}


def pack_key(proj: str, adapter_idx: int, layer_id: int, expert_id: int) -> int:
    p = PROJ_IDS.get(proj, 0)
    return (p << PROJ_SHIFT) | (adapter_idx << ADAPTER_SHIFT) | (layer_id << LAYER_SHIFT) | expert_id


def unpack_key(key: int) -> Tuple[str, int, int, int]:
    return (PROJ_NAMES[(key >> PROJ_SHIFT) & 0xF],
            (key >> ADAPTER_SHIFT) & 0xFFF,
            (key >> LAYER_SHIFT) & 0xFF,
            key & 0xFF)


def pack_adapter_key(adapter_idx: int) -> int:
    return adapter_idx


# ---------------------------------------------------------------------------
# Vectorized key generation with numpy
# ---------------------------------------------------------------------------

def build_keys_uniform(
    trace_rows: List[Dict],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    projections: List[str],
    seed: int,
) -> np.ndarray:
    """Build packed key array with uniform routing (fully vectorized).

    Total accesses per request = target_output_len * num_layers * len(projections) * top_k
    """
    rng = np.random.default_rng(seed)

    n_projs = len(projections)
    proj_ids = np.array([PROJ_IDS[p] for p in projections], dtype=np.uint32)

    # Group requests by target_output_len for vectorized batching
    # For simplicity, assume all have same target_output_len from the trace
    target_len = trace_rows[0].get("target_output_len", 64) if trace_rows else 64
    n_requests = len(trace_rows)

    # adapter indices
    adapter_indices = np.array(
        [_parse_adapter_idx(r["adapter_id"]) for r in trace_rows], dtype=np.uint32
    )

    # Grid dimensions: request × token × layer × projection × expert
    # Total keys: n_requests * target_len * num_layers * n_projs * top_k

    # Build layer and projection grids
    # layer_ids: shape (num_layers,), tiled across requests/tokens/projs/experts
    layer_grid = np.arange(num_layers, dtype=np.uint32)  # (L,)
    proj_grid = proj_ids  # (P,)

    # For each (request, token, layer): sample top_k experts uniformly
    n_rtl = n_requests * target_len * num_layers
    # Generate expert choices without replacement for each (r,t,l)
    # We'll generate a random permutation of [0..num_experts) for each (r,t,l) and take first top_k
    # This is expensive for large n_rtl, so we use a different approach:
    # For each (r,t,l), independently sample top_k from [0, num_experts)
    # Using the algorithm: generate top_k random numbers and argsort

    # Actually, for uniform random without replacement, we can use:
    # For each row, generate top_k columns from rng.choice with replace=False
    # But that requires a loop. Instead, use the sorted-indices trick:
    # Generate (n_rtl, num_experts) random values, argsort each row, take first top_k

    # This is too memory-heavy for n_rtl * num_experts. Instead, use a chunked approach.
    chunk_size = 10_000  # Process in chunks of 10K (request,token,layer) tuples
    expert_all = np.empty((n_rtl, top_k), dtype=np.uint32)

    for start in range(0, n_rtl, chunk_size):
        end = min(start + chunk_size, n_rtl)
        size = end - start
        # Generate random matrix and take top_k by argsort
        rand_mat = rng.random(size=(size, num_experts))
        top_k_idx = np.argpartition(rand_mat, top_k, axis=1)[:, :top_k]
        expert_all[start:end] = top_k_idx.astype(np.uint32)

    # Now build the full key array using broadcasting
    # Dimensions: (R, T, L, P, K) -> flattened
    R, T, L, P, K = n_requests, target_len, num_layers, n_projs, top_k

    # adapter_idx: (R, 1, 1, 1, 1) -> broadcast to (R, T, L, P, K)
    adapter_grid = adapter_indices.reshape(R, 1, 1, 1, 1)

    # layer_id: (1, 1, L, 1, 1)
    layer_grid_5d = layer_grid.reshape(1, 1, L, 1, 1)

    # proj_id: (1, 1, 1, P, 1)
    proj_grid_5d = proj_grid.reshape(1, 1, 1, P, 1)

    # expert_id: (R*T*L, K) -> reshape to (R, T, L, 1, K)
    expert_5d = expert_all.reshape(R, T, L, 1, K)

    # Broadcast and pack
    # packed = proj << 28 | adapter << 16 | layer << 8 | expert
    packed = (proj_grid_5d.astype(np.uint32) << PROJ_SHIFT) | \
             (adapter_grid << ADAPTER_SHIFT) | \
             (layer_grid_5d << LAYER_SHIFT) | \
             expert_5d

    return packed.ravel()


def build_keys_sticky(
    trace_rows: List[Dict],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    projections: List[str],
    sticky_prob: float,
    preferred_experts_per_layer: int,
    seed: int,
) -> np.ndarray:
    """Build packed key array with sticky routing (vectorized).

    Per-adapter preferred expert sets with stickiness.
    """
    rng = np.random.default_rng(seed)
    n_projs = len(projections)
    proj_ids = np.array([PROJ_IDS[p] for p in projections], dtype=np.uint32)

    target_len = trace_rows[0].get("target_output_len", 64) if trace_rows else 64
    n_requests = len(trace_rows)
    adapter_indices = np.array(
        [_parse_adapter_idx(r["adapter_id"]) for r in trace_rows], dtype=np.uint32
    )

    # Pre-compute preferred expert sets per unique adapter per layer
    unique_adapters = np.unique(adapter_indices)
    # preferred[adapter_idx] = (num_layers, preferred_experts_per_layer) array
    preferred: Dict[int, np.ndarray] = {}
    for aidx in unique_adapters:
        preferred[int(aidx)] = np.stack([
            rng.choice(num_experts, size=min(preferred_experts_per_layer, num_experts), replace=False)
            for _ in range(num_layers)
        ]).astype(np.uint32)

    # For sticky routing, we process per-adapter because different adapters
    # have different preferred sets. Process adapters in batches.
    all_keys_list: List[np.ndarray] = []

    for aidx in unique_adapters:
        aidx = int(aidx)
        # How many requests for this adapter?
        mask = adapter_indices == aidx
        n_req = int(mask.sum())
        if n_req == 0:
            continue

        pref_experts = preferred[aidx]  # (L, preferred_per_layer)

        R, T, L, P, K = n_req, target_len, num_layers, n_projs, top_k

        # Decide sticky vs uniform per (request, token)
        is_sticky = rng.random(size=(R, T)) < sticky_prob  # (R, T)

        # For each (request, token, layer), select experts
        # If sticky: sample from preferred[aidx][layer]
        # If not sticky: sample uniformly from [0, num_experts)

        # Generate all experts at once:
        # Sticky experts: for each (r,t,l), if sticky, sample from preferred
        # Non-sticky: uniform random

        # Build expert array: (R, T, L, K)
        experts = np.empty((R, T, L, K), dtype=np.uint32)

        # First, generate all uniform experts
        n_rtl = R * T * L
        # Uniform: random top-k without replacement
        chunk_size = 10_000
        uniform_experts = np.empty((n_rtl, K), dtype=np.uint32)
        for start in range(0, n_rtl, chunk_size):
            end = min(start + chunk_size, n_rtl)
            size = end - start
            rand_mat = rng.random(size=(size, num_experts))
            top_k_idx = np.argpartition(rand_mat, K, axis=1)[:, :K]
            uniform_experts[start:end] = top_k_idx.astype(np.uint32)

        # Then, generate sticky experts (sample from preferred sets)
        # For each layer, preferred[aidx][layer] gives the candidate experts
        sticky_experts = np.empty((R, T, L, K), dtype=np.uint32)
        for li in range(L):
            n_rt = R * T
            # Sample K from preferred[aidx][li] for each (r,t)
            pref_set = pref_experts[li]  # (preferred_per_layer,)
            rand_idx = rng.random(size=(n_rt, len(pref_set)))
            sel_idx = np.argpartition(rand_idx, K, axis=1)[:, :K]
            sticky_experts[:, :, li, :] = pref_set[sel_idx].reshape(R, T, K)

        # Mix sticky and uniform based on is_sticky
        is_sticky_exp = is_sticky[:, :, np.newaxis, np.newaxis]  # (R, T, 1, 1)
        experts = np.where(is_sticky_exp, sticky_experts, uniform_experts.reshape(R, T, L, K))

        # Build packed keys: (R, T, L, P, K)
        adapter_grid = np.uint32(aidx)
        layer_grid = np.arange(L, dtype=np.uint32).reshape(1, 1, L, 1, 1)
        proj_grid = proj_ids.reshape(1, 1, 1, P, 1)
        expert_5d = experts.reshape(R, T, L, 1, K)

        packed = (proj_grid.astype(np.uint32) << PROJ_SHIFT) | \
                 (adapter_grid << ADAPTER_SHIFT) | \
                 (layer_grid << LAYER_SHIFT) | \
                 expert_5d

        all_keys_list.append(packed.ravel())

    return np.concatenate(all_keys_list) if all_keys_list else np.array([], dtype=np.uint32)


def build_keys_profiled(
    runtime_routing_log: Path,
    *,
    projections: List[str],
) -> np.ndarray:
    """Build packed key array from runtime routing log."""
    keys_list: List[int] = []
    with runtime_routing_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            adapter_idx = entry.get("adapter_idx", 0)
            if "adapter_id" in entry and "adapter_idx" not in entry:
                adapter_idx = _parse_adapter_idx(entry["adapter_id"])
            layer_id = entry["layer_id"]
            expert_id = entry["expert_id"]

            if "projection" in entry and entry["projection"]:
                projs = [entry["projection"]]
            else:
                projs = projections

            for proj in projs:
                keys_list.append(pack_key(proj, adapter_idx, layer_id, expert_id))

    return np.array(keys_list, dtype=np.uint32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_adapter_idx(adapter_id: str) -> int:
    digits = []
    for ch in reversed(adapter_id):
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if digits:
        return int("".join(reversed(digits)))
    return hash(adapter_id) % (10 ** 6)


# ---------------------------------------------------------------------------
# LRU cache simulator (packed int keys)
# ---------------------------------------------------------------------------

class LRUCacheSim:
    """LRU cache simulator keyed by packed int keys."""

    def __init__(self, capacity: int):
        self.capacity = max(capacity, 0)
        self._cache: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, key: int) -> bool:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        if self.capacity > 0:
            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)
            self._cache[key] = None
        return False

    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total else 0.0


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# LRU simulation
# ---------------------------------------------------------------------------

def simulate_joint_lru(
    warmup_keys: np.ndarray,
    measurement_keys: np.ndarray,
    cache_capacity: int,
) -> Dict:
    """Run LRU simulation on packed joint keys with adapter proxy."""
    cache = LRUCacheSim(cache_capacity)
    for k in warmup_keys:
        cache.access(int(k))

    cache.hits = 0
    cache.misses = 0

    for k in measurement_keys:
        cache.access(int(k))
    joint_hits = cache.hits
    joint_misses = cache.misses
    joint_total = joint_hits + joint_misses
    joint_miss_rate = joint_misses / joint_total if joint_total else 0.0

    # Adapter proxy LRU
    warmup_adapter = (warmup_keys >> ADAPTER_SHIFT) & 0xFFF
    meas_adapter = (measurement_keys >> ADAPTER_SHIFT) & 0xFFF

    adapter_cache = LRUCacheSim(cache_capacity)
    for k in warmup_adapter:
        adapter_cache.access(int(k))
    adapter_cache.hits = 0
    adapter_cache.misses = 0
    for k in meas_adapter:
        adapter_cache.access(int(k))

    ap_hits = adapter_cache.hits
    ap_misses = adapter_cache.misses
    ap_total = ap_hits + ap_misses
    ap_miss_rate = ap_misses / ap_total if ap_total else 0.0

    return {
        "joint_hits": joint_hits,
        "joint_misses": joint_misses,
        "joint_miss_rate": joint_miss_rate,
        "adapter_proxy_hits": ap_hits,
        "adapter_proxy_misses": ap_misses,
        "adapter_proxy_miss_rate": ap_miss_rate,
    }


def compute_compulsory_miss_rate(keys: np.ndarray) -> float:
    unique = len(np.unique(keys))
    return unique / len(keys) if len(keys) else 0.0


def compute_reuse_distance_stats(keys: np.ndarray, sample_size: int = 200_000) -> Tuple[int, int, int]:
    """Compute reuse distance percentiles. Samples for large streams."""
    if len(keys) == 0:
        return 0, 0, 0

    # Sample indices for efficiency
    if len(keys) > sample_size:
        sample_idx = np.sort(np.random.default_rng(0).choice(len(keys), size=sample_size, replace=False))
        sampled = keys[sample_idx]
    else:
        sampled = keys

    # Build position map: key -> last position
    last_pos: Dict[int, int] = {}
    distances: List[int] = []

    for i, k in enumerate(sampled):
        k = int(k)
        if k in last_pos:
            # Count unique keys between last_pos[k] and i in the sampled array
            segment = set(sampled[last_pos[k] + 1:i].tolist())
            distances.append(len(segment))
        last_pos[k] = i

    if not distances:
        return 0, 0, 0

    sd = sorted(distances)
    n = len(sd)
    return sd[int(n * 0.50)], sd[int(n * 0.90)], sd[int(n * 0.99)]


# ---------------------------------------------------------------------------
# Object size formula
# ---------------------------------------------------------------------------

def compute_object_size_bytes(
    *,
    rank: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype_bytes: int = 2,
    projection: str = "gate",
) -> int:
    if projection in ("moe_expert_gate", "moe_expert_up", "gate", "up"):
        input_dim = hidden_dim
        output_dim = intermediate_dim
    else:
        input_dim = intermediate_dim
        output_dim = hidden_dim
    return rank * (input_dim + output_dim) * dtype_bytes


def mb_to_objects(mb: float, object_size_bytes: int) -> int:
    if object_size_bytes <= 0:
        return 0
    return int(mb * 1024 * 1024 // object_size_bytes)


# ---------------------------------------------------------------------------
# Pressure classification
# ---------------------------------------------------------------------------

def classify_pressure(miss_rate: float) -> Optional[str]:
    if 0.15 <= miss_rate <= 0.25:
        return "low"
    if 0.35 <= miss_rate <= 0.50:
        return "medium"
    if miss_rate >= 0.60:
        return "high"
    return None


# ---------------------------------------------------------------------------
# Budget sweep
# ---------------------------------------------------------------------------

def sweep_budgets(
    warmup_keys: np.ndarray,
    measurement_keys: np.ndarray,
    cache_budgets_objects: List[int],
    trace_name: str,
    routing_mode: str,
    diversity: int,
    overlap_fraction: float,
    length_mode: str,
    decode_target_tokens: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    projections: str,
    object_size_bytes: int,
    object_size_assumptions: str,
    seed: int,
) -> List[Dict]:
    unique_joint = len(np.unique(measurement_keys))
    unique_adapters = len(np.unique((measurement_keys >> ADAPTER_SHIFT) & 0xFFF))
    compulsory_rate = compute_compulsory_miss_rate(measurement_keys)
    rd_p50, rd_p90, rd_p99 = compute_reuse_distance_stats(measurement_keys)

    rows: List[Dict] = []
    for budget in cache_budgets_objects:
        result = simulate_joint_lru(warmup_keys, measurement_keys, budget)
        budget_mb = budget * object_size_bytes / (1024 * 1024) if object_size_bytes > 0 else 0.0
        pressure = classify_pressure(result["joint_miss_rate"])

        rows.append({
            "trace_name": trace_name,
            "routing_mode": routing_mode,
            "diversity": diversity,
            "overlap_fraction": overlap_fraction,
            "length_mode": length_mode,
            "decode_target_tokens": decode_target_tokens,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "projections": projections,
            "cache_budget_mb": round(budget_mb, 2),
            "cache_budget_objects": budget,
            "warmup_accesses": len(warmup_keys),
            "measurement_accesses": len(measurement_keys),
            "joint_hits": result["joint_hits"],
            "joint_misses": result["joint_misses"],
            "joint_miss_rate": result["joint_miss_rate"],
            "adapter_proxy_hits": result["adapter_proxy_hits"],
            "adapter_proxy_misses": result["adapter_proxy_misses"],
            "adapter_proxy_miss_rate": result["adapter_proxy_miss_rate"],
            "unique_joint_objects": unique_joint,
            "unique_adapters": unique_adapters,
            "compulsory_joint_miss_rate": compulsory_rate,
            "reuse_distance_p50": rd_p50,
            "reuse_distance_p90": rd_p90,
            "reuse_distance_p99": rd_p99,
            "pressure_level": pressure or "",
            "selected_for_e2e": False,
            "object_size_bytes": object_size_bytes,
            "object_size_assumptions": object_size_assumptions,
            "seed": seed,
        })
    return rows


# ---------------------------------------------------------------------------
# Budget selection
# ---------------------------------------------------------------------------

def select_budgets(rows: List[Dict]) -> List[Dict]:
    regime_centers = {"low": 0.20, "medium": 0.425, "high": 0.70}
    selected: Dict[str, Optional[Dict]] = {k: None for k in regime_centers}

    for row in rows:
        level = row.get("pressure_level", "")
        if level and level in regime_centers:
            center = regime_centers[level]
            cur = selected[level]
            if cur is None or abs(row["joint_miss_rate"] - center) < abs(cur["joint_miss_rate"] - center):
                selected[level] = row

    for srow in selected.values():
        if srow is not None:
            srow["selected_for_e2e"] = True

    return rows


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    measurement_keys: np.ndarray,
    all_rows: List[Dict],
) -> List[str]:
    errors: List[str] = []

    # Check 1: zero-cache -> miss rate ≈ 1.0
    zero_rows = [r for r in all_rows if r["cache_budget_objects"] == 0]
    if zero_rows:
        for r in zero_rows:
            if r["joint_miss_rate"] < 0.99:
                errors.append(
                    f"Check 1 FAIL: zero-cache joint_miss_rate={r['joint_miss_rate']:.4f} "
                    f"(expected ~1.0) for {r['trace_name']}/{r['routing_mode']}"
                )
    else:
        errors.append("Check 1 FAIL: no zero-cache row found in sweep")

    # Check 2: full-cache -> miss rate ≈ compulsory
    unique_joint = len(np.unique(measurement_keys))
    full_cache_rows = [r for r in all_rows if r["cache_budget_objects"] >= unique_joint]
    if full_cache_rows:
        for r in full_cache_rows:
            diff = abs(r["joint_miss_rate"] - r["compulsory_joint_miss_rate"])
            if diff > 0.05:
                errors.append(
                    f"Check 2 FAIL: full-cache miss_rate={r['joint_miss_rate']:.4f} "
                    f"vs compulsory={r['compulsory_joint_miss_rate']:.4f} "
                    f"(diff={diff:.4f}) for {r['trace_name']}/{r['routing_mode']}. "
                    f"Possible phase/key/budget bug."
                )
    else:
        errors.append(
            f"Check 2 FAIL: no budget >= unique_joint_objects={unique_joint} in sweep. "
            f"Add larger budgets."
        )

    # Check 4: projection in joint keys
    if len(measurement_keys) > 0:
        sample = measurement_keys[:100]
        for k in sample:
            pid = (int(k) >> PROJ_SHIFT) & 0xF
            if pid >= len(PROJ_NAMES):
                errors.append(f"Check 4 FAIL: invalid projection_id={pid} in packed key")
                break

    # Check 5: selected high pressure >= 60%
    selected_high = [r for r in all_rows if r.get("selected_for_e2e") and r["pressure_level"] == "high"]
    if not selected_high or all(r["joint_miss_rate"] < 0.60 for r in selected_high):
        errors.append("Check 5 FAIL: no selected high-pressure row with joint_miss_rate >= 60%")

    return errors


# ---------------------------------------------------------------------------
# Write compressed access stream (optional)
# ---------------------------------------------------------------------------

def write_keys_to_gz(
    keys: np.ndarray,
    gz_path: Path,
    sample_path: Path,
    sample_size: int = 5000,
    phase: str = "measurement",
) -> None:
    """Write packed keys to compressed JSONL.gz and a small sample."""
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for k in keys:
            proj, adapter_idx, layer_id, expert_id = unpack_key(int(k))
            f.write(json.dumps({
                "phase": phase, "projection": proj,
                "adapter_idx": adapter_idx, "layer_id": layer_id,
                "expert_id": expert_id,
            }, ensure_ascii=True) + "\n")
            count += 1
            if count % 2_000_000 == 0:
                print(f"    ... {count:,} keys written", flush=True)
    print(f"  Wrote {gz_path} ({count:,} keys)")

    sample = keys[:sample_size] if len(keys) > sample_size else keys
    with sample_path.open("w", encoding="utf-8") as f:
        for k in sample:
            proj, adapter_idx, layer_id, expert_id = unpack_key(int(k))
            f.write(json.dumps({
                "phase": phase, "projection": proj,
                "adapter_idx": adapter_idx, "layer_id": layer_id,
                "expert_id": expert_id,
            }, ensure_ascii=True) + "\n")
    print(f"  Wrote {sample_path} ({len(sample)} sample rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path,
                    default=REPO_ROOT / "artifacts" / "evaluation" / "p3_traces")
    p.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "artifacts" / "evaluation" / "p3")

    p.add_argument("--routing-mode",
                    choices=["synthetic_uniform", "synthetic_sticky", "runtime_profiled"],
                    default="synthetic_sticky")

    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--projections", default="moe_expert_gate,moe_expert_up,moe_expert_down",
                    help="Comma-separated projection names (must match runtime)")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--sticky-prob", type=float, default=0.8)
    p.add_argument("--preferred-experts-per-layer", type=int, default=16)

    p.add_argument("--runtime-routing-log", type=Path, default=None)

    p.add_argument("--cache-budget-objects", default="128,512,2048,8192,32768,65536,131072",
                    help="Comma-separated cache budget in objects")
    p.add_argument("--cache-budget-mb", default=None,
                    help="Comma-separated cache budget in MB (alternative)")

    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=4096)
    p.add_argument("--intermediate-dim", type=int, default=11008)
    p.add_argument("--dtype-bytes", type=int, default=2)

    p.add_argument("--diversities", default=None,
                    help="Comma-separated diversity values to process")
    p.add_argument("--write-stream", action="store_true", default=False,
                    help="Write compressed JSONL access stream (slow)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Discover traces
# ---------------------------------------------------------------------------

def discover_traces(trace_dir: Path, diversities: Optional[List[int]]) -> List[Tuple[str, int, Path, Path]]:
    traces: List[Tuple[str, int, Path, Path]] = []

    for meas_path in sorted(trace_dir.glob("*_div*_measurement.jsonl")):
        name = meas_path.stem.replace("_measurement", "")
        parts = name.split("_div")
        if len(parts) != 2:
            continue
        trace_type = parts[0]
        try:
            div = int(parts[1])
        except ValueError:
            continue
        if diversities and div not in diversities:
            continue
        warmup_path = trace_dir / f"{trace_type}_div{div}_warmup.jsonl"
        if warmup_path.exists():
            traces.append((name, div, warmup_path, meas_path))

    if not traces:
        for trace_type in ["pressure_disjoint", "pressure_low_overlap", "pressure_mixed_overlap"]:
            warmup_path = trace_dir / f"{trace_type}_warmup.jsonl"
            meas_path = trace_dir / f"{trace_type}_measurement.jsonl"
            if warmup_path.exists() and meas_path.exists():
                traces.append((trace_type, 0, warmup_path, meas_path))

    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    projections = [p.strip() for p in args.projections.split(",")]

    # Projection sanity check
    allowed_projections = {"moe_expert_gate", "moe_expert_up", "moe_expert_down"}
    if not set(projections).issubset(allowed_projections):
        raise ValueError(
            f"Projection names must match runtime LoRATargetType identifiers. "
            f"Got {projections}, allowed {sorted(allowed_projections)}. "
            f"Use --projections moe_expert_gate,moe_expert_up,moe_expert_down"
        )

    diversities = (
        [int(d.strip()) for d in args.diversities.split(",")]
        if args.diversities else None
    )

    # Object size
    obj_sizes = {}
    for proj in PROJ_NAMES:
        obj_sizes[proj] = compute_object_size_bytes(
            rank=args.rank, hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            dtype_bytes=args.dtype_bytes, projection=proj,
        )
    avg_obj_size = sum(obj_sizes.values()) // len(PROJ_NAMES)
    obj_assumptions = (
        f"rank={args.rank}, hidden_dim={args.hidden_dim}, "
        f"intermediate_dim={args.intermediate_dim}, dtype_bytes={args.dtype_bytes}, "
        f"gate/up={obj_sizes[PROJ_NAMES[0]]}B, down={obj_sizes[PROJ_NAMES[2]]}B, avg={avg_obj_size}B"
    )

    # Budgets
    if args.cache_budget_mb:
        budget_mbs = [float(x.strip()) for x in args.cache_budget_mb.split(",")]
        budget_objects = [mb_to_objects(mb, avg_obj_size) for mb in budget_mbs]
    else:
        budget_objects = [int(x.strip()) for x in args.cache_budget_objects.split(",")]

    if 0 not in budget_objects:
        budget_objects.insert(0, 0)

    # Discover traces
    traces = discover_traces(trace_dir, diversities)
    if not traces:
        print("ERROR: No trace files found", file=sys.stderr)
        return 1

    all_rows: List[Dict] = []
    high_pressure_reached = False

    for trace_name, div, warmup_path, meas_path in traces:
        print(f"\n--- {trace_name} (div={div}) ---", flush=True)

        warmup_rows = load_trace(warmup_path)
        measurement_rows = load_trace(meas_path)
        print(f"  Warmup: {len(warmup_rows)} requests", flush=True)
        print(f"  Measurement: {len(measurement_rows)} requests", flush=True)

        # Meta params
        overlap_fraction = 0.0
        length_mode = "fixed"
        decode_target = 64
        meta_path = trace_dir / "pressure_trace_meta.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)
            for entry in meta.get("traces", []):
                if entry.get("trace_name") == trace_name:
                    overlap_fraction = entry.get("overlap_fraction", 0.0)
                    length_mode = entry.get("length_mode", "fixed")
                    decode_target = entry.get("decode_target_tokens", 64)
                    break

        # Build keys
        print(f"  Building access stream (routing={args.routing_mode}) ...", flush=True)

        if args.routing_mode == "synthetic_uniform":
            warmup_keys = build_keys_uniform(
                warmup_rows, num_layers=args.num_layers,
                num_experts=args.num_experts, top_k=args.top_k,
                projections=projections, seed=args.seed,
            )
            measurement_keys = build_keys_uniform(
                measurement_rows, num_layers=args.num_layers,
                num_experts=args.num_experts, top_k=args.top_k,
                projections=projections, seed=args.seed + 1,
            )
        elif args.routing_mode == "synthetic_sticky":
            warmup_keys = build_keys_sticky(
                warmup_rows, num_layers=args.num_layers,
                num_experts=args.num_experts, top_k=args.top_k,
                projections=projections,
                sticky_prob=args.sticky_prob,
                preferred_experts_per_layer=args.preferred_experts_per_layer,
                seed=args.seed,
            )
            measurement_keys = build_keys_sticky(
                measurement_rows, num_layers=args.num_layers,
                num_experts=args.num_experts, top_k=args.top_k,
                projections=projections,
                sticky_prob=args.sticky_prob,
                preferred_experts_per_layer=args.preferred_experts_per_layer,
                seed=args.seed + 1,
            )
        elif args.routing_mode == "runtime_profiled":
            if not args.runtime_routing_log or not args.runtime_routing_log.exists():
                print("ERROR: --runtime-routing-log required", file=sys.stderr)
                return 1
            all_keys = build_keys_profiled(
                args.runtime_routing_log, projections=projections,
            )
            split = len(warmup_rows)
            total = len(warmup_rows) + len(measurement_rows)
            frac = split / total if total else 0
            split_idx = int(len(all_keys) * frac)
            warmup_keys = all_keys[:split_idx]
            measurement_keys = all_keys[split_idx:]
        else:
            return 1

        print(f"  Warmup keys: {len(warmup_keys):,}", flush=True)
        print(f"  Measurement keys: {len(measurement_keys):,}", flush=True)

        # Write compressed stream (optional)
        if args.write_stream:
            stream_label = f"{trace_name}_{args.routing_mode}"
            gz_path = output_dir / f"joint_access_stream_{stream_label}.jsonl.gz"
            sample_path = output_dir / f"joint_access_stream_{stream_label}_sample.jsonl"
            all_k = np.concatenate([warmup_keys, measurement_keys])
            write_keys_to_gz(all_k, gz_path, sample_path)
        else:
            # Always write a small sample
            stream_label = f"{trace_name}_{args.routing_mode}"
            sample_path = output_dir / f"joint_access_stream_{stream_label}_sample.jsonl"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample = measurement_keys[:5000] if len(measurement_keys) > 5000 else measurement_keys
            with sample_path.open("w", encoding="utf-8") as f:
                for k in sample:
                    proj, adapter_idx, layer_id, expert_id = unpack_key(int(k))
                    f.write(json.dumps({
                        "phase": "measurement", "projection": proj,
                        "adapter_idx": adapter_idx, "layer_id": layer_id,
                        "expert_id": expert_id,
                    }, ensure_ascii=True) + "\n")
            print(f"  Wrote {sample_path} ({len(sample)} sample rows)", flush=True)

        # Full-cache sanity budget
        unique_joint = len(np.unique(measurement_keys))
        max_budget_needed = unique_joint + 100

        # Add intermediate budgets for finer granularity
        # Log-spaced from 2048 to unique_joint, plus linear steps near 2048-8192
        log_min = 11  # 2^11 = 2048
        log_max = int(np.ceil(np.log2(max(unique_joint, 2)))) + 1
        log_spaced = [2**i for i in range(log_min, log_max + 1) if 2**i <= max_budget_needed]
        # Finer steps between 2K and 8K (where low/medium regime often sits)
        fine_steps = list(range(512, 8192, 256))
        sweep_list = sorted(set(budget_objects + log_spaced + fine_steps + [max_budget_needed]))

        print(f"  Sweeping {len(sweep_list)} budgets (unique_joint={unique_joint:,}) ...", flush=True)
        rows = sweep_budgets(
            warmup_keys, measurement_keys,
            cache_budgets_objects=sweep_list,
            trace_name=trace_name,
            routing_mode=args.routing_mode,
            diversity=div,
            overlap_fraction=overlap_fraction,
            length_mode=length_mode,
            decode_target_tokens=decode_target,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            projections=args.projections,
            object_size_bytes=avg_obj_size,
            object_size_assumptions=obj_assumptions,
            seed=args.seed,
        )

        rows = select_budgets(rows)
        all_rows.extend(rows)

        for r in rows:
            if r.get("selected_for_e2e"):
                print(f"  Selected {r['pressure_level']}: budget={r['cache_budget_objects']:,} "
                      f"({r['cache_budget_mb']:.1f} MB), joint_miss={r['joint_miss_rate']:.4f}, "
                      f"adapter_proxy_miss={r['adapter_proxy_miss_rate']:.4f}", flush=True)
                if r["pressure_level"] == "high" and r["joint_miss_rate"] >= 0.60:
                    high_pressure_reached = True

    # CSV
    fieldnames = [
        "trace_name", "routing_mode", "diversity", "overlap_fraction",
        "length_mode", "decode_target_tokens",
        "num_layers", "num_experts", "top_k", "projections",
        "cache_budget_mb", "cache_budget_objects",
        "warmup_accesses", "measurement_accesses",
        "joint_hits", "joint_misses", "joint_miss_rate",
        "adapter_proxy_hits", "adapter_proxy_misses", "adapter_proxy_miss_rate",
        "unique_joint_objects", "unique_adapters",
        "compulsory_joint_miss_rate",
        "reuse_distance_p50", "reuse_distance_p90", "reuse_distance_p99",
        "pressure_level", "selected_for_e2e",
        "object_size_bytes", "object_size_assumptions",
        "seed",
    ]
    csv_path = output_dir / "pressure_cache_calibration.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nWrote {csv_path} ({len(all_rows)} rows)", flush=True)

    # Sanity checks
    print("\n--- Sanity Checks ---", flush=True)
    check_errors = run_sanity_checks(measurement_keys, all_rows)
    if check_errors:
        for err in check_errors:
            print(f"  {err}", file=sys.stderr)
        fatal = [e for e in check_errors if "FAIL" in e]
        if fatal:
            print(f"\nFAIL: {len(fatal)} sanity check(s) failed.", file=sys.stderr)
            return 1
    else:
        print("  All sanity checks PASSED", flush=True)

    if not high_pressure_reached:
        print("\nFAIL: No selected configuration reached >= 60% joint miss rate.", file=sys.stderr)
        return 1

    print("\nPASS: High-pressure regime (>= 60% joint miss rate) confirmed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
